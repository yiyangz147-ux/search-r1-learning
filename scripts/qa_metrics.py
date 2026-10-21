"""Dataset-specific QA metrics, checked against pinned official evaluator sources.

The pure exact-match function follows upstream normalization, including its
empty-normalized-string edge cases. ``score_answer`` additionally rejects a raw
empty prediction, because this project's generation protocol requires an answer.
NQ-open's official primary score is EM; its token F1 here is only auxiliary.
See tests/fixtures/official_qa_metrics/NOTICE.md for attribution and provenance.
"""

from collections import Counter
import re
import string
import unicodedata


METRIC_VERSION = "official-qa-v1"
_SOURCE_URLS = {
    "nq": "https://github.com/google-research/language/blob/e2b407d154bdbfa4a0718e68cefa4aa1a28c2471/language/orqa/utils/eval_utils.py",
    "hotpotqa": "https://github.com/hotpotqa/hotpot/blob/fa3a36370899e1d85822de61e58c85ea19993154/hotpot_evaluate_v1.py",
    "search-r1": "https://github.com/PeterGriffinJin/Search-R1/blob/068516be64388adf17734628eb84ba768934f720/verl/utils/reward_score/qa_em.py",
}
_PROFILE_NAMES = {
    "nq": "nq-open-orqa",
    "hotpotqa": "hotpotqa-official-answer",
    "search-r1": "search-r1-normalized-em",
}
_BOOLEAN_ANSWERS = frozenset(("yes", "no", "noanswer"))
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _profile(data_source):
    """Resolve known source names without silently mislabelling unknown datasets."""
    if data_source is None:
        return "search-r1"
    key = data_source.strip().lower().replace("_", "-")
    if key in ("nq", "nq-open", "nq-search", "natural-questions"):
        return "nq"
    if key in ("hotpotqa", "hotpot-qa", "hotpotqa-search"):
        return "hotpotqa"
    if key in ("search-r1", "searchr1"):
        return "search-r1"
    raise ValueError(f"Unsupported QA metric data_source: {data_source}")


def _gold_answers(gold):
    if isinstance(gold, str):
        return [gold]
    return list(gold)


def normalize_answer(text, data_source=None):
    """Apply official source-specific normalization; NFD is specific to NQ-open."""
    if _profile(data_source) == "nq":
        text = unicodedata.normalize("NFD", text)
    text = text.lower().translate(_PUNCTUATION_TABLE)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match_score(prediction, ground_truth, data_source=None):
    """Pure upstream-compatible normalized equality, including empty strings."""
    return normalize_answer(prediction, data_source) == normalize_answer(ground_truth, data_source)


def token_f1_score(prediction, ground_truth, data_source=None):
    """Official HotpotQA answer F1, or auxiliary token overlap for other sources."""
    profile = _profile(data_source)
    prediction = normalize_answer(prediction, profile)
    ground_truth = normalize_answer(ground_truth, profile)
    if profile == "hotpotqa" and prediction != ground_truth:
        if prediction in _BOOLEAN_ANSWERS or ground_truth in _BOOLEAN_ANSWERS:
            return 0.0
    prediction_tokens, ground_truth_tokens = prediction.split(), ground_truth.split()
    common = sum((Counter(prediction_tokens) & Counter(ground_truth_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(answer, gold, data_source=None):
    """Best strict EM and token F1 over every provided alias; never substring EM."""
    _profile(data_source)
    if not answer.strip():
        return {"em": 0, "f1": 0.0}
    aliases = _gold_answers(gold)
    return {
        "em": max((int(exact_match_score(answer, alias, data_source)) for alias in aliases), default=0),
        "f1": max((token_f1_score(answer, alias, data_source) for alias in aliases), default=0.0),
    }


def metric_spec(data_source=None):
    """Return serializable metric provenance for reports and resume manifests."""
    profile = _profile(data_source)
    return {
        "profile": _PROFILE_NAMES[profile],
        "version": METRIC_VERSION,
        "data_source": profile,
        "source_urls": [_SOURCE_URLS[profile]],
        "normalization": ["NFD"] * (profile == "nq") + [
            "lowercase", "remove_ASCII_punctuation", "remove_English_articles", "collapse_whitespace"],
        "em": "maximum_normalized_exact_match_over_all_gold_aliases",
        "f1": ("official_HotpotQA_answer_token_F1" if profile == "hotpotqa"
               else "auxiliary_token_overlap_F1_not_an_official_NQ_primary_metric" if profile == "nq"
               else "auxiliary_token_overlap_F1"),
        "raw_empty_prediction": "invalid_output_scores_zero",
        "normalized_empty_F1": 0.0,
        "search_r1_sub_em": "diagnostic_character_containment_only_never_acceptance",
        "search_r1_sub_em_source_url": _SOURCE_URLS["search-r1"],
    }


def answer_diagnostics(answer, gold, data_source=None):
    """Explain strict mismatch; expose unsafe upstream sub-EM only as a diagnostic.

    Search-R1 sub-EM intentionally uses its own upstream normalization even when
    the strict score uses the NQ profile. A match is not evidence of correctness:
    e.g. ``not Paris`` and ``Paris or London`` contain the gold ``Paris``.
    """
    aliases = _gold_answers(gold)
    normalized = normalize_answer(answer, data_source)
    search_normalized = normalize_answer(answer)
    alias_scores = []
    for alias in aliases:
        alias_scores.append({
            "answer": alias,
            "normalized_answer": normalize_answer(alias, data_source),
            "em": int(exact_match_score(answer, alias, data_source)),
            "f1": token_f1_score(answer, alias, data_source),
            "search_r1_sub_em": int(normalize_answer(alias) in search_normalized),
        })
    strict = score_answer(answer, aliases, data_source)
    sub_em = max((row["search_r1_sub_em"] for row in alias_scores), default=0)
    warnings = ["search_r1_sub_em_is_character_containment_not_exact_match_or_semantic_correctness"]
    if not answer.strip():
        warnings.append("raw_empty_prediction_is_invalid_even_if_pure_upstream_EM_matches")
    if not normalized:
        warnings.append("prediction_normalizes_to_empty_string")
    if any(not row["normalized_answer"] for row in alias_scores):
        warnings.append("gold_alias_normalizes_to_empty_string_official_normalization_loses_information")
    if sub_em and not strict["em"]:
        warnings.append("substring_only_match_cannot_make_a_trajectory_pass")
    return {
        "metric_spec": metric_spec(data_source),
        "raw_prediction_valid": bool(answer.strip()),
        "normalized_prediction": normalized,
        "aliases": alias_scores,
        "em": strict["em"],
        "f1": strict["f1"],
        "search_r1_sub_em": sub_em,
        "search_r1_sub_em_is_acceptance_criterion": False,
        "warnings": warnings,
    }
