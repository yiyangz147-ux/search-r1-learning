"""Evidence-reviewed answer aliases for SFT training candidates only.

This optional answer gate supplements strict official EM with exact matches to
explicitly reviewed aliases. It never rewrites the dataset's gold answers or
changes the reported EM/F1. A passing answer still needs the caller's length and
retrieval-evidence checks. Manifest evidence is an auditable review record; this
module does not fetch sources or establish that a reviewer told the truth.

Manifest version 1 has exactly these fields (hashes below are placeholders)::

    {
      "version": 1,
      "scope": "sft_train_only",
      "entries": [{
        "sample_id": "training-example-1",
        "data_source": "hotpotqa",
        "source_split": "train",
        "question_sha256": "<sha256 of the original question UTF-8 bytes>",
        "golden_answers_sha256": "<sha256 of canonical original gold list>",
        "aliases": ["four acts"],
        "review": {
          "method": "manual_evidence_review",
          "reason": "The reviewed source explicitly gives the play's act count.",
          "evidence": [{
            "source": "https://example.org/source-for-this-training-question",
            "quote": "The play consists of four acts."
          }]
        }
      }]
    }

Use ``question_sha256(question)`` and ``golden_answers_sha256(gold_list)``
to create the two bindings. Load once, validate the complete index against the
actual training candidate list before generation, then pass it to each check::

    index = load_reviewed_aliases(path)
    validate_reviewed_aliases(index, candidates)
    gate = check_sft_answer(candidate, answer, index)

No substring, token-F1 threshold, model judgement, or external API can admit an
answer through this module. An alias is valid for one immutable train sample.
"""

import hashlib
import json
from pathlib import Path
import re

from qa_metrics import normalize_answer, score_answer


_ENTRY_FIELDS = frozenset(("sample_id", "data_source", "source_split",
                           "question_sha256", "golden_answers_sha256", "aliases", "review"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROTECTED_PARTITION_WORDS = frozenset(("validation", "valid", "val", "dev", "test",
                                        "eval", "evaluation", "heldout", "holdout"))


class ReviewedAliasesError(ValueError):
    """An alias manifest or its binding to a train candidate is invalid."""


def _fail(message):
    raise ReviewedAliasesError(message)


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(f"{label} must contain exactly: {', '.join(sorted(expected))}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a nonempty string")
    return value


def question_sha256(question):
    """SHA256 of the original question, without trimming or normalization."""
    _text(question, "question")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def golden_answers_sha256(golden_answers):
    """SHA256 of the ordered, original gold list's canonical UTF-8 JSON."""
    if not isinstance(golden_answers, list) or not golden_answers:
        _fail("golden_answers must be a nonempty list of strings")
    for answer in golden_answers:
        _text(answer, "golden_answers item")
    canonical = json.dumps(golden_answers, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_entry(entry):
    _fields(entry, _ENTRY_FIELDS, "alias entry")
    for field in ("sample_id", "data_source"):
        _text(entry[field], field)
    if entry["source_split"] != "train":
        _fail("reviewed aliases are restricted to source_split='train'")
    for field in ("question_sha256", "golden_answers_sha256"):
        if not isinstance(entry[field], str) or not _SHA256.fullmatch(entry[field]):
            _fail(f"{field} must be a lowercase hexadecimal SHA256")
    aliases = entry["aliases"]
    if not isinstance(aliases, list) or not aliases:
        _fail("aliases must be a nonempty list of strings")
    seen = set()
    for alias in aliases:
        _text(alias, "alias")
        try:
            normalized = normalize_answer(alias, entry["data_source"])
        except (TypeError, ValueError) as exc:
            raise ReviewedAliasesError("unsupported alias data_source") from exc
        if not normalized:
            _fail("an alias must not normalize to an empty answer")
        if normalized in seen:
            _fail("duplicate normalized alias in one entry")
        seen.add(normalized)
    review = entry["review"]
    _fields(review, ("method", "reason", "evidence"), "alias review")
    _text(review["method"], "review method")
    _text(review["reason"], "review reason")
    if not isinstance(review["evidence"], list) or not review["evidence"]:
        _fail("review evidence must be a nonempty list")
    for evidence in review["evidence"]:
        _fields(evidence, ("source", "quote"), "review evidence item")
        _text(evidence["source"], "evidence source")
        _text(evidence["quote"], "evidence quote")


def _validate_index(index):
    if not isinstance(index, dict):
        _fail("reviewed alias index must be a sample_id-to-entry dictionary")
    for sample_id, entry in index.items():
        _validate_entry(entry)
        if sample_id != entry["sample_id"]:
            _fail("alias index key does not match entry sample_id")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    _fail(f"invalid JSON constant: {value}")


def load_reviewed_aliases(path):
    """Load a strict version-1 JSON manifest; return ``{sample_id: entry}``.

    Review evidence is preserved verbatim in each returned entry. Duplicate JSON
    fields, sample IDs, and normalized aliases are rejected. Empty entry lists
    are allowed and never add an answer to the strict-EM acceptance set.
    """
    try:
        with Path(path).open(encoding="utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook=_unique_object,
                                 parse_constant=_invalid_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReviewedAliasesError("invalid reviewed-alias JSON") from exc
    _fields(manifest, ("version", "scope", "entries"), "alias manifest")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        _fail("unsupported reviewed-alias manifest version")
    if manifest["scope"] != "sft_train_only":
        _fail("reviewed-alias manifest scope must be 'sft_train_only'")
    if not isinstance(manifest["entries"], list):
        _fail("manifest entries must be a list")
    index = {}
    for entry in manifest["entries"]:
        _validate_entry(entry)
        sample_id = entry["sample_id"]
        if sample_id in index:
            _fail("duplicate alias sample_id")
        index[sample_id] = entry
    return index


def _require_training_sample(sample):
    """Reject held-out copies even when their upstream source split is train."""
    if not isinstance(sample, dict):
        _fail("candidate must be a dictionary")
    if sample.get("source_split") != "train":
        _fail("reviewed aliases cannot be used with a non-train sample")
    metadata = [("candidate", sample)]
    extra_info = sample.get("extra_info")
    if extra_info is not None:
        if not isinstance(extra_info, dict):
            _fail("candidate extra_info must be a dictionary when supplied")
        metadata.append(("candidate extra_info", extra_info))
    for label, info in metadata:
        for field in ("partition", "split"):
            value = info.get(field)
            if value is None:
                continue
            _text(value, f"{label} {field}")
            words = set(re.findall(r"[a-z0-9]+", value.lower()))
            if (words & _PROTECTED_PARTITION_WORDS or
                    {"held", "out"}.issubset(words) or {"hold", "out"}.issubset(words)):
                _fail("reviewed aliases cannot be used with a protected evaluation partition")


def _bind_entry(entry, sample):
    _require_training_sample(sample)
    if sample.get("sample_id") != entry["sample_id"]:
        _fail("alias sample_id does not match candidate")
    if sample.get("data_source") != entry["data_source"]:
        _fail("alias data_source does not match candidate")
    if question_sha256(sample.get("question")) != entry["question_sha256"]:
        _fail("alias question_sha256 does not match candidate")
    if golden_answers_sha256(sample.get("golden_answers")) != entry["golden_answers_sha256"]:
        _fail("alias golden_answers_sha256 does not match candidate")


def validate_reviewed_aliases(index, candidates):
    """Validate every entry against the complete generation input; return None.

    The input list may contain candidates without patches, but all candidates
    must be train examples outside protected evaluation partitions, and sample
    IDs must be unique. Unknown patch IDs and any changed question, ordered gold
    list, source, or split fail closed.
    """
    _validate_index(index)
    candidate_index = {}
    for sample in candidates:
        _require_training_sample(sample)
        sample_id = _text(sample.get("sample_id"), "candidate sample_id")
        if sample_id in candidate_index:
            _fail("duplicate candidate sample_id")
        candidate_index[sample_id] = sample
    for sample_id, entry in index.items():
        if sample_id not in candidate_index:
            _fail("reviewed alias references an unknown candidate sample_id")
        _bind_entry(entry, candidate_index[sample_id])


def check_sft_answer(sample, answer, reviewed_aliases=None):
    """Return original-gold EM/F1 and the separate SFT answer-gate decision.

    With an alias index, non-train samples and protected evaluation partitions
    are rejected even if no patch exists for their ID. A matching alias is
    applied only after rechecking the sample binding, so changed questions or
    gold labels cannot reuse a previous review.
    """
    if not isinstance(sample, dict):
        _fail("sample must be a dictionary")
    if not isinstance(answer, str):
        _fail("answer must be a string")
    entry = None
    if reviewed_aliases is not None:
        _validate_index(reviewed_aliases)
        _require_training_sample(sample)
        entry = reviewed_aliases.get(sample.get("sample_id"))
        if entry is not None:
            _bind_entry(entry, sample)
    score = score_answer(answer, sample["golden_answers"], sample.get("data_source"))
    matched_alias = None
    if entry is not None:
        matched_alias = next((alias for alias in entry["aliases"]
                              if score_answer(answer, [alias], entry["data_source"])["em"]), None)
    official_em = bool(score["em"])
    alias_em = matched_alias is not None
    return {
        "policy": "strict_em_or_reviewed_alias" if reviewed_aliases is not None else "strict_em",
        "passed": official_em or alias_em,
        "method": "official_em" if official_em else "reviewed_alias" if alias_em else "none",
        "em": official_em,
        "f1": float(score["f1"]),
        "reviewed_alias_em": alias_em,
        "matched_alias": matched_alias,
    }
