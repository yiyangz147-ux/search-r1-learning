"""Offline differential checks against pinned official pure evaluator functions."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import sys
import unicodedata
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qa_metrics as metrics


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "official_qa_metrics"


def official_functions(snapshot, function_names):
    """Execute selected pure AST functions only; ignore imports and program entry."""
    tree = ast.parse((FIXTURES / snapshot).read_text(encoding="utf-8"), filename=snapshot)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
    if {node.name for node in selected} != set(function_names):
        raise AssertionError("Pinned official snapshot is missing requested pure functions")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"re": re, "string": string, "unicodedata": unicodedata, "Counter": Counter}
    exec(compile(module, str(FIXTURES / snapshot), "exec"), namespace)
    return namespace


class OfficialMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orqa = official_functions("google_orqa_eval_utils.py", {
            "normalize_answer", "exact_match_score", "metric_max_over_ground_truths", "is_correct"})
        cls.hotpot = official_functions("hotpot_evaluate_v1.py", {
            "normalize_answer", "exact_match_score", "f1_score"})
        cls.search = official_functions("search_r1_qa_em.py", {
            "normalize_answer", "em_check", "subem_check"})

    def test_pinned_snapshots_and_licenses_match_manifest(self):
        sources = json.loads((FIXTURES / "sources.json").read_text())["sources"]
        self.assertEqual(set(sources), {"nq_open_orqa", "hotpotqa", "search_r1"})
        for source in sources.values():
            self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
            self.assertIn(source["commit"], source["source_url"])
            self.assertEqual(source["license"], "Apache-2.0")
            for file_key, hash_key in (("snapshot", "sha256"), ("license_snapshot", "license_sha256")):
                self.assertEqual(hashlib.sha256((FIXTURES / source[file_key]).read_bytes()).hexdigest(),
                                 source[hash_key])

    def test_normalization_matches_each_official_source(self):
        cases = ["The Paris!", "  A  name\tand AN article\n", "A+", "The The", "!!!", "",
                 "Café", "Cafe\u0301", "á", "Édith Piaf", "a—b", "don't", "a.b", "a\u00a0b"]
        for source, official in ((None, self.search), ("search-r1", self.search),
                                 ("nq", self.orqa), ("hotpotqa", self.hotpot)):
            for text in cases:
                with self.subTest(source=source, text=text):
                    self.assertEqual(metrics.normalize_answer(text, source), official["normalize_answer"](text))

    def test_exact_match_agrees_on_unicode_aliases_prose_and_empty_norm(self):
        pairs = [("The Paris!", "Paris"), ("Paris, France", "Paris"), ("Café", "Cafe\u0301"),
                 ("café", "cafe"), ("A+", "!!!"), ("", "the"), ("yes", "no"),
                 ("Paul Tibbets", "Paul Warfield Tibbets Jr."), ("stock car", "an anthropomorphic stock car")]
        for source, official in (("nq", self.orqa), ("hotpotqa", self.hotpot)):
            for prediction, gold in pairs:
                with self.subTest(source=source, prediction=prediction, gold=gold):
                    self.assertEqual(metrics.exact_match_score(prediction, gold, source),
                                     official["exact_match_score"](prediction, gold))
        for prediction, gold in pairs:
            self.assertEqual(metrics.exact_match_score(prediction, gold), bool(self.search["em_check"](prediction, [gold])))

    def test_nq_nfd_equivalence_does_not_remove_diacritics_or_change_hotpot(self):
        self.assertEqual(metrics.score_answer("Café", ["Cafe\u0301"], "nq")["em"], 1)
        self.assertEqual(metrics.score_answer("Café", ["Cafe\u0301"], "hotpotqa")["em"], 0)
        self.assertEqual(metrics.score_answer("Café", ["Cafe\u0301"])["em"], 0)
        self.assertEqual(metrics.score_answer("Café", ["cafe"], "nq")["em"], 0)

    def test_hotpot_f1_matches_official_boolean_and_empty_normalization_cases(self):
        pairs = [("yes", "yes"), ("yes indeed", "yes"), ("yes", "yes indeed"),
                 ("noanswer", "no answer"), ("no", "no thank you"), ("yes", "no"),
                 ("yes Paris", "Paris"), ("Paris", "yes Paris"), ("yes Paris", "no Paris"),
                 ("", ""), ("The The", "!!!"), ("A+", "A+"),
                 ("red red blue", "red blue blue blue"), ("Paris, France", "Paris")]
        for prediction, gold in pairs:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertAlmostEqual(metrics.token_f1_score(prediction, gold, "hotpotqa"),
                                       self.hotpot["f1_score"](prediction, gold)[0])
        self.assertAlmostEqual(metrics.token_f1_score("yes Paris", "Paris", "hotpotqa"), 2 / 3)
        self.assertAlmostEqual(metrics.token_f1_score("yes indeed", "yes", "nq"), 2 / 3)

    def test_all_aliases_are_compared_and_best_f1_is_retained(self):
        aliases = ["Lyon", "New York", "NYC"]
        self.assertEqual(metrics.score_answer("NYC", aliases, "nq"), {"em": 1, "f1": 1.0})
        self.assertEqual(self.orqa["is_correct"](aliases, "NYC", False), True)
        aliases = ["Lyon", "the Paris"]
        actual = metrics.score_answer("Paris France", aliases, "hotpotqa")
        self.assertEqual(actual["em"], 0)
        self.assertAlmostEqual(actual["f1"], max(self.hotpot["f1_score"]("Paris France", g)[0] for g in aliases))
        self.assertEqual(metrics.score_answer("NYC", "NYC"), {"em": 1, "f1": 1.0})
        self.assertEqual(metrics.score_answer("NYC", []), {"em": 0, "f1": 0.0})

    def test_raw_empty_output_is_invalid_without_rewriting_pure_official_em(self):
        self.assertTrue(metrics.exact_match_score("", "the", "nq"))
        self.assertTrue(self.orqa["exact_match_score"]("", "the"))
        for source in (None, "nq", "hotpotqa"):
            self.assertEqual(metrics.score_answer("  \t", ["the"], source), {"em": 0, "f1": 0.0})
            self.assertEqual(metrics.score_answer("A+", ["A+"], source), {"em": 1, "f1": 0.0})

    def test_character_subem_diagnostic_matches_upstream_but_cannot_pass(self):
        pairs = [("not Paris", ["Paris"]), ("Paris or London", ["Paris"]),
                 ("Parisian", ["Paris"]), ("The Exorcist (1973)", ["The Exorcist"]),
                 ("stock car", ["an anthropomorphic stock car"]), ("anything", ["!!!"])]
        for prediction, gold in pairs:
            with self.subTest(prediction=prediction):
                diagnostic = metrics.answer_diagnostics(prediction, gold, "nq")
                self.assertEqual(diagnostic["search_r1_sub_em"], self.search["subem_check"](prediction, gold))
                self.assertFalse(diagnostic["search_r1_sub_em_is_acceptance_criterion"])
                self.assertEqual(diagnostic["em"], metrics.score_answer(prediction, gold, "nq")["em"])
                self.assertNotIn("passed", diagnostic)
        for prediction in ("not Paris", "Paris or London", "Parisian"):
            diagnostic = metrics.answer_diagnostics(prediction, ["Paris"])
            self.assertEqual(diagnostic["search_r1_sub_em"], 1)
            self.assertEqual(diagnostic["em"], 0)
            self.assertIn("substring_only_match_cannot_make_a_trajectory_pass", diagnostic["warnings"])

    def test_diagnostics_preserve_each_alias_and_warn_about_information_loss(self):
        diagnostic = metrics.answer_diagnostics("Paul Tibbets", ["Paul Warfield Tibbets Jr.", "Paul Tibbets"], "nq")
        self.assertEqual(diagnostic["normalized_prediction"], "paul tibbets")
        self.assertEqual([row["em"] for row in diagnostic["aliases"]], [0, 1])
        self.assertEqual(diagnostic["aliases"][1]["normalized_answer"], "paul tibbets")
        diagnostic = metrics.answer_diagnostics("!!!", ["The The"], "hotpotqa")
        self.assertIn("prediction_normalizes_to_empty_string", diagnostic["warnings"])
        self.assertTrue(any("gold_alias_normalizes_to_empty_string" in x for x in diagnostic["warnings"]))
        self.assertEqual(diagnostic["em"], 1)
        self.assertEqual(diagnostic["f1"], 0.0)

    def test_provenance_spec_is_json_serializable_and_marks_nq_f1_auxiliary(self):
        for source in (None, "nq", "hotpotqa", "search-r1"):
            specification = metrics.metric_spec(source)
            self.assertEqual(specification["version"], metrics.METRIC_VERSION)
            self.assertTrue(specification["source_urls"])
            json.dumps(specification)
        self.assertIn("not_an_official_NQ_primary_metric", metrics.metric_spec("nq")["f1"])
        self.assertEqual(metrics.metric_spec("nq")["normalization"][0], "NFD")
        self.assertNotIn("NFD", metrics.metric_spec("hotpotqa")["normalization"])
        self.assertEqual(metrics.metric_spec("nq_search"), metrics.metric_spec("nq"))
        self.assertEqual(metrics.metric_spec("hotpotqa_search"), metrics.metric_spec("hotpotqa"))
        with self.assertRaisesRegex(ValueError, "Unsupported QA metric"):
            metrics.metric_spec("unverified-dataset")


if __name__ == "__main__":
    unittest.main()
