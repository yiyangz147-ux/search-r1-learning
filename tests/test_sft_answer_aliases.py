"""Local fixtures for reviewed SFT aliases; no production data or network."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sft_answer_aliases as aliases


class ReviewedAliasTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"sample_id": "play-count-1", "data_source": "hotpotqa",
                       "source_split": "train", "question": "How many acts does Play X have?",
                       "golden_answers": ["four"]}
        self.entry = {
            "sample_id": self.sample["sample_id"], "data_source": self.sample["data_source"],
            "source_split": "train", "question_sha256": aliases.question_sha256(self.sample["question"]),
            "golden_answers_sha256": aliases.golden_answers_sha256(self.sample["golden_answers"]),
            "aliases": ["four acts"], "review": {"method": "manual_evidence_review",
                "reason": "The cited source gives the play's act count.",
                "evidence": [{"source": "https://fixture.invalid/play-x",
                              "quote": "Play X consists of four acts."}]}}
        self.manifest = {"version": 1, "scope": "sft_train_only", "entries": [self.entry]}

    def load(self, manifest=None, *, raw=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.json"
            path.write_text(raw if raw is not None else json.dumps(
                self.manifest if manifest is None else manifest, ensure_ascii=False), encoding="utf-8")
            return aliases.load_reviewed_aliases(path)

    def test_hashes_preserve_original_question_order_and_unicode(self):
        question = " Café? \n"
        self.assertEqual(aliases.question_sha256(question), hashlib.sha256(question.encode("utf-8")).hexdigest())
        gold = ["Café", "Cafe\u0301"]
        canonical = json.dumps(gold, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(aliases.golden_answers_sha256(gold), hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        self.assertNotEqual(aliases.golden_answers_sha256(gold), aliases.golden_answers_sha256(gold[::-1]))
        self.assertNotEqual(aliases.question_sha256(question), aliases.question_sha256(question.strip()))

    def test_explicit_reviewed_alias_pass_does_not_change_official_em_or_gold(self):
        index = self.load()
        self.assertIsNone(aliases.validate_reviewed_aliases(index, [self.sample]))
        original = deepcopy(self.sample)
        strict = aliases.check_sft_answer(self.sample, "four acts")
        reviewed = aliases.check_sft_answer(self.sample, "Four acts!", index)
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["policy"], "strict_em")
        self.assertTrue(reviewed["passed"])
        self.assertFalse(reviewed["em"])
        self.assertAlmostEqual(reviewed["f1"], 2 / 3)
        self.assertEqual(reviewed["method"], "reviewed_alias")
        self.assertEqual(reviewed["matched_alias"], "four acts")
        self.assertTrue(reviewed["reviewed_alias_em"])
        self.assertEqual(reviewed["policy"], "strict_em_or_reviewed_alias")
        self.assertEqual(self.sample, original)
        self.assertEqual(index[self.sample["sample_id"]]["review"], self.entry["review"])

    def test_official_match_and_unpatched_train_sample_remain_strict(self):
        index = self.load()
        matched = aliases.check_sft_answer(self.sample, "Four", index)
        self.assertTrue(matched["em"])
        self.assertEqual(matched["method"], "official_em")
        self.assertFalse(matched["reviewed_alias_em"])
        untouched = dict(self.sample, sample_id="other-training-case")
        self.assertTrue(aliases.check_sft_answer(untouched, "four", index)["passed"])
        self.assertFalse(aliases.check_sft_answer(untouched, "four acts", index)["passed"])

    def test_wrong_number_negation_and_partial_strings_never_pass(self):
        index = self.load()
        for answer in ("forty acts", "fourteen acts", "not four acts", "four acts or five acts",
                       "The answer might be four acts", "", "  "):
            with self.subTest(answer=answer):
                result = aliases.check_sft_answer(self.sample, answer, index)
                self.assertFalse(result["passed"])
                self.assertFalse(result["reviewed_alias_em"])
                self.assertEqual(result["method"], "none")

    def test_nq_normalization_uses_official_nfd_for_aliases(self):
        self.sample.update(data_source="nq", golden_answers=["elsewhere"])
        self.entry.update(data_source="nq", aliases=["Café"],
                          golden_answers_sha256=aliases.golden_answers_sha256(self.sample["golden_answers"]))
        result = aliases.check_sft_answer(self.sample, "Cafe\u0301", self.load())
        self.assertTrue(result["passed"])
        self.assertFalse(result["em"])
        self.assertEqual(result["f1"], 0.0)

    def test_no_patch_can_be_applied_to_any_validation_or_test_split(self):
        index = self.load()
        for split in ("validation", "val", "dev", "test", None, "TRAIN"):
            for sample_id in (self.sample["sample_id"], "unpatched-case"):
                with self.subTest(split=split, sample_id=sample_id):
                    candidate = dict(self.sample, source_split=split, sample_id=sample_id)
                    with self.assertRaises(aliases.ReviewedAliasesError):
                        aliases.check_sft_answer(candidate, "four", index)
                    with self.assertRaises(aliases.ReviewedAliasesError):
                        aliases.validate_reviewed_aliases(index, [candidate])
        self.assertTrue(aliases.check_sft_answer(dict(self.sample, source_split="val"), "four")["em"])

    def test_changed_question_gold_or_source_invalidates_binding_even_for_official_match(self):
        index = self.load()
        for update in ({"question": self.sample["question"] + " "},
                       {"golden_answers": ["four", "4"]},
                       {"golden_answers": ["four acts"]}, {"data_source": "nq"}):
            with self.subTest(update=update):
                candidate = dict(self.sample, **update)
                with self.assertRaises(aliases.ReviewedAliasesError):
                    aliases.validate_reviewed_aliases(index, [candidate])
                with self.assertRaises(aliases.ReviewedAliasesError):
                    aliases.check_sft_answer(candidate, "four", index)

    def test_protected_partitions_cannot_reuse_a_train_alias_binding(self):
        index = self.load()
        for partition in ("validation", "val", "dev", "test", "official_evaluation",
                          "independent_validation", "held-out", "hold_out", "TEST", " validation "):
            for metadata in ({"partition": partition}, {"split": partition},
                             {"extra_info": {"split": partition}},
                             {"extra_info": {"partition": partition}}):
                for sample_id in (self.sample["sample_id"], "unpatched-case"):
                    with self.subTest(metadata=metadata, sample_id=sample_id):
                        candidate = dict(self.sample, sample_id=sample_id, **metadata)
                        with self.assertRaises(aliases.ReviewedAliasesError):
                            aliases.check_sft_answer(candidate, "four", index)
                        with self.assertRaises(aliases.ReviewedAliasesError):
                            aliases.validate_reviewed_aliases(index, [candidate])
        # Strict reporting without a reviewed-alias policy remains available.
        held_out = dict(self.sample, partition="validation")
        self.assertFalse(aliases.check_sft_answer(held_out, "four acts")["passed"])

    def test_training_partitions_and_absent_partition_remain_supported(self):
        index = self.load()
        for metadata in ({}, {"partition": "train"}, {"partition": "sft_initial"},
                         {"partition": "sft_reserve"}, {"extra_info": {"split": "train"}},
                         {"partition": "sft_initial", "extra_info": {"split": "train"}}):
            with self.subTest(metadata=metadata):
                candidate = dict(self.sample, **metadata)
                aliases.validate_reviewed_aliases(index, [candidate])
                self.assertTrue(aliases.check_sft_answer(candidate, "four acts", index)["passed"])

    def test_unknown_ids_duplicate_candidates_and_mutated_index_fail(self):
        index = self.load()
        with self.assertRaises(aliases.ReviewedAliasesError):
            aliases.validate_reviewed_aliases(index, [dict(self.sample, sample_id="unknown")])
        with self.assertRaises(aliases.ReviewedAliasesError):
            aliases.validate_reviewed_aliases(index, [self.sample, deepcopy(self.sample)])
        with self.assertRaises(aliases.ReviewedAliasesError):
            aliases.check_sft_answer(self.sample, "four acts", {"wrong-index-key": self.entry})
        index[self.sample["sample_id"]]["source_split"] = "test"
        with self.assertRaises(aliases.ReviewedAliasesError):
            aliases.check_sft_answer(self.sample, "four", index)

    def test_duplicate_ids_aliases_empty_normalization_and_unsupported_source_rejected(self):
        bad_manifests = []
        duplicate = deepcopy(self.manifest)
        duplicate["entries"].append(deepcopy(self.entry))
        bad_manifests.append(duplicate)
        for changes in ({"aliases": ["four acts", "The FOUR acts!"]},
                        {"aliases": ["four acts", "four acts"]},
                        {"aliases": ["The"]}, {"aliases": ["!!!"]},
                        {"aliases": []}, {"aliases": [4]}, {"aliases": "four acts"},
                        {"data_source": "unknown-data"}, {"source_split": "val"}):
            manifest = deepcopy(self.manifest)
            manifest["entries"][0].update(changes)
            bad_manifests.append(manifest)
        for manifest in bad_manifests:
            with self.subTest(manifest=manifest):
                with self.assertRaises(aliases.ReviewedAliasesError):
                    self.load(manifest)

    def test_strict_manifest_and_evidence_schema_reject_invalid_records(self):
        bad_manifests = []
        for changes in ({"version": True}, {"version": 2}, {"scope": "all_splits"},
                        {"entries": {}}, {"extra": "not allowed"}):
            manifest = deepcopy(self.manifest)
            manifest.update(changes)
            bad_manifests.append(manifest)
        for changes in ({"question_sha256": "bad-hash"},
                        {"golden_answers_sha256": "A" * 64}, {"sample_id": " "},
                        {"extra": "not allowed"}, {"review": {"method": "human", "reason": "ok", "evidence": []}},
                        {"review": {"method": "", "reason": "ok", "evidence": [{"source": "x", "quote": "q"}]}},
                        {"review": {"method": "human", "reason": "ok", "evidence": [{"source": "x", "quote": " "}]}},
                        {"review": {"method": "human", "reason": "ok", "evidence": [{"source": "x", "quote": "q", "extra": 1}]}}):
            manifest = deepcopy(self.manifest)
            manifest["entries"][0].update(changes)
            bad_manifests.append(manifest)
        for manifest in bad_manifests:
            with self.subTest(manifest=manifest):
                with self.assertRaises(aliases.ReviewedAliasesError):
                    self.load(manifest)
        for raw in ('{"version":1,"version":1,"scope":"sft_train_only","entries":[]}',
                    '{"version":NaN,"scope":"sft_train_only","entries":[]}',
                    '{"version":1,"scope":"sft_train_only","entries":['):
            with self.subTest(raw=raw):
                with self.assertRaises(aliases.ReviewedAliasesError):
                    self.load(raw=raw)

    def test_empty_index_cannot_relax_em(self):
        index = self.load({"version": 1, "scope": "sft_train_only", "entries": []})
        aliases.validate_reviewed_aliases(index, [self.sample])
        self.assertFalse(aliases.check_sft_answer(self.sample, "four acts", index)["passed"])
        self.assertTrue(aliases.check_sft_answer(self.sample, "four", index)["passed"])


if __name__ == "__main__":
    unittest.main()
