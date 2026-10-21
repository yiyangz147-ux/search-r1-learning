"""Behavioral integrity tests for the final full-embedding RAG evaluation."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_full_embedding_eval as preparation
import report_full_embedding_eval as reporting
from search_sft_protocol import build_prompt
from sft_reward_monitor import rollout


EXAMPLES = [
    {"sample_id": "question-a", "question": "What is the capital of France?",
     "golden_answers": ["Paris"], "data_source": "nq"},
    {"sample_id": "question-b", "question": "Is Rome in Italy?",
     "golden_answers": ["yes"], "data_source": "hotpotqa"},
]


class FullEmbeddingEvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "questions.jsonl"
        self.source.write_text("".join(json.dumps(row) + "\n" for row in EXAMPLES))
        self.output = self.root / "evaluation"
        self.output.mkdir()
        self.config_path = self.root / "config.json"
        raw = {"model_path": str(self.root / "base"), "adapter_path": str(self.root / "adapter"),
               "training_prepared_dir": str(self.root / "prepared"), "output_dir": str(self.output),
               "data_file": str(self.source), "expected_data_sha256": preparation.sha256(self.source),
               "expected_count": 2, "expected_source_counts": {"nq": 1, "hotpotqa": 1},
               "num_workers": 1, "selected_gpu_uuids": ["GPU-test"]}
        self.config_path.write_text(json.dumps(raw))
        self.config = preparation.load_config(self.config_path)
        self.examples = preparation.load_examples(self.config)
        examples_text = "".join(preparation.canonical(row) + "\n" for row in self.examples)
        (self.output / "examples.jsonl").write_text(examples_text)
        assets = {}
        preparation.record_asset(self.source, assets)
        self.manifest = {"format_version": 1, "complete": True, "subset": preparation.SUBSET,
            "configuration": self.config, "config_sha256": preparation.digest_text(preparation.canonical(self.config)),
            "selected": 2, "sources": {"nq": 1, "hotpotqa": 1}, "num_workers": 1,
            "data_sha256": preparation.sha256(self.source), "selected_sha256": preparation.selected_hash(self.examples),
            "examples_sha256": preparation.digest_text(examples_text), "asset_sha256": assets,
            "code_sha256": {name: preparation.sha256(Path(preparation.__file__).with_name(name))
                            for name in preparation.CODE_FILES},
            "training_monitor_subset": {"evaluation_sample_ids": ["question-a"]}}
        preparation.atomic_json(self.output / "run_manifest.json", self.manifest)
        (self.output / "workers").mkdir()
        self.journal = self.output / "workers" / "worker-00.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, index, outputs):
        texts = iter(outputs)
        row = rollout(self.examples[index], lambda _context, _budget: {
            "text": next(texts), "generated_tokens": 12, "input_tokens": 30},
            lambda query: {"query": query, "content": "<information>Evidence.</information>", "documents": []},
            self.config)
        row.update(id=row["sample_id"], sample_index=index, worker_index=0,
                   run_binding_sha256=preparation.digest_text(preparation.canonical(self.manifest)))
        return row

    def write_records(self, rows):
        self.journal.write_text("".join(preparation.canonical(row) + "\n" for row in rows))

    def test_correct_answer_with_bad_format_retains_answer_reward(self):
        row = self.record(0, ["Translation: <answer>The Paris!</answer>"])
        actual = reporting.rescore_record(row, self.examples[0], self.config)
        self.assertEqual(actual["total_reward"], 1)
        self.assertEqual(actual["format_reward"], 0)

    def test_format_success_and_final_means_include_failed_answers(self):
        correct = self.record(0, ["<think>I know this.</think><answer>Paris</answer>"])
        wrong = self.record(1, ["<think>I know this.</think><answer>no</answer>"])
        self.write_records([correct, wrong])
        result = reporting.report(self.config_path)
        self.assertTrue(result["complete"])
        self.assertEqual(result["accuracy_final"], 50)
        self.assertEqual(result["overall"]["format_reward"], 1)
        self.assertEqual(result["excluding_training_monitor"]["overall"]["selected"], 1)
        self.assertEqual(result["excluding_training_monitor"]["overall"]["total_reward"], 0)

    def test_missing_outputs_have_no_final_accuracy(self):
        self.write_records([self.record(0, ["<answer>Paris</answer>"])])
        result = reporting.report(self.config_path)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["accuracy_final"])
        self.assertIsNone(result["overall"]["total_reward"])
        self.assertEqual(result["overall"]["attempted_mean_total_reward"], 1)
        self.assertEqual(result["overall"]["pending"], 1)

    def test_unfinished_generation_is_a_terminal_zero(self):
        row = self.record(0, ["I am thinking", "Still thinking"])
        self.assertEqual(row["status"], "invalid_action_limit")
        rescored = reporting.rescore_record(row, self.examples[0], self.config)
        self.assertEqual(rescored["total_reward"], 0)
        self.assertEqual(rescored["format_reward"], 0)

    def test_report_rejects_forged_answer_even_with_forged_reward(self):
        row = self.record(0, ["<answer>London</answer>"])
        row.update(answer="Paris", total_reward=1, answer_reward=1, qa_em=1, qa_f1=1)
        with self.assertRaisesRegex(ValueError, "answer/status"):
            reporting.rescore_record(row, self.examples[0], self.config)

    def test_report_rejects_forged_format_flags(self):
        row = self.record(0, ["Translation: <answer>Paris</answer>"])
        row["events"][0]["format_valid"] = True
        row["format_reward"] = 1
        with self.assertRaisesRegex(ValueError, "format_reward"):
            reporting.rescore_record(row, self.examples[0], self.config)

    def test_duplicate_committed_predictions_are_rejected(self):
        row = self.record(0, ["<answer>Paris</answer>"])
        self.write_records([row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate committed"):
            reporting.load_predictions(self.config, self.manifest, self.examples)

    def test_infrastructure_error_remains_pending(self):
        row = self.record(0, ["<answer>Paris</answer>"])
        row["status"] = "retrieval_error"
        self.write_records([row])
        records, _, errors, _ = reporting.load_predictions(self.config, self.manifest, self.examples)
        self.assertEqual(records, {})
        self.assertEqual(len(errors), 1)
        self.assertEqual(reporting.group_report(self.examples, records)["pending"], 2)

    def test_partial_utf8_tail_is_ignored(self):
        row = self.record(0, ["<answer>Paris</answer>"])
        self.write_records([row])
        with self.journal.open("ab") as stream:
            stream.write(b'{"unfinished":"\xe4\xb8')
        records, _, _, _ = reporting.load_predictions(self.config, self.manifest, self.examples)
        self.assertEqual(set(records), {"question-a"})

    def test_verification_rejects_changed_source_and_code_binding(self):
        preparation.verify_run(self.config_path)
        changed = copy.deepcopy(self.manifest)
        changed["code_sha256"][preparation.CODE_FILES[0]] = "0" * 64
        preparation.atomic_json(self.output / "run_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "code changed"):
            preparation.verify_run(self.config_path)
        preparation.atomic_json(self.output / "run_manifest.json", self.manifest)
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "byte count changed"):
            preparation.verify_run(self.config_path)

    def test_monitor_matching_uses_question_and_gold_not_historical_id(self):
        monitor = self.root / "monitor.jsonl"
        changed = {**EXAMPLES[0], "sample_id": "new-monitor-id", "question": "What is the capital of France"}
        monitor.write_text(json.dumps(changed) + "\n")
        config = {**self.config, "reward_monitor_file": str(monitor), "expected_reward_monitor_count": 1}
        subset = preparation.bind_monitor_subset(config, self.examples, {})
        self.assertEqual(subset["evaluation_sample_ids"], ["question-a"])
        changed["golden_answers"] = ["London"]
        monitor.write_text(json.dumps(changed) + "\n")
        with self.assertRaisesRegex(ValueError, "gold aliases disagree"):
            preparation.bind_monitor_subset(config, self.examples, {})


if __name__ == "__main__":
    unittest.main()
