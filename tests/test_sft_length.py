"""Length gates use the same real strings and masking boundaries as training."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_sft_dataset import SearchSFTDataset
from sft_length import check_episode_length, episode_steps, tokenizer_provenance


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 100000
    name_or_path = "local-test-only"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class LengthTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"question": "Capital of France"}
        self.events = [
            {"role": "assistant", "content": "<think>Find evidence.</think><search>France capital</search>"},
            {"role": "environment", "content": "<information>Doc 1: Paris is the capital.</information>"},
            {"role": "assistant", "content": "<think>The passage supports Paris.</think><answer>Paris</answer>"},
        ]
        self.tokenizer = CharacterTokenizer()

    def test_gate_lengths_and_supervision_match_training_dataset(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        result = check_episode_length(self.sample, self.events, self.tokenizer, 10000)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["step_lengths"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.parquet"
            rows = [{"sample_id": "fixture", **step} for step in episode_steps(self.sample, self.events)]
            pq.write_table(pa.Table.from_pylist(rows), path)
            dataset = SearchSFTDataset(path, self.tokenizer, max_length=10000, prompt_format="base")
        self.assertEqual(result["step_lengths"], [len(item["input_ids"]) for item in dataset])
        self.assertEqual(result["supervised_tokens"],
                         [sum(label != -100 for label in item["labels"]) for item in dataset])
        self.assertEqual(dataset[0]["input_ids"][-1], ord("\n"))
        self.assertEqual(dataset[1]["input_ids"][-1], self.tokenizer.eos_token_id)

    def test_exact_boundary_passes_and_one_less_rejects_entire_question(self):
        result = check_episode_length(self.sample, self.events, self.tokenizer, 10000)
        maximum = result["max_step_tokens"]
        self.assertTrue(check_episode_length(self.sample, self.events, self.tokenizer, maximum)["passed"])
        failed = check_episode_length(self.sample, self.events, self.tokenizer, maximum - 1)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["reason"], "over_max_length")

    def test_boundary_straddling_tokens_do_not_create_supervision(self):
        class WholeTextTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [1], "offset_mapping": [(0, len(text))]}
        result = check_episode_length(self.sample, self.events, WholeTextTokenizer(), 10000)
        self.assertEqual(result["reason"], "no_response_tokens")
        self.assertEqual(result["supervised_tokens"], [0, 1])

    def test_configuration_errors_and_missing_actions(self):
        self.assertEqual(check_episode_length(self.sample, [], self.tokenizer, 100)["reason"],
                         "no_assistant_steps")
        with self.assertRaisesRegex(ValueError, "positive"):
            check_episode_length(self.sample, self.events, self.tokenizer, 0)
        self.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "requires EOS"):
            check_episode_length(self.sample, self.events, self.tokenizer, 100)
        self.tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            check_episode_length(self.sample, self.events, self.tokenizer, 100)

    def test_fingerprint_records_loaded_tokenizer_not_only_alias(self):
        class Backend:
            def __init__(self, value):
                self.value = value
            def to_str(self):
                return self.value
        self.assertIsNone(tokenizer_provenance(None))
        self.tokenizer.backend_tokenizer = Backend("first")
        before = tokenizer_provenance(self.tokenizer)
        self.tokenizer.backend_tokenizer = Backend("changed")
        after = tokenizer_provenance(self.tokenizer)
        self.assertEqual(before["name_or_path"], after["name_or_path"])
        self.assertNotEqual(before["backend_sha256"], after["backend_sha256"])


if __name__ == "__main__":
    unittest.main()
