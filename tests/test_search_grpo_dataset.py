"""CPU tests for the raw Base dataset/agent boundary (requires torch+pyarrow)."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_sft_protocol import build_prompt


class BaseTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        if add_special_tokens:
            raise AssertionError("Base dataset must not add tokenizer special tokens")
        return [ord(char) + 1 for char in text]

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("Base dataset must never apply chat template")


def row(sample_id="nq-1", question="What is the capital of France?"):
    return {
        "prompt": build_prompt(question), "agent_name": "search_r1_base", "data_source": "nq",
        "reward_model": {"style": "rule", "ground_truth": {"target": ["Paris"]}},
        "extra_info": {"sample_id": sample_id, "question": question, "index": 7},
    }


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("pyarrow"),
                     "Requires CPU torch and pyarrow; run in server SFT .venv")
class RawBaseDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from search_grpo_dataset import SearchR1BaseDataset
        cls.pa, cls.pq, cls.dataset_class = pa, pq, SearchR1BaseDataset

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write(self, rows, filename="data.parquet"):
        path = self.directory / filename
        self.pq.write_table(self.pa.Table.from_pylist(rows), path)
        return path

    def make(self, rows, **config):
        tokenizer = BaseTokenizer()
        dataset = self.dataset_class(self.write(rows), tokenizer, {"max_prompt_length": 1024, **config})
        return dataset, tokenizer

    def test_raw_prompt_tokenization_left_padding_mask_and_positions(self):
        original = row()
        dataset, tokenizer = self.make([original])
        item = dataset[0]
        actual = [ord(char) + 1 for char in original["prompt"]]
        padding = 1024 - len(actual)
        self.assertGreater(padding, 0)
        self.assertEqual(tokenizer.calls, [(original["prompt"], False)])
        self.assertEqual(item["input_ids"].tolist(), [0] * padding + actual)
        self.assertEqual(item["attention_mask"].tolist(), [0] * padding + [1] * len(actual))
        self.assertEqual(item["position_ids"].tolist(), [0] * padding + list(range(len(actual))))
        self.assertEqual(item["raw_prompt_ids"], actual)
        self.assertEqual(item["raw_prompt"], [{"role": "user", "content": original["prompt"]}])
        self.assertEqual(item["index"], 7)
        self.assertEqual(item["tools_kwargs"], {})

    def test_agent_kwargs_keep_question_id_gold_and_item_mutation_isolated(self):
        from search_grpo_agent_loop import example_from_kwargs
        dataset, _ = self.make([row()])
        item = dataset[0]
        example = example_from_kwargs(item)
        self.assertEqual(example, {"sample_id": "nq-1", "question": row()["extra_info"]["question"],
                                   "data_source": "nq", "golden_answers": ["Paris"]})
        item["extra_info"]["question"] = "Changed?"
        item["reward_model"]["ground_truth"]["target"].append("wrong")
        self.assertEqual(dataset[0]["extra_info"]["question"], row()["extra_info"]["question"])
        self.assertEqual(dataset[0]["reward_model"]["ground_truth"]["target"], ["Paris"])

    def test_chat_wrapped_or_modified_prompt_is_rejected(self):
        for changed in ["<|im_start|>user\n" + row()["prompt"] + "<|im_end|>",
                        row()["prompt"] + " ", [{"role": "user", "content": row()["prompt"]}]]:
            original = row()
            original["prompt"] = changed
            with self.subTest(prompt_type=type(changed).__name__):
                with self.assertRaisesRegex(ValueError, "raw Search-R1 Base prompt"):
                    self.make([original])

    def test_duplicate_ids_across_files_and_overlong_prompt_are_rejected(self):
        first = self.write([row()], "first.parquet")
        second = self.write([row()], "second.parquet")
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            self.dataset_class([first, second], BaseTokenizer(), {"max_prompt_length": 1024})
        with self.assertRaisesRegex(ValueError, "exceeds 10"):
            self.make([row()], max_prompt_length=10)

    def test_missing_id_empty_data_wrong_agent_and_missing_pad_are_rejected(self):
        missing = row()
        missing["extra_info"].pop("sample_id")
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            self.make([missing])
        with self.assertRaisesRegex(ValueError, "Empty selected"):
            self.make([])
        wrong = row()
        wrong["agent_name"] = "tool_agent"
        with self.assertRaisesRegex(ValueError, "Unexpected agent_name"):
            self.make([wrong])
        dataset, tokenizer = self.make([row()])
        tokenizer.pad_token_id = None
        with self.assertRaisesRegex(ValueError, "pad token"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
