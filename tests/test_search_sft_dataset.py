"""Tokenization fixtures for Base continuation and explicit chat-format SFT."""

from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import PROMPT
from search_sft_dataset import SearchSFTDataset


class BaseTokenizer:
    """Fast-tokenizer stub without a chat_template or pretrained files."""

    is_fast = True
    eos_token_id = 100000

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("A Base continuation must not call a chat template")

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class ChatTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "[user]" + messages[0]["content"] + "[assistant]<think>\n"


class SearchSFTDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "steps.parquet"

    def tearDown(self):
        self.temp.cleanup()

    def write_steps(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.path)

    def test_default_base_uses_raw_prompt_and_only_supervises_current_action(self):
        question = PROMPT.format(question="What is the capital of France?")
        search = "<think>Find the capital.</think><search>France capital</search>"
        history = search + "<information>Paris is the capital of France.</information>"
        answer = "<think>The evidence says Paris.</think><answer>Paris</answer>"
        self.write_steps([
            {"sample_id": "france", "question": question, "history": "", "response": search},
            {"sample_id": "france", "question": question, "history": history, "response": answer},
        ])
        tokenizer = BaseTokenizer()
        self.assertFalse(hasattr(tokenizer, "chat_template"))
        dataset = SearchSFTDataset(self.path, tokenizer)

        self.assertEqual([text for text, _ in tokenizer.calls], [question + search, question + history + answer])
        for index, (prefix, response) in enumerate([(question, search), (question + history, answer)]):
            item = dataset[index]
            final = index == 1
            expected_ids = [ord(character) for character in prefix + response]
            expected_labels = [-100] * len(prefix) + [ord(character) for character in response]
            if final:
                expected_ids.append(tokenizer.eos_token_id)
                expected_labels.append(tokenizer.eos_token_id)
            self.assertEqual(item["input_ids"], expected_ids)
            self.assertEqual(item["labels"], expected_labels)
            self.assertEqual(item["attention_mask"], [1] * len(expected_ids))
            self.assertEqual(tokenizer.calls[index][1], {"add_special_tokens": False, "return_offsets_mapping": True})

    def test_explicit_chat_keeps_template_options_and_strips_preopened_think(self):
        question = "Question: What is the capital of France?\n"
        response = "<think>Use evidence.</think><answer>Paris</answer>"
        self.write_steps([{"sample_id": "france", "question": question, "history": "", "response": response}])
        tokenizer = ChatTokenizer()
        dataset = SearchSFTDataset(self.path, tokenizer, prompt_format="chat")

        self.assertEqual(tokenizer.template_calls, [(
            [{"role": "user", "content": question}],
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True},
        )])
        self.assertEqual(tokenizer.calls[0][0], "[user]" + question + "[assistant]" + response)
        supervised = "".join(chr(label) for label in dataset[0]["labels"] if label not in {-100, tokenizer.eos_token_id})
        self.assertEqual(supervised, response)

    def test_invalid_prompt_format_is_rejected_before_reading_data(self):
        with self.assertRaisesRegex(ValueError, "prompt_format must be 'base' or 'chat'"):
            SearchSFTDataset(self.path, BaseTokenizer(), prompt_format="automatic")

    def test_token_crossing_prompt_response_boundary_is_masked(self):
        class BoundaryTokenizer(BaseTokenizer):
            def __call__(self, text, **kwargs):
                self.assert_text = text
                # The second token spans the last context character and first
                # response character, so supervising it would leak context loss.
                return {"input_ids": [10, 20, 30, 40], "offset_mapping": [(0, 2), (2, 4), (4, 5), (5, 6)]}

        self.write_steps([{"sample_id": "boundary", "question": "a", "history": "bc", "response": "XYZ"}])
        tokenizer = BoundaryTokenizer()
        dataset = SearchSFTDataset(self.path, tokenizer, prompt_format="base")
        self.assertEqual(tokenizer.assert_text, "abcXYZ")
        self.assertEqual(dataset[0]["labels"], [-100, -100, 30, 40])

    def test_overlength_answer_drops_all_steps_for_that_question(self):
        search = "<search>capital</search>"
        self.write_steps([
            {"sample_id": "too_long", "question": "q", "history": "", "response": search},
            {"sample_id": "too_long", "question": "q", "history": "", "response": "<answer>" + "x" * 100 + "</answer>"},
            {"sample_id": "fits", "question": "q", "history": "", "response": search},
            {"sample_id": "fits", "question": "q", "history": "", "response": "<answer>Paris</answer>"},
        ])
        dataset = SearchSFTDataset(self.path, BaseTokenizer(), max_length=40)
        self.assertEqual(dataset.input_steps, 4)
        self.assertEqual(dataset.retained_steps, 2)
        self.assertEqual(dataset.dropped_sample_ids, {"too_long"})
        self.assertEqual(dataset.dropped["over_max_length"], 1)


if __name__ == "__main__":
    unittest.main()
