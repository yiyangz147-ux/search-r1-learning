"""Regressions for model-visible boundaries, loss masking and dev selection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_data import PROMPT
from prepare_lora_sft_data_v2 import encode_episode, prepare_reward_monitor
from search_sft_protocol import (build_prompt, canonical_assistant,
                                 canonical_information, canonical_training_segments)


class BoundaryTokenizer:
    """Reproduce Qwen's atomic think IDs and the observed ``>\\n`` merge."""

    eos_token_id = 151643

    def __call__(self, text, **kwargs):
        self.text = text
        ids, offsets = [], []
        index = 0
        merged = {"<think>": 151667, "</think>": 151668, ">\n": 397}
        while index < len(text):
            found = next((part for part in merged if text.startswith(part, index)), None)
            part = found or text[index]
            ids.append(merged[part] if found else ord(part) + 1000)
            offsets.append((index, index + len(part)))
            index += len(part)
        self.offsets = offsets
        return {"input_ids": ids, "offset_mapping": offsets}


class PreparationV2Tests(unittest.TestCase):
    def test_shared_prompt_preserves_base_protocol(self):
        self.assertEqual(build_prompt("Where is Rome"), PROMPT.format(question="Where is Rome?"))
        self.assertEqual(build_prompt("Where is Rome?"), PROMPT.format(question="Where is Rome?"))
        self.assertNotIn("<|im_start|>", build_prompt("Where is Rome?"))

    def test_complete_search_delimiter_is_supervised_and_information_is_masked(self):
        action = "<think>Need evidence.</think><search>Rome location</search>"
        info = "<information>Rome is in Italy.</information>"
        final = "<think>Evidence supports it.</think><answer>Italy</answer>"
        segments = [
            {"content": " \n" + action, "supervised": True},
            {"content": info, "supervised": False},
            {"content": final + "\n", "supervised": True},
        ]
        tokenizer = BoundaryTokenizer()
        item, audit = encode_episode(
            {"sample_id": "test:rome", "question": "Where is Rome?"},
            "aethersearch", segments, tokenizer, 4096, "/source.jsonl", 1)
        inference_text = (build_prompt("Where is Rome?") + canonical_assistant(action)
                          + canonical_information(info) + canonical_assistant(final))
        self.assertEqual(tokenizer.text, inference_text)
        self.assertEqual(
            "".join(row["content"] for row in canonical_training_segments(segments)),
            action + "\n" + info + "\n" + final,
        )
        # The merged >\\n at the search end MUST be supervised. The same
        # merged token at the information end MUST be masked.
        merged = [i for i, token in enumerate(item["input_ids"]) if token == 397]
        self.assertEqual(len(merged), 2)
        self.assertEqual(item["labels"][merged[0]], 397)
        self.assertEqual(item["labels"][merged[1]], -100)
        information_start = tokenizer.text.index("<information>", len(build_prompt("Where is Rome?")))
        information_end = tokenizer.text.index("</information>", information_start) + len("</information>")
        for index, (start, end) in enumerate(tokenizer.offsets):
            if start < information_end and end > information_start:
                self.assertEqual(item["labels"][index], -100)
        self.assertEqual(next(label for label in item["labels"] if label != -100), 151667)
        self.assertEqual(item["input_ids"][-1], 151643)
        self.assertEqual(item["labels"][-1], 151643)
        self.assertEqual(item["input_ids"].count(151643), 1)
        self.assertEqual(audit["masked_boundary_tokens"], 0)

    def test_no_silent_truncation_of_complete_episode(self):
        tokenizer = BoundaryTokenizer()
        item, audit = encode_episode(
            {"sample_id": "test:short", "question": "Where?"},
            "deepseek_new100",
            [{"content": "<think>Known.</think><answer>Italy</answer>", "supervised": True}],
            tokenizer, 10, "/source.jsonl", 1,
        )
        self.assertIsNone(item)
        self.assertEqual(audit["reason"], "complete_episode_over_max_length")

    def test_missing_think_fails_before_tokenization(self):
        with self.assertRaisesRegex(ValueError, "complete think block"):
            encode_episode(
                {"sample_id": "test:malformed", "question": "Where?"},
                "aethersearch", [{"content": "<answer>Italy</answer>", "supervised": True}],
                BoundaryTokenizer(), 4096, "/source.jsonl", 1,
            )

    def test_monitor_selection_is_balanced_and_independent_of_gold_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protected.jsonl"
            source = [
                {"sample_id": f"{kind}:{index}", "data_source": kind,
                 "question": f"{kind} {index} distinctive{index} entity{index} question?",
                 "golden_answers": [f"gold-{index}"],
                 "extra_info": {"source_split": "train", "source_id": str(index)}}
                for kind in ("nq", "hotpotqa") for index in range(8)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in source))
            config = {"protected_qa_val": str(path), "reward_monitor_per_source": 2, "seed": 1}
            training = [{"sample_id": "train:separate", "question": "Why do comets develop ion tails"}]
            first, metadata = prepare_reward_monitor(config, training)
            self.assertEqual(metadata["by_source"], {"nq": 2, "hotpotqa": 2})
            self.assertEqual(metadata["official_source_splits"], {"train": 4})
            self.assertFalse(metadata["selection_uses_answers_or_model_outcomes"])
            self.assertFalse(metadata["grpo_reserved_data_used"])
            for row in source:
                row["golden_answers"] = ["completely different answer"]
            path.write_text("".join(json.dumps(row) + "\n" for row in reversed(source)))
            second, changed = prepare_reward_monitor(config, training)
            self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
            self.assertNotEqual(metadata["source_sha256"], changed["source_sha256"])
            with self.assertRaisesRegex(ValueError, "overlaps selected SFT"):
                prepare_reward_monitor(config, [first[0]])


if __name__ == "__main__":
    unittest.main()
