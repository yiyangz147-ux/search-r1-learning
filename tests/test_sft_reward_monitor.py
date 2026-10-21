"""Offline behavioral tests for real-rollout SFT reward monitoring."""

from pathlib import Path
import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sft_reward_monitor as monitor
from prepare_data import PROMPT


EXAMPLE = {
    "sample_id": "heldout-1", "question": "What is the capital of France?",
    "golden_answers": ["Paris"], "data_source": "nq",
}


def prompt(question):
    return PROMPT.format(question=question if question.endswith("?") else question + "?")


def information(content):
    return "\n" + content.strip() + "\n"


class FakeGeneration:
    def __init__(self, texts, token_count=10):
        self.texts = iter(texts)
        self.contexts = []
        self.budgets = []
        self.token_count = token_count

    def __call__(self, context, budget):
        self.contexts.append(context)
        self.budgets.append(budget)
        return {"text": next(self.texts), "generated_tokens": min(self.token_count, budget),
                "input_tokens": 20}


class RewardMonitorTests(unittest.TestCase):
    def run_rollout(self, outputs, config=None, retrieve_fn=None):
        generate = FakeGeneration(outputs)
        queries = []

        def retrieval(query):
            queries.append(query)
            return {"query": query, "content": "<information>Doc 1: Paris is the capital of France.</information>",
                    "documents": [{"id": "1", "contents": "Paris is the capital of France."}]}

        record = monitor.rollout(EXAMPLE, generate, retrieve_fn or retrieval, config or {},
                                 build_prompt_fn=prompt, canonical_information_fn=information)
        return record, generate, queries

    def test_recover_answer_from_broken_prefix_but_reject_format(self):
        text = "Translation:\n<answer>The Paris!</answer>"
        self.assertEqual(monitor.parse_action(text)["value"], "The Paris!")
        self.assertFalse(monitor.valid_step_format(text))
        record, _, _ = self.run_rollout([text])
        self.assertEqual(record["answer_reward"], 1)
        self.assertEqual(record["total_reward"], 1)
        self.assertEqual(record["format_reward"], 0)

    def test_first_complete_action_wins(self):
        action = monitor.parse_action("prose <search>France capital</search><answer>Paris</answer>")
        self.assertEqual(action["action"], "search")
        self.assertEqual(action["value"], "France capital")
        self.assertIsNone(monitor.parse_action("<answer>Paris"))
        self.assertIsNone(monitor.parse_action("<search> </search>"))

    def test_format_requires_one_complete_nonempty_think_then_action(self):
        good = "<think>I should verify this fact.</think>\n<search>France capital</search>"
        self.assertTrue(monitor.valid_step_format(good))
        for bad in [
            "<answer>Paris</answer>",
            "<think></think><answer>Paris</answer>",
            "<think>why<search>x</search></think><answer>Paris</answer>",
            "<think>why</think><answer>Paris</answer> trailing prose",
            "<think>why</think><answer><answer>Paris</answer></answer>",
            "<think>why</think><answer>Paris</answer><answer>London</answer>",
            "<think>why</think><search>" + "x" * 501 + "</search>",
        ]:
            with self.subTest(bad=bad):
                self.assertFalse(monitor.valid_step_format(bad))

    def test_real_observation_and_question_only_reach_generation(self):
        record, generate, queries = self.run_rollout([
            "<think>I need evidence.</think><search>France capital</search>",
            "<think>The document gives the capital.</think><answer>Paris</answer>",
        ])
        self.assertEqual(queries, ["France capital"])
        self.assertEqual(record["format_reward"], 1)
        self.assertEqual(record["total_reward"], 1)
        self.assertEqual(record["search_count"], 1)
        self.assertEqual(generate.contexts[0], prompt(EXAMPLE["question"]))
        self.assertNotIn("Paris", generate.contexts[0])
        self.assertIn("</search>\n<information>Doc 1:", generate.contexts[1])
        self.assertTrue(generate.contexts[1].endswith("</information>\n"))

    def test_generated_closing_newline_is_not_duplicated_in_next_prompt(self):
        _, generate, _ = self.run_rollout([
            " \n<think>Look up.</think><search>France capital</search>\n",
            "<think>Answer.</think><answer>Paris</answer>",
        ])
        self.assertIn("</search>\n<information>", generate.contexts[1])
        self.assertNotIn("</search>\n\n<information>", generate.contexts[1])
        self.assertIn("?\n<think>", generate.contexts[1])

    def test_invalid_feedback_continues_without_erasing_correct_answer(self):
        record, generate, queries = self.run_rollout([
            "Translation: incomplete",
            "<think>I can now answer.</think><answer>Paris</answer>",
        ])
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["total_reward"], 1)
        self.assertEqual(record["format_reward"], 0)
        self.assertIn("valid complete action", generate.contexts[1])
        self.assertEqual(queries, [])

    def test_invalid_actions_stop_at_configured_limit(self):
        record, _, _ = self.run_rollout(["nonsense", "still nonsense"])
        self.assertEqual(record["status"], "invalid_action_limit")
        self.assertEqual(record["model_calls"], 2)
        self.assertEqual(record["total_reward"], 0)
        self.assertEqual(record["format_reward"], 0)

    def test_search_limit_sends_feedback_and_never_calls_service_again(self):
        record, generate, queries = self.run_rollout([
            "<think>Look up.</think><search>one</search>",
            "<think>Look up again.</think><search>two</search>",
            "<think>Answer.</think><answer>Paris</answer>",
        ], {"max_searches": 1})
        self.assertEqual(queries, ["one"])
        self.assertIn("search budget is exhausted", generate.contexts[-1])
        self.assertEqual(record["total_reward"], 1)
        self.assertEqual(record["format_reward"], 0)

    def test_retriever_failure_is_not_scored_as_model_failure(self):
        def failure(query):
            raise RuntimeError("retriever offline")
        with self.assertRaisesRegex(RuntimeError, "retriever offline"):
            self.run_rollout(["<think>Look up.</think><search>one</search>"], retrieve_fn=failure)

    def test_token_budget_bounds_recovery_loop(self):
        record, generate, _ = self.run_rollout(
            ["bad", "still bad"], {"max_total_generated_tokens": 15})
        self.assertEqual(generate.budgets, [15, 5])
        self.assertEqual(record["generated_tokens"], 15)
        self.assertEqual(record["total_reward"], 0)

    def test_unfinished_trajectory_gets_zero_format_even_if_valid_search(self):
        record, _, _ = self.run_rollout(
            ["<think>Look up.</think><search>France capital</search>"], {"max_turns": 1})
        self.assertEqual(record["status"], "turn_limit")
        self.assertEqual(record["format_reward"], 0)
        self.assertEqual(record["total_reward"], 0)

    def test_prompt_example_answer_is_never_scored(self):
        self.assertIn("<answer> Beijing </answer>", prompt(EXAMPLE["question"]))
        record, _, _ = self.run_rollout(["nonsense", "nonsense"])
        self.assertEqual(record["answer"], "")
        self.assertEqual(record["qa_em"], 0)

    def test_total_reward_is_not_answer_plus_format(self):
        record, _, _ = self.run_rollout(
            ["<think>Known fact.</think><answer>Paris</answer>"])
        self.assertEqual(record["total_reward"], 1)
        self.assertEqual(record["format_reward"], 1)
        with self.assertRaisesRegex(ValueError, "format_weight"):
            monitor.validate_config({"format_weight": 0.1})

    def test_dataset_em_and_search_r1_em_are_explicitly_separate(self):
        score = monitor.score_trajectory("Café", ["Cafe\u0301"], "nq",
                                         completed=True, all_steps_valid=True)
        self.assertEqual(score["qa_em"], 1)  # NQ applies Unicode NFD.
        self.assertEqual(score["answer_reward"], 0)  # Search-R1 does not.
        self.assertEqual(score["format_reward"], 1)

    def test_aggregation_uses_all_examples_not_only_completed_answers(self):
        good, _, _ = self.run_rollout(["<think>Known.</think><answer>Paris</answer>"])
        bad, _, _ = self.run_rollout(["nonsense", "nonsense"])
        metrics = monitor.aggregate_records([good, bad], 3.5)
        self.assertEqual(metrics["reward/num_examples"], 2)
        self.assertEqual(metrics["reward/answer_reward"], 0.5)
        self.assertEqual(metrics["reward/format_reward"], 0.5)
        self.assertEqual(metrics["reward/answer_rate"], 0.5)
        self.assertEqual(metrics["reward/elapsed_seconds"], 3.5)
        with self.assertRaises(ValueError):
            monitor.aggregate_records([], 0)

    def test_configuration_rejects_unbounded_or_invalid_settings(self):
        for config in [{"max_turns": 0}, {"max_new_tokens": True},
                       {"retriever_url": "http://user:secret@localhost/retrieve"},
                       {"retriever_url": "file:///etc/passwd"}, {"retrieval_retries": -1}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                monitor.validate_config(config)


@unittest.skipUnless(os.environ.get("SFT_REWARD_TEST_TOKENIZER"),
                     "Set SFT_REWARD_TEST_TOKENIZER for the tiny CPU generation smoke test")
class TinyGenerationTests(unittest.TestCase):
    def test_generation_artifacts_rng_training_state_and_no_gradients(self):
        import numpy as np
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(os.environ["SFT_REWARD_TEST_TOKENIZER"],
                                                  local_files_only=True, use_fast=True)
        config = Qwen3Config(
            vocab_size=max(tokenizer.get_vocab().values()) + 1,
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, use_cache=False,
        )
        model = Qwen3ForCausalLM(config).cpu().train()
        model.model.norm.eval()  # Mixed submodule state must survive.
        modes = [module.training for module in model.modules()]
        trainable = [parameter.requires_grad for parameter in model.parameters()]
        forward_cache_flags = []
        hook = model.register_forward_pre_hook(
            lambda module, args, kwargs: forward_cache_flags.append(kwargs.get("use_cache")),
            with_kwargs=True)
        torch_before = torch.random.get_rng_state().clone()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rank0.jsonl"
            metrics = monitor.run_reward_eval(
                model, tokenizer, [EXAMPLE],
                {"max_new_tokens": 1, "max_turns": 1, "max_invalid_actions": 1},
                output)
            self.assertEqual(metrics["reward/num_examples"], 1)
            record = json.loads(output.read_text().strip())
            self.assertEqual(record["generated_tokens"], 1)
            self.assertEqual(record["search_count"], 0)
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text())["status"], "complete")
            self.assertTrue(output.with_suffix(".summary.json").is_file())
            with self.assertRaises(FileExistsError):
                monitor.run_reward_eval(model, tokenizer, [EXAMPLE], {}, output)
        hook.remove()
        self.assertEqual(forward_cache_flags, [True])
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(trainable, [parameter.requires_grad for parameter in model.parameters()])
        self.assertFalse(model.config.use_cache)
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
