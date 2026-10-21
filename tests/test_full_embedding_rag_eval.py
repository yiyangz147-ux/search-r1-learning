"""Behavioral tests for batched rollout parity, token accounting and resume."""

from pathlib import Path
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_full_embedding_rag as evaluator
import sft_reward_monitor as monitor


def example(index):
    return evaluator.normalize_example({"id": f"sample-{index}", "question": f"Question {index}",
                                         "data_source": "nq", "gold_answers": ["Paris"]}, index)


def answer(value="Paris"):
    return f"<think>I have sufficient evidence.</think><answer>{value}</answer>"


def search(query="France capital"):
    return f"<think>I should look this up.</think><search>{query}</search>"


def retrieve(query):
    return {"query": query, "content": "<information>Doc 1: Paris.</information>",
            "documents": [{"id": "wiki-1", "contents": "Paris."}]}


class FakeBatch:
    def __init__(self, plans, *, tokens=10):
        self.plans = {index: iter(plan) for index, plan in plans.items()}
        self.tokens = tokens
        self.calls = []

    def __call__(self, contexts, budgets):
        self.calls.append((list(contexts), list(budgets)))
        packets = []
        for context, budget in zip(contexts, budgets):
            index = next(index for index in self.plans if f"Question: Question {index}?\n" in context)
            text = next(self.plans[index])
            if text is None:
                packets.append({"context_limit": True})
            else:
                packets.append({"text": text, "generated_tokens": min(self.tokens, budget),
                                "input_tokens": 40, "stop_reason": "action_stop"})
        return packets


class BatchedRolloutTests(unittest.TestCase):
    def test_batched_rollouts_match_training_monitor_question_for_question(self):
        plans = {
            0: [search(), answer()],
            1: ["Translation:\n<answer>Paris</answer>"],
            2: ["nonsense", "still nonsense"],
            3: [search("one"), search("two"), answer()],
            4: [None],
            5: [search("a"), search("b"), search("c")],
        }
        config = {"batch_size": 3, "max_searches": 1, "max_turns": 3}
        batched = evaluator.batched_rollouts([example(index) for index in plans],
                                             FakeBatch(plans), retrieve, config)
        by_id = {row["sample_id"]: row for row in batched}
        for index, plan in plans.items():
            single = FakeBatch({index: plan})
            record = monitor.rollout(example(index), lambda context, budget: single([context], [budget])[0],
                                     retrieve, config)
            batch_record = by_id[record["sample_id"]]
            for key in ("answer", "status", "total_reward", "format_reward", "qa_em", "qa_f1",
                        "search_count", "generated_tokens", "model_calls", "initial_prompt"):
                with self.subTest(index=index, key=key):
                    self.assertEqual(batch_record[key], record[key])
            for batch_event, event in zip(batch_record["events"], record["events"]):
                self.assertEqual({key: batch_event[key] for key in event}, event)

    def test_dynamic_refill_and_per_question_token_budget(self):
        plans = {0: [answer()], 1: [search(), "bad"], 2: [answer()]}
        generator = FakeBatch(plans)
        config = {"batch_size": 2, "max_new_tokens": 10, "max_total_generated_tokens": 15}
        records = evaluator.batched_rollouts([example(index) for index in plans], generator, retrieve, config)
        self.assertEqual(len(generator.calls[0][0]), 2)
        self.assertEqual(generator.calls[1][1], [5, 10])
        self.assertIn("Question: Question 2?", generator.calls[1][0][1])
        one = next(row for row in records if row["sample_index"] == 1)
        self.assertEqual(one["status"], "generated_token_limit")
        self.assertEqual(one["generated_tokens"], 15)
        self.assertEqual(one["format_reward"], 0)

    def test_gold_is_not_prefilled_and_boundary_newlines_match(self):
        generator = FakeBatch({0: [search() + "\n", answer()]})
        evaluator.batched_rollouts([example(0)], generator, retrieve, {"batch_size": 1})
        self.assertNotIn("Paris", generator.calls[0][0][0])
        self.assertIn("</search>\n<information>", generator.calls[1][0][0])
        self.assertNotIn("</search>\n\n<information>", generator.calls[1][0][0])

    def test_infrastructure_failure_is_propagated_without_a_failure_reward(self):
        saved = []

        def fail(_query):
            raise RuntimeError("retriever offline")

        with self.assertRaisesRegex(RuntimeError, "retriever offline"):
            evaluator.batched_rollouts([example(0)], FakeBatch({0: [search()]}), fail,
                                       {"batch_size": 1}, on_record=saved.append)
        self.assertEqual(saved, [])

    def test_bad_generator_cannot_overrun_budget(self):
        with self.assertRaisesRegex(RuntimeError, "over-budget"):
            evaluator.batched_rollouts([example(0)], lambda contexts, budgets: [
                {"text": answer(), "generated_tokens": 513, "input_tokens": 20}],
                retrieve, {"batch_size": 1})

    def test_native_retrieval_batches_one_round_without_changing_trajectories(self):
        plans = {0: [search("first-query"), answer()], 1: [search("second-query"), answer()]}
        config = {"batch_size": 2}
        requests = []

        def many(queries):
            requests.append(queries)
            return [retrieve(query) for query in queries]

        records = evaluator.batched_rollouts([example(0), example(1)], FakeBatch(plans),
                                             lambda _: self.fail("single retrieval should not be used"),
                                             config, retrieve_many=many)
        sequential = evaluator.batched_rollouts([example(0), example(1)], FakeBatch(plans), retrieve, config)
        self.assertEqual(requests, [["first-query", "second-query"]])
        self.assertNotIn("Paris", json.dumps(requests))
        for actual, expected in zip(records, sequential):
            for key in ("events", "status", "answer", "search_count", "generated_tokens",
                        "total_reward", "format_reward"):
                self.assertEqual(actual[key], expected[key])

    def test_completed_answer_is_saved_before_a_sibling_retrieval_batch_fails(self):
        saved = []

        def fail(_queries):
            raise RuntimeError("retriever batch offline")

        with self.assertRaisesRegex(RuntimeError, "retriever batch offline"):
            evaluator.batched_rollouts([example(0), example(1)],
                                       FakeBatch({0: [search()], 1: [answer()]}), retrieve,
                                       {"batch_size": 2}, retrieve_many=fail, on_record=saved.append)
        self.assertEqual([row["sample_index"] for row in saved], [1])
        self.assertEqual(saved[0]["total_reward"], 1)


class NativeRetrievalTests(unittest.TestCase):
    def test_http_query_batch_keeps_each_passage_matched_and_html_escaped(self):
        calls = []

        class Opener:
            def open(self, request, timeout):
                calls.append(json.loads(request.data))
                return io.BytesIO(json.dumps({"result": [
                    [{"document": {"id": "one", "contents": "First <title> & fact"}, "score": 0.8}],
                    [{"document": {"id": "two", "contents": "Second passage"}, "score": 0.7}],
                ]}).encode())

        with patch.object(evaluator, "build_opener", return_value=Opener()):
            results = evaluator.retrieve_many(["first query", "second query"], {})
        self.assertEqual(calls, [{"queries": ["first query", "second query"], "topk": 3,
                                  "return_scores": True}])
        self.assertEqual([row["query"] for row in results], ["first query", "second query"])
        self.assertEqual([row["documents"][0]["id"] for row in results], ["one", "two"])
        self.assertEqual(results[0]["content"],
                         "<information>Doc 1: First &lt;title&gt; &amp; fact</information>")
        self.assertEqual(results[1]["batch_index"], 1)
        self.assertEqual(results[0]["transport_attempts"], 1)

    def test_empty_passage_or_incorrect_result_count_raises(self):
        for raw in [{"result": []}, {"result": [[]]}, {"result": [[{"document": {"contents": ""}}]]}]:
            class Opener:
                def open(self, request, timeout):
                    return io.BytesIO(json.dumps(raw).encode())
            with self.subTest(raw=raw), patch.object(evaluator, "build_opener", return_value=Opener()):
                with self.assertRaises(RuntimeError):
                    evaluator.retrieve_many(["query"], {})


class GenerationMicrobatchTests(unittest.TestCase):
    def fake_generator(self, config):
        generator = object.__new__(evaluator.BatchedGenerator)
        generator.config = config
        calls = []

        def tokenize(contexts, **kwargs):
            return {"input_ids": [[1] * int(context) for context in contexts]}

        def generate(rows, budget):
            calls.append(([row[0] for row in rows], budget))
            return [(row[0], {"index": row[0], "budget": budget, "context_limited": row[2]})
                    for row in rows]

        generator.tokenizer, generator._generate = tokenize, generate
        return generator, calls

    def test_active_pool_32_uses_four_model_microbatches_of_eight(self):
        generator, calls = self.fake_generator({"max_context_tokens": 8192})
        results = generator(["100"] * 32, [512] * 32)
        self.assertEqual(calls, [(list(range(start, start + 8)), 512) for start in range(0, 32, 8)])
        self.assertEqual([packet["index"] for packet in results], list(range(32)))

    def test_mixed_budgets_context_limits_and_microbatch_output_order(self):
        generator, calls = self.fake_generator({"max_context_tokens": 100, "generation_microbatch_size": 1})
        results = generator(["20", "92", "30", "92", "100"], [10, 10, 5, 5, 10])
        self.assertEqual(calls, [([0], 10), ([1], 8), ([2], 5), ([3], 5)])
        self.assertEqual([packet.get("index") for packet in results], [0, 1, 2, 3, None])
        self.assertEqual([packet.get("budget") for packet in results], [10, 8, 5, 5, None])
        self.assertTrue(results[1]["context_limited"])
        self.assertFalse(results[3]["context_limited"])
        self.assertEqual(results[4], {"context_limit": True})
        generator.config["generation_microbatch_size"] = 0
        with self.assertRaisesRegex(ValueError, "generation_microbatch_size"):
            generator(["20"], [10])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "worker-00.jsonl"
        self.examples = {0: example(0), 1: example(1)}
        self.record = evaluator.batched_rollouts([example(0)], FakeBatch({0: [answer()]}), retrieve,
                                                {"batch_size": 1})[0]
        self.record.update(worker_index=0, run_binding_sha256="binding")

    def tearDown(self):
        self.directory.cleanup()

    def load(self):
        return evaluator.load_progress(self.path, self.examples, 0, 1, "binding")

    def write(self, raw):
        self.path.write_bytes(raw)

    def line(self):
        return (evaluator.canonical(self.record) + "\n").encode()

    def test_resume_skips_only_verified_terminal_records(self):
        self.write(self.line())
        loaded = self.load()
        self.assertEqual(list(loaded), [0])
        self.assertEqual(loaded[0]["answer"], "Paris")

    def test_repairs_partial_final_write_but_preserves_completed_line(self):
        self.write(self.line() + b'{"sample_index":1,"events":[')
        self.assertEqual(len(self.load()), 1)
        self.assertEqual(self.path.read_bytes(), self.line())

    def test_valid_final_record_without_newline_is_kept(self):
        self.write(self.line().rstrip(b"\n"))
        self.assertEqual(len(self.load()), 1)
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))

    def test_malformed_complete_line_is_not_deleted(self):
        original = self.line() + b'{"broken"\n'
        self.write(original)
        with self.assertRaisesRegex(ValueError, "Malformed complete"):
            self.load()
        self.assertEqual(self.path.read_bytes(), original)

    def test_binding_reward_shard_and_duplicates_are_rejected(self):
        for change in [{"run_binding_sha256": "different"}, {"total_reward": 0},
                       {"sample_index": 1}, {"status": "infrastructure_error"},
                       {"generated_tokens": 999}]:
            record = {**self.record, **change}
            self.write((evaluator.canonical(record) + "\n").encode())
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()
        self.write(self.line() + self.line())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_journal_fsync_resume_and_duplicate_protection(self):
        completed = {}
        journal = evaluator.Journal(self.path, completed, 0, "binding", 2)
        try:
            with redirect_stdout(io.StringIO()):
                journal.append(self.record)
            self.assertEqual(list(completed), [0])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                journal.append(self.record)
        finally:
            journal.close()
        self.assertEqual(len(self.load()), 1)


if __name__ == "__main__":
    unittest.main()
