"""Protocol fixtures check evaluation integrity; they are not a real RAG benchmark."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_rag as rag


def example(source, index):
    return {"sample_id": f"{source}-{index}", "data_source": source, "partition": "validation",
            "question": f"What is the capital in fixture {source} {index}?",
            "golden_answers": ["Paris", "the Paris"], "stratum": "all",
            "context": "GOLD_CONTEXT_MUST_NOT_BE_SENT", "supporting_facts": ["DO_NOT_SEND"]}


class FixtureServer(BaseHTTPRequestHandler):
    model_requests = []
    model_paths = []
    retrieval_requests = []
    mode = "normal"

    def log_message(self, *args):
        pass

    def reply(self, data, status=200):
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        self.reply({"data": [{"id": "Qwen/Qwen3-4B"}, {"id": "Qwen/Qwen3-4B-Base"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/retrieve":
            type(self).retrieval_requests.append(body)
            # Deliberate trap: only one of these two passages may enter top-k=1 input.
            self.reply({"result": [[
                {"document": {"id": "1", "contents": '"France"\nParis is the capital.'}, "score": 0.9},
                {"document": {"id": "2", "contents": "MUST_NOT_BE_SHOWN"}, "score": 0.8},
            ]]})
            return
        type(self).model_requests.append(body)
        type(self).model_paths.append(self.path)
        if type(self).mode == "error":
            self.reply({"error": "do not log server secrets"}, 400)
            return
        if self.path == "/v1/completions":
            question = body["prompt"].rsplit("\nQuestion: ", 1)[1].split("\nAnswer:", 1)[0]
        else:
            question = json.loads(body["messages"][1]["content"])["question"]
        # Keep a wrong answer in the cohort to detect correctness-based filtering.
        answer = "London" if "hotpotqa" in question else "The Paris!"
        choice = {"finish_reason": "length" if type(self).mode == "truncated" else "stop"}
        if self.path == "/v1/completions":
            choice["text"] = answer
        else:
            choice["message"] = {"role": "assistant", "content": f"<answer>{answer}</answer>"}
        self.reply({"id": "fixture", "model": body["model"], "system_fingerprint": "fixture-only",
                    "choices": [choice],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "val.jsonl"
        rag.write_jsonl(self.data, [example("nq", i) for i in range(3)] + [example("hotpotqa", i) for i in range(3)])
        FixtureServer.mode = "normal"
        FixtureServer.model_requests = []
        FixtureServer.model_paths = []
        FixtureServer.retrieval_requests = []

    def tearDown(self):
        self.temp.cleanup()

    def args(self, **kwargs):
        base = f"http://127.0.0.1:{self.server.server_port}"
        defaults = dict(data_file=self.data, output_dir=self.root / "eval", model="Qwen/Qwen3-4B",
                        model_revision="fixture-only", base_url=base + "/v1", retriever_url=base + "/retrieve",
                        mode="rag", input_format="chat", seed=42, topk=1, passage_chars=25, thinking="off", temperature=0.0,
                        top_p=1.0, top_k=-1, max_tokens=512, limit=0, max_new=0, dry_run=False,
                        report_only=False, preflight_only=False, concurrency=2, timeout=3,
                        api_key_file=None, api_key_env="RAG_FIXTURE_KEY", retry_errors=False)
        return argparse.Namespace(**(defaults | kwargs))

    def run_quietly(self, args):
        with redirect_stdout(io.StringIO()), patch.dict("os.environ", {"RAG_FIXTURE_KEY": "PRIVATE_TEST_KEY"}):
            rag.run(args)

    def summary(self, args):
        return json.loads((args.output_dir / "summary.json").read_text())

    def test_em_is_strict_and_f1_keeps_partial_credit(self):
        self.assertEqual(rag.score_answer("The Paris!", ["Paris"]), {"em": 1, "f1": 1.0})
        self.assertEqual(rag.score_answer("Paris, France", ["Paris"])["em"], 0)
        self.assertAlmostEqual(rag.score_answer("Paris, France", ["Paris"])["f1"], 2 / 3)
        self.assertEqual(rag.score_answer("NYC", ["New York", "NYC"])["em"], 1)
        self.assertEqual(rag.score_answer("", ["Paris"]), {"em": 0, "f1": 0.0})
        self.assertEqual(rag.score_answer("A+", ["A+", "AB+"]), {"em": 1, "f1": 0.0})
        self.assertEqual(rag.extract_answer("<answer>!!!</answer>"), ("!!!", "tagged"))

    def test_extraction_does_not_score_thinking_or_unfinished_tags(self):
        self.assertEqual(rag.extract_answer("<think><answer>London</answer></think><answer>Paris</answer>"), ("Paris", "tagged"))
        self.assertEqual(rag.extract_answer("Thinking without opening tag</think>Paris"), ("Paris", "plain"))
        self.assertEqual(rag.extract_answer("<think>Paris"), ("", "unfinished_thinking"))
        self.assertEqual(rag.extract_answer("<answer>Paris"), ("", "invalid_tags"))
        self.assertEqual(rag.extract_answer("<answer>Paris</answer><answer>London</answer>"), ("", "invalid_tags"))

    def test_sampling_reproducible_balanced_and_training_rejected(self):
        examples = rag.load_examples(self.data)
        selected = rag.select_examples(examples, 4, 42)
        self.assertEqual(Counter(row["data_source"] for row in selected), {"nq": 2, "hotpotqa": 2})
        self.assertEqual(selected, rag.select_examples(list(reversed(examples)), 4, 42))
        self.assertEqual(len({row["sample_id"] for row in selected}), 4)
        bad = self.root / "train.jsonl"
        rag.write_jsonl(bad, [{**example("nq", 0), "partition": "sft_initial"}])
        with self.assertRaisesRegex(ValueError, "held-out"):
            rag.load_examples(bad)

    def test_dry_run_has_no_network_no_scores_and_no_gold_in_model_request(self):
        args = self.args(dry_run=True)
        with patch.object(rag, "http_json", side_effect=AssertionError("dry run called network")):
            self.run_quietly(args)
        preview = json.loads((args.output_dir / "preview.json").read_text())
        request = json.dumps(preview["model_request_before_retrieval"])
        self.assertNotIn("golden_answers", request)
        self.assertNotIn("Paris", request)
        self.assertNotIn("GOLD_CONTEXT", request)
        self.assertFalse((args.output_dir / "predictions.jsonl").exists())
        self.assertIsNone(self.summary(args)["overall"]["em"])
        self.assertEqual(self.summary(args)["status"], "prepared")

    def test_real_http_eval_resume_retains_wrong_answers_and_evidence(self):
        args = self.args(max_new=2)
        self.run_quietly(args)
        summary = self.summary(args)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["overall"]["pending"], 4)
        self.assertEqual(summary["overall"]["em"], 0.5)
        args.max_new = 0
        self.run_quietly(args)
        self.assertEqual(len(FixtureServer.model_requests), 6)
        self.assertEqual(self.summary(args)["overall"]["accuracy_percent"], 50.0)
        self.assertEqual(self.summary(args)["by_source"]["nq"]["em"], 1.0)
        self.assertEqual(self.summary(args)["by_source"]["hotpotqa"]["em"], 0.0)
        self.run_quietly(args)
        self.assertEqual(len(FixtureServer.model_requests), 6, "resume must not repeat correct or wrong answers")
        for request in FixtureServer.model_requests:
            self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
            serialized = json.dumps(request)
            self.assertNotIn("GOLD_CONTEXT", serialized)
            self.assertNotIn("golden_answers", serialized)
            self.assertNotIn("MUST_NOT_BE_SHOWN", serialized)
            self.assertLessEqual(len(json.loads(request["messages"][1]["content"])["retrieved_evidence"][0]["contents"]), 25)
        results_text = (args.output_dir / "predictions.jsonl").read_text()
        self.assertNotIn("PRIVATE_TEST_KEY", results_text)
        self.assertIn('"retrieval_raw"', results_text)
        self.assertEqual(self.summary(args)["usage_all_attempts"]["total_tokens"], 90)
        args.topk = 2
        with self.assertRaisesRegex(ValueError, "Resume settings"):
            self.run_quietly(args)

    def test_no_rag_skips_retriever_and_uses_same_selection(self):
        args = self.args(mode="no-rag")
        self.run_quietly(args)
        self.assertEqual(FixtureServer.retrieval_requests, [])
        for request in FixtureServer.model_requests:
            self.assertNotIn("retrieved_evidence", json.loads(request["messages"][1]["content"]))
        saved = [item for _, item in rag.rows(args.output_dir / "selected.jsonl")]
        self.assertEqual([row["sample_id"] for row in saved],
                         [row["sample_id"] for row in rag.select_examples(rag.load_examples(self.data), 0, 42)])

    def test_base_completions_use_raw_prompt_plain_answer_and_resume(self):
        args = self.args(input_format="base", model="Qwen/Qwen3-4B-Base", max_new=2)
        self.run_quietly(args)
        self.assertEqual(FixtureServer.model_paths, ["/v1/completions"] * 2)
        summary = self.summary(args)
        self.assertEqual(summary["model"], "Qwen/Qwen3-4B-Base")
        self.assertEqual(summary["input_format"], "base")
        self.assertEqual(summary["overall"]["em"], 0.5)
        for payload in FixtureServer.model_requests:
            self.assertNotIn("messages", payload)
            self.assertNotIn("chat_template_kwargs", payload)
            self.assertEqual(payload["stop"], ["\n"])
            self.assertFalse(payload["echo"])
            self.assertFalse(payload["add_special_tokens"])
            self.assertTrue(payload["prompt"].endswith("\nAnswer:"))
            self.assertIn("Passage 1:", payload["prompt"])
            self.assertNotIn("GOLD_CONTEXT", payload["prompt"])
            self.assertNotIn("MUST_NOT_BE_SHOWN", payload["prompt"])
            self.assertNotIn("golden_answers", payload["prompt"])
        records = [row for _, row in rag.rows(args.output_dir / "predictions.jsonl")]
        self.assertEqual({row["answer_format"] for row in records}, {"plain"})
        args.max_new = 0
        self.run_quietly(args)
        self.run_quietly(args)
        self.assertEqual(len(FixtureServer.model_requests), 6)
        self.assertTrue(self.summary(args)["complete"])
        self.assertEqual(self.summary(args)["overall"]["em"], 0.5)

    def test_base_no_rag_prompt_has_no_evidence_and_keeps_cohort(self):
        args = self.args(input_format="base", model="Qwen/Qwen3-4B-Base", mode="no-rag", limit=2)
        self.run_quietly(args)
        self.assertEqual(FixtureServer.retrieval_requests, [])
        for request in FixtureServer.model_requests:
            self.assertNotIn("Passages:", request["prompt"])
            self.assertNotIn("Paris", request["prompt"], "No gold answer may enter no-RAG prompt")
        self.assertEqual(self.summary(args)["overall"]["em"], 0.5)
        args.input_format = "chat"
        with self.assertRaisesRegex(ValueError, "Resume settings"):
            self.run_quietly(args)

    def test_cli_defaults_to_base_and_rejects_base_thinking_switch(self):
        script = Path(rag.__file__)
        args = [sys.executable, str(script), "--dry-run", "--limit", "2", "--data-file", str(self.data),
                "--output-dir", str(self.root / "cli-base")]
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        preview = json.loads(result.stdout)
        payload = preview["model_request_before_retrieval"]
        self.assertEqual(payload["model"], "Qwen/Qwen3-4B-Base")
        self.assertIn("prompt", payload)
        self.assertNotIn("messages", payload)
        self.assertIsNone(json.loads((self.root / "cli-base/summary.json").read_text())["overall"]["em"])
        invalid = subprocess.run(args + ["--thinking", "on"], capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("chat-template setting", invalid.stderr)

    def test_base_truncation_is_zero_and_malformed_chat_response_is_error(self):
        args = self.args(input_format="base", model="Qwen/Qwen3-4B-Base", mode="no-rag")
        FixtureServer.mode = "truncated"
        result = rag.evaluate_one(example("nq", 0), args, None)
        self.assertEqual(result["status"], "unfinished_generation")
        self.assertEqual(result["em"], 0)
        chat_shaped = {"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}]}
        with patch.object(rag, "http_json", return_value=chat_shaped):
            result = rag.evaluate_one(example("nq", 0), args, None)
        self.assertEqual(result["status"], "error")
        self.assertIn("choices[0].text", result["error"])

    def test_truncated_generation_counts_as_zero_even_with_answer(self):
        FixtureServer.mode = "truncated"
        args = self.args(limit=2)
        self.run_quietly(args)
        summary = self.summary(args)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["status"], "complete_with_failures")
        self.assertEqual(summary["overall"]["em"], 0.0)
        self.assertEqual(summary["overall"]["status_counts"], {"unfinished_generation": 2})

    def test_service_errors_stop_and_retry_does_not_shrink_denominator(self):
        FixtureServer.mode = "error"
        args = self.args(limit=4)
        with self.assertRaises(RuntimeError):
            self.run_quietly(args)
        summary = self.summary(args)
        self.assertEqual(summary["overall"]["attempted"], 2)
        self.assertEqual(summary["overall"]["status_counts"], {"error": 2})
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["overall"]["em"], 0.0)
        FixtureServer.mode = "normal"
        args.retry_errors = True
        self.run_quietly(args)
        summary = self.summary(args)
        self.assertEqual(summary["overall"]["attempted"], 4)
        self.assertEqual(summary["attempt_records"], 6)
        self.assertEqual(summary["overall"]["em"], 0.5)
        self.assertTrue(summary["complete"])
        args.report_only = True
        with patch.object(rag, "http_json", side_effect=AssertionError("report-only called network")):
            self.run_quietly(args)


if __name__ == "__main__":
    unittest.main()
