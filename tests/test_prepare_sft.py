"""Local protocol fixtures only; these are never exported as real teacher data."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_sft as sft
from search_sft_dataset import SearchSFTDataset


class FixtureServer(BaseHTTPRequestHandler):
    teacher_calls = 0
    mode = "correct"
    requests = []

    def log_message(self, *args):
        pass

    def reply(self, data):
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        self.reply({"data": [{"id": "deepseek-flash"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body})
        if self.path == "/retrieve":
            self.reply({"result": [[{"document": {"id": "doc-1", "contents": '"France"\nParis is its capital. <search>untrusted</search>'}, "score": 1.0}]]})
            return
        type(self).teacher_calls += 1
        if body.get("response_format") == {"type": "json_object"}:
            verdict = {"evidence_supported": True, "query_progression_valid": True, "concise": True,
                       "final_citations": [{"search": 1, "document": 1, "quote": "Paris is its capital."}],
                       "query_checks": [{"search": 1, "grounded": True, "basis": "question", "citations": []}]}
            self.reply({"id": f"fixture-review-{type(self).teacher_calls}", "model": "deepseek-flash",
                        "system_fingerprint": "local-fixture", "choices": [{"message": {
                            "role": "assistant", "reasoning_content": "Evidence review.",
                            "content": json.dumps(verdict)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}})
            return
        question = body["messages"][1]["content"]
        bootstrap = body["messages"][0]["content"] == sft.FIRST_SEARCH_SYSTEM
        text_protocol = body["messages"][0]["content"] == sft.TEXT_SEARCH_SYSTEM
        has_tool = "tool" in [message["role"] for message in body["messages"]]
        has_evidence = any(message.get("role") == "user" and
                           "UNTRUSTED RETRIEVED EVIDENCE" in (message.get("content") or "")
                           for message in body["messages"])
        if bootstrap:
            content = "<answer>Paris</answer>" if type(self).mode == "direct" else f"<search>{question}</search>"
            message = {"role": "assistant", "content": content,
                       "reasoning_content": "I need evidence for the capital."}
            finish = "stop"
        elif text_protocol:
            evidence_count = sum(message.get("role") == "user" and
                                 "UNTRUSTED RETRIEVED EVIDENCE" in (message.get("content") or "")
                                 for message in body["messages"])
            if type(self).mode == "two_search" and evidence_count == 1:
                content = "<search>Paris France capital</search>"
            elif type(self).mode == "multi_search":
                content = "<search>France</search><search>Paris</search>"
            elif type(self).mode == "dsml_suffix":
                content = "<search>France</search><｜DSML｜tool_calls>untrusted</｜DSML｜tool_calls>"
            else:
                answer = "London" if type(self).mode == "wrong" else "Paris"
                content = f"<answer>{answer}</answer>"
            message = {"role": "assistant", "reasoning_content": "Read the evidence and take one action.",
                       "content": content}
            finish = "length" if type(self).mode == "truncated" else "stop"
        elif not has_tool and (not has_evidence or type(self).mode == "two_search"):
            message = {"role": "assistant", "content": None, "reasoning_content": "I need evidence for the capital.",
                       "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": question})}}]}
            finish = "tool_calls"
        else:
            answer = "London" if type(self).mode == "wrong" else "Paris"
            message = {"role": "assistant", "reasoning_content": "The retrieved passage gives the answer.",
                       "content": f"<answer>{answer}</answer>"}
            finish = "length" if type(self).mode == "truncated" else "stop"
        self.reply({"id": f"fixture-{type(self).teacher_calls}", "model": "deepseek-flash",
                    "system_fingerprint": "local-fixture", "choices": [{"message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}})


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 100000

    def apply_chat_template(self, messages, **kwargs):
        return "[user]" + messages[0]["content"] + "[assistant]<think>\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class PipelineTests(unittest.TestCase):
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
        FixtureServer.mode = "correct"
        FixtureServer.teacher_calls = 0
        FixtureServer.requests = []

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        data = self.root / "data"
        for source, evaluation in [("nq", "test"), ("hotpotqa", "dev")]:
            raw = data / "raw" / source
            raw.mkdir(parents=True)
            examples = [{"id": f"{source}-{i}", "question": f"Capital of France {source} {i}?",
                         "golden_answers": ["Paris"], "metadata": {"type": "bridge" if i % 2 else "comparison", "level": "medium"}}
                        for i in range(8)]
            # Duplicate plus a question present in a held-out dataset.
            examples += [examples[0], {"id": "leak", "question": "Held out?", "golden_answers": ["Paris"]}]
            sft.write_jsonl(raw / "train.jsonl", examples)
            sft.write_jsonl(raw / f"{evaluation}.jsonl", [{"question": "Held out?"}])
        args = argparse.Namespace(data_dir=data, output_dir=self.root / "selection", nq_count=2,
                                  hotpotqa_count=2, reserve_ratio=1.0, seed=42)
        with redirect_stdout(io.StringIO()):
            sft.sample_command(args)
        return args

    def generation_args(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        return argparse.Namespace(input_dir=self.root / "selection", output_dir=self.root / "generation",
            api_key_file=None, api_key_env="SFT_TEST_KEY", base_url=base, model="deepseek-flash",
            retriever_url=base + "/retrieve", topk=3, passage_chars=1800, max_searches=3,
            min_searches=1, max_attempts=2, max_tokens=2048, reasoning_effort="high",
            concurrency=2, timeout=10, limit=0, generation_mode="native",
            student_tokenizer=None, student_max_length=4096, quality_review=False,
            review_model=None, review_reasoning_effort="low", review_max_tokens=4096,
            preflight_only=False)

    def run_generate(self, args):
        with patch.dict("os.environ", {"SFT_TEST_KEY": "local-test-only"}), redirect_stdout(io.StringIO()):
            sft.generate_command(args)

    def test_sampling_decontaminates_and_is_reproducible(self):
        args = self.prepare()
        first = (args.output_dir / "candidates.jsonl").read_bytes()
        args.output_dir = self.root / "selection_again"
        with redirect_stdout(io.StringIO()):
            sft.sample_command(args)
        self.assertEqual(first, (args.output_dir / "candidates.jsonl").read_bytes())
        items = [row for _, row in sft.rows(args.output_dir / "selected.jsonl")]
        self.assertEqual(Counter(row["data_source"] for row in items), {"nq": 2, "hotpotqa": 2})
        self.assertNotIn("held out", {sft.question_key(row["question"]) for row in items})
        self.assertEqual(len({sft.question_key(row["question"]) for _, row in sft.rows(args.output_dir / "candidates.jsonl")}), 8)

    def test_real_http_loop_resume_export_and_loss_mask(self):
        self.prepare()
        args = self.generation_args()
        args.limit = 1
        self.run_generate(args)
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        self.assertFalse(report["complete"])
        args.limit = 0
        self.run_generate(args)
        self.assertEqual(FixtureServer.teacher_calls, 8)
        self.run_generate(args)
        self.assertEqual(FixtureServer.teacher_calls, 8, "resume must not purchase completed teacher calls")
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["usage"]["total_tokens"], 480)
        accepted = [row for _, row in sft.rows(args.output_dir / "accepted.jsonl")]
        self.assertIn("&lt;search&gt;untrusted&lt;/search&gt;", accepted[0]["trajectory"])
        self.assertEqual(accepted[0]["search_count"], 1)
        audit_text = "".join(path.read_text() for path in (args.output_dir / "audit").glob("*.jsonl"))
        self.assertNotIn("local-test-only", audit_text)
        export = argparse.Namespace(input_dir=args.output_dir, output_dir=self.root / "export",
                                    allow_partial=False, val_ratio=0.5, seed=42)
        with redirect_stdout(io.StringIO()):
            sft.export_command(export)
        train = [row for _, row in sft.rows(export.output_dir / "train.episodes.jsonl")]
        val = [row for _, row in sft.rows(export.output_dir / "val.episodes.jsonl")]
        self.assertTrue({row["sample_id"] for row in train}.isdisjoint({row["sample_id"] for row in val}))
        dataset = SearchSFTDataset(export.output_dir / "train.steps.parquet", CharacterTokenizer(), max_length=10000)
        self.assertEqual(len(dataset), 4)
        for row in dataset:
            supervised = "".join(chr(label) for label in row["labels"] if label not in (-100, 100000))
            self.assertNotIn("<information>", supervised)
            self.assertNotIn("Question:", supervised)
            self.assertTrue(supervised.startswith("<think>"))
            self.assertIn("</search>" if "<search>" in supervised else "</answer>", supervised)
        args.topk = 4
        with self.assertRaisesRegex(ValueError, "Resume settings differ"):
            self.run_generate(args)

    def test_wrong_or_truncated_answers_do_not_become_sft(self):
        self.prepare()
        args = self.generation_args()
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        for mode, reason in [("wrong", "answer_em_mismatch"), ("truncated", "unfinished_generation:length")]:
            FixtureServer.mode = mode
            result = sft.teacher_episode(sample, args, "local-test-only", 1)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["reject_reason"], reason)
        self.assertEqual(sft.normalize_answer("The Paris!"), "paris")
        self.assertNotEqual(sft.normalize_answer("Paris, France"), sft.normalize_answer("Paris"))

    def reviewed_alias_fixture(self):
        from sft_answer_aliases import golden_answers_sha256, question_sha256, load_reviewed_aliases
        self.prepare()
        args = self.generation_args()
        candidate = next(row for _, row in sft.rows(args.input_dir / "candidates.jsonl")
                         if row["data_source"] == "nq")
        candidate["golden_answers"] = ["Paris, France"]
        sft.write_jsonl(args.input_dir / "candidates.jsonl", [candidate])
        selection = json.loads((args.input_dir / "selection_manifest.json").read_text())
        selection.update(targets={"nq": 1}, candidates_sha256=sft.fingerprints(args.input_dir / "candidates.jsonl")[0])
        sft.save_json(args.input_dir / "selection_manifest.json", selection)
        args.answer_aliases_file = self.root / "reviewed_aliases.json"
        entry = {"sample_id": candidate["sample_id"], "data_source": "nq", "source_split": "train",
                 "question_sha256": question_sha256(candidate["question"]),
                 "golden_answers_sha256": golden_answers_sha256(candidate["golden_answers"]),
                 "aliases": ["Paris"], "review": {"method": "LOCAL_FIXTURE_ONLY",
                    "reason": "ALIASES_REVIEW_MARKER", "evidence": [{"source": "local fixture doc-1",
                    "quote": "Paris is its capital."}]}}
        sft.save_json(args.answer_aliases_file, {"version": 1, "scope": "sft_train_only", "entries": [entry]})
        args.reviewed_answer_aliases = load_reviewed_aliases(args.answer_aliases_file)
        args.quality_review = True
        return args, candidate

    def test_reviewed_train_alias_keeps_original_em_and_exports_with_bound_manifest(self):
        args, _ = self.reviewed_alias_fixture()
        self.run_generate(args)
        episode = next(sft.rows(args.output_dir / "accepted.jsonl"))[1]
        self.assertFalse(episode["answer_em"], "Reviewed aliases must never inflate original-gold EM")
        self.assertEqual(episode["answer_check"]["method"], "reviewed_alias")
        self.assertTrue(episode["quality_review"]["passed"])
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual(report["answer_checks"]["reviewed_alias_accepted_episodes"], 1)
        sent = json.dumps(FixtureServer.requests)
        self.assertNotIn("Paris, France", sent)
        self.assertNotIn("ALIASES_REVIEW_MARKER", sent)
        export = argparse.Namespace(input_dir=args.output_dir, output_dir=self.root / "alias_export",
                                    allow_partial=False, val_ratio=0.0, seed=42)
        with redirect_stdout(io.StringIO()):
            sft.export_command(export)
        self.assertEqual(len(list(sft.rows(export.output_dir / "train.steps.jsonl"))), 2)
        args.answer_aliases_file.write_text(args.answer_aliases_file.read_text() + "\n")
        export.output_dir = self.root / "alias_tampered"
        with self.assertRaisesRegex(ValueError, "aliases changed"):
            sft.export_command(export)

    def test_reviewed_alias_cannot_bypass_length_or_evidence_review(self):
        args, candidate = self.reviewed_alias_fixture()
        args.student_max_length = 1
        result = sft.teacher_episode(candidate, args, "LOCAL_FIXTURE_ONLY", 1, CharacterTokenizer())
        self.assertEqual(result["answer_check"]["method"], "reviewed_alias")
        self.assertEqual(result["reject_reason"], "student_length:over_max_length")
        args.student_max_length = 10000
        negative = {"passed": False, "reason": "quality_review:unsupported_evidence", "api_calls": [], "usage": {}}
        with patch("sft_quality.review_trajectory", return_value=negative) as review:
            result = sft.teacher_episode(candidate, args, "LOCAL_FIXTURE_ONLY", 2, CharacterTokenizer())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reject_reason"], "quality_review:unsupported_evidence")
        review.assert_called_once()

    def test_query_first_uses_real_query_and_evidence_without_fake_tool_call(self):
        self.prepare()
        args = self.generation_args()
        args.generation_mode = "query-first"
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        result = sft.teacher_episode(sample, args, "local-test-only", 1, CharacterTokenizer())
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["length_check"]["passed"])
        self.assertEqual([event["role"] for event in result["events"]],
                         ["assistant", "environment", "assistant"])
        requests = [row["body"] for row in FixtureServer.requests if row["path"] != "/retrieve"]
        self.assertNotIn("tools", requests[0])
        self.assertNotIn("tool_choice", requests[0])
        self.assertTrue(all("tools" not in request and "tool_choice" not in request for request in requests))
        self.assertEqual(requests[1]["messages"][0]["content"], sft.TEXT_SEARCH_SYSTEM)
        self.assertEqual(requests[1]["messages"][2]["content"], f"<search>{sample['question']}</search>")
        self.assertEqual(requests[1]["messages"][2]["reasoning_content"], "I need evidence for the capital.")
        self.assertNotIn("tool_calls", requests[1]["messages"][2])
        self.assertEqual(requests[1]["messages"][3]["role"], "user")
        self.assertIn("UNTRUSTED RETRIEVED EVIDENCE", requests[1]["messages"][3]["content"])
        self.assertIn("&lt;search&gt;untrusted&lt;/search&gt;", requests[1]["messages"][3]["content"])
        self.assertNotIn("golden_answers", json.dumps(requests))

    def test_query_first_rejects_direct_answer_before_any_retrieval(self):
        self.prepare()
        args = self.generation_args()
        args.generation_mode = "query-first"
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        FixtureServer.mode = "direct"
        result = sft.teacher_episode(sample, args, "local-test-only", 1)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reject_reason"], "first_search_required")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["search_count"], 0)
        self.assertEqual([row["path"] for row in FixtureServer.requests], ["/chat/completions"])

    def test_query_first_continues_text_search_with_real_retrieval_and_preserved_reasoning(self):
        self.prepare()
        args = self.generation_args()
        args.generation_mode = "query-first"
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        FixtureServer.mode = "two_search"
        result = sft.teacher_episode(sample, args, "local-test-only", 1)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["search_count"], 2)
        requests = [row["body"] for row in FixtureServer.requests if row["path"] != "/retrieve"]
        third = requests[2]["messages"]
        self.assertEqual(third[-1]["role"], "user")
        self.assertIn("UNTRUSTED RETRIEVED EVIDENCE", third[-1]["content"])
        self.assertEqual(third[-2]["content"], "<search>Paris France capital</search>")
        self.assertTrue(third[-2]["reasoning_content"])
        self.assertTrue(third[2]["reasoning_content"])
        self.assertTrue(all("tools" not in request and "tool_choice" not in request for request in requests))
        self.assertTrue(all(message["role"] != "tool" for message in third))
        retrievals = [row["body"] for row in FixtureServer.requests if row["path"] == "/retrieve"]
        self.assertEqual([row["queries"] for row in retrievals], [[sample["question"]], ["Paris France capital"]])

    def test_query_first_rejects_multiple_actions_and_dsml_without_repair(self):
        self.prepare()
        args = self.generation_args()
        args.generation_mode = "query-first"
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        for mode in ("multi_search", "dsml_suffix"):
            with self.subTest(mode=mode):
                FixtureServer.mode = mode
                FixtureServer.requests = []
                result = sft.teacher_episode(sample, args, "local-test-only", 1)
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["reject_reason"], "invalid_text_action_format")
                self.assertEqual(result["search_count"], 1)
                self.assertEqual(len([row for row in FixtureServer.requests if row["path"] == "/retrieve"]), 1)

    def test_text_search_budget_is_enforced_before_another_retrieval(self):
        self.prepare()
        args = self.generation_args()
        args.generation_mode = "query-first"
        args.max_searches = 1
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        FixtureServer.mode = "two_search"
        result = sft.teacher_episode(sample, args, "local-test-only", 1)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reject_reason"], "search_limit_exceeded")
        requests = [row["body"] for row in FixtureServer.requests if row["path"] != "/retrieve"]
        self.assertIn("No searches remain", requests[-1]["messages"][-1]["content"])
        self.assertEqual(len([row for row in FixtureServer.requests if row["path"] == "/retrieve"]), 1)

    def test_student_length_rejection_happens_before_paid_review(self):
        self.prepare()
        args = self.generation_args()
        args.quality_review = True
        args.student_max_length = 1
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        with patch("sft_quality.review_trajectory") as review:
            result = sft.teacher_episode(sample, args, "local-test-only", 1, CharacterTokenizer())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reject_reason"], "student_length:over_max_length")
        self.assertFalse(result["length_check"]["passed"])
        review.assert_not_called()

    def test_quality_rejection_uses_reserve_and_usage_includes_reviews(self):
        self.prepare()
        args = self.generation_args()
        args.quality_review = True
        args.max_attempts = 1
        initial = [row for _, row in sft.rows(args.input_dir / "selected.jsonl")]
        rejected_id = initial[0]["sample_id"]

        def review(sample, *unused):
            passed = sample["sample_id"] != rejected_id
            return {"passed": passed, "reason": "ok" if passed else "quality_review:unsupported_evidence",
                    "usage": {"total_tokens": 7}, "api_call": {"usage": {"total_tokens": 7}}}

        with patch("sft_quality.review_trajectory", side_effect=review):
            self.run_generate(args)
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        accepted = [row for _, row in sft.rows(args.output_dir / "accepted.jsonl")]
        self.assertTrue(report["complete"])
        self.assertNotIn(rejected_id, {row["sample_id"] for row in accepted})
        self.assertTrue(any(row["selection"] == "reserve" for row in accepted))
        self.assertEqual(report["attempts"], 5)
        self.assertEqual(report["quality_checks"]["review_passed_attempts"], 4)
        self.assertEqual(report["quality_checks"]["review_rejected_attempts"], 1)
        self.assertEqual(report["usage"]["total_tokens"], 5 * (120 + 7))
        self.assertTrue(all(row["quality_review"]["passed"] for row in accepted))

    def test_preflight_has_no_generation_review_or_attempt_checkpoint(self):
        self.prepare()
        args = self.generation_args()
        args.preflight_only = True
        args.generation_mode = "query-first"
        args.quality_review = True
        args.student_tokenizer = "local-fixture-tokenizer"
        with patch("prepare_sft.load_student_tokenizer", return_value=CharacterTokenizer()) as load, \
                patch("sft_quality.review_trajectory") as review:
            self.run_generate(args)
        load.assert_called_once_with("local-fixture-tokenizer")
        review.assert_not_called()
        self.assertEqual(FixtureServer.teacher_calls, 0)
        self.assertFalse((args.output_dir / "attempts.jsonl").exists())
        self.assertFalse((args.output_dir / "accepted.jsonl").exists())
        self.assertFalse((args.output_dir / "generation_report.json").exists())
        report = json.loads((args.output_dir / "preflight_report.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"]["student_tokenizer"], "passed")
        self.assertEqual(report["generation_calls"], 0)
        self.assertEqual(report["review_calls"], 0)
        self.assertEqual([row["path"] for row in FixtureServer.requests], ["/retrieve"])
        args.generation_mode = "native"
        with patch("prepare_sft.load_student_tokenizer", return_value=CharacterTokenizer()):
            with self.assertRaisesRegex(ValueError, "Resume settings differ"):
                self.run_generate(args)

    def test_v4_real_http_generation_review_length_export_and_resume(self):
        self.prepare()
        args = self.generation_args()
        args.generation_mode = "query-first"
        args.quality_review = True
        args.student_tokenizer = "local-fixture-tokenizer"
        with patch("prepare_sft.load_student_tokenizer", return_value=CharacterTokenizer()):
            self.run_generate(args)
            self.run_generate(args)
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        accepted = [row for _, row in sft.rows(args.output_dir / "accepted.jsonl")]
        self.assertEqual(FixtureServer.teacher_calls, 12)
        self.assertTrue(report["complete"])
        self.assertEqual(report["usage"]["total_tokens"], 720)
        self.assertEqual(report["quality_checks"]["review_passed_attempts"], 4)
        self.assertEqual(report["quality_checks"]["length_passed_attempts"], 4)
        self.assertGreater(report["quality_checks"]["accepted_max_step_tokens"], 0)
        self.assertTrue(all(row["quality_review"]["passed"] and row["length_check"]["passed"] for row in accepted))
        audits = [json.loads(line) for path in (args.output_dir / "audit").glob("*.jsonl")
                  for line in path.read_text().splitlines()]
        review_audits = [row for row in audits if row["kind"] == "quality_review"]
        self.assertEqual(len(review_audits), 4)
        self.assertNotIn("golden_answers", json.dumps([row["request"] for row in review_audits]))
        export = argparse.Namespace(input_dir=args.output_dir, output_dir=self.root / "export",
                                    allow_partial=False, val_ratio=0, seed=42)
        with redirect_stdout(io.StringIO()):
            sft.export_command(export)
        dataset = SearchSFTDataset(export.output_dir / "train.steps.parquet", CharacterTokenizer(), max_length=4096)
        self.assertEqual(len(dataset), 8)
        manifest = json.loads((export.output_dir / "sft_manifest.json").read_text())
        self.assertTrue(manifest["quality_review"])
        self.assertEqual(manifest["generation_version"], 4)

    def test_review_model_is_checked_without_generation(self):
        self.prepare()
        args = self.generation_args()
        args.preflight_only = True
        args.quality_review = True
        args.review_model = "unavailable-review-model"
        with self.assertRaisesRegex(ValueError, "unavailable-review-model"):
            self.run_generate(args)
        self.assertEqual(FixtureServer.teacher_calls, 0)
        report = json.loads((args.output_dir / "preflight_report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["phase"], "models")

    def test_fatal_review_checkpoints_paid_teacher_turns_and_stops_next_batch(self):
        self.prepare()
        args = self.generation_args()
        args.quality_review = True
        with patch("sft_quality.review_trajectory", side_effect=sft.FatalAPIError(
                "HTTP 402 from local-fixture; check configuration/balance")):
            with self.assertRaisesRegex(sft.FatalAPIError, "HTTP 402"):
                self.run_generate(args)
        self.assertEqual(FixtureServer.teacher_calls, 4, "only the first two-question batch may run")
        attempts = [row for _, row in sft.rows(args.output_dir / "attempts.jsonl")]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(row["reject_reason"] == "fatal_api_error" for row in attempts))
        self.assertTrue(all(len(row["api_calls"]) == 2 for row in attempts))
        self.assertTrue(all(row["attempt"] == 1 and row["usage"]["total_tokens"] == 120 for row in attempts))
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        self.assertFalse(report["complete"])
        self.assertEqual(report["usage"]["total_tokens"], 240)
        self.assertIn("HTTP 402", report["fatal_error"])
        self.assertEqual((args.output_dir / "accepted.jsonl").read_text(), "")
        previously_attempted = {row["sample_id"] for row in attempts}
        with patch("sft_quality.review_trajectory", return_value={"passed": True, "reason": "ok", "usage": {}}):
            self.run_generate(args)
        attempts = [row for _, row in sft.rows(args.output_dir / "attempts.jsonl")]
        for sample_id in previously_attempted:
            self.assertEqual([row["attempt"] for row in attempts if row["sample_id"] == sample_id], [1, 2])

    def test_all_review_calls_are_recorded_and_cumulative_usage_counted_once(self):
        self.prepare()
        args = self.generation_args()
        args.quality_review = True
        review_calls = [{"id": "review-1", "review_attempt": 1, "usage": {"total_tokens": 7}},
                        {"id": "review-2", "review_attempt": 2, "usage": {"total_tokens": 11}}]
        review = {"passed": True, "reason": "passed", "api_calls": review_calls,
                  "api_call": review_calls[-1], "usage": {"total_tokens": 18},
                  "review_attempts": [{"attempt": 1, "reason": "quality_review:invalid_json"},
                                      {"attempt": 2, "reason": "passed"}]}
        with patch("sft_quality.review_trajectory", return_value=review):
            self.run_generate(args)
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        accepted = [row for _, row in sft.rows(args.output_dir / "accepted.jsonl")]
        self.assertTrue(report["complete"])
        self.assertEqual(report["usage"]["total_tokens"], 4 * (120 + 18))
        for episode in accepted:
            self.assertEqual([call["id"] for call in episode["api_calls"] if call["kind"] == "quality_review"],
                             ["review-1", "review-2"])
            self.assertEqual(episode["usage"]["total_tokens"], 138)
            self.assertEqual(len(episode["quality_review"]["review_attempts"]), 2)

    def test_partial_review_calls_and_usage_survive_later_fatal_error(self):
        self.prepare()
        args = self.generation_args()
        args.quality_review = True

        def failed_review(*unused):
            error = sft.FatalAPIError("HTTP 402 from local-fixture; check configuration/balance")
            first_call = {"id": "paid-review-1", "review_attempt": 1, "usage": {"total_tokens": 7}}
            error.partial_review_result = {"passed": False, "reason": "quality_review:http_error",
                "api_calls": [first_call], "api_call": first_call, "usage": {"total_tokens": 7},
                "review_attempts": [{"attempt": 1, "reason": "quality_review:invalid_json"},
                                    {"attempt": 2, "reason": "quality_review:http_error"}]}
            raise error

        with patch("sft_quality.review_trajectory", side_effect=failed_review):
            with self.assertRaisesRegex(sft.FatalAPIError, "HTTP 402"):
                self.run_generate(args)
        report = json.loads((args.output_dir / "generation_report.json").read_text())
        attempts = [row for _, row in sft.rows(args.output_dir / "attempts.jsonl")]
        self.assertEqual(FixtureServer.teacher_calls, 4, "fatal errors must stop the next batch")
        self.assertEqual(report["usage"]["total_tokens"], 2 * (120 + 7))
        self.assertFalse(report["complete"])
        for episode in attempts:
            self.assertEqual(episode["reject_reason"], "fatal_api_error")
            self.assertEqual(episode["usage"]["total_tokens"], 127)
            self.assertEqual(len(episode["api_calls"]), 3)
            self.assertEqual(episode["api_calls"][-1]["id"], "paid-review-1")
            self.assertEqual(episode["api_calls"][-1]["kind"], "quality_review")
            self.assertEqual(len(episode["quality_review"]["review_attempts"]), 2)

    def test_partial_review_usage_survives_transport_failure(self):
        self.prepare()
        args = self.generation_args()
        args.quality_review = True
        sample = next(sft.rows(args.input_dir / "selected.jsonl"))[1]
        error = RuntimeError("Transport/JSON failure from local-fixture")
        error.partial_review_result = {"passed": False, "reason": "quality_review:http_error",
            "api_calls": [{"id": "paid-review-1", "usage": {"total_tokens": 7}}],
            "usage": {"total_tokens": 7}, "review_attempts": [{"attempt": 1}, {"attempt": 2}]}
        with patch("sft_quality.review_trajectory", side_effect=error):
            episode = sft.teacher_episode(sample, args, "local-test-only", 1)
        self.assertEqual(episode["status"], "rejected")
        self.assertEqual(episode["usage"]["total_tokens"], 127)
        self.assertEqual(episode["api_calls"][-1]["id"], "paid-review-1")
        self.assertEqual(episode["reject_reason"], str(error))

    def test_output_lock_rejects_second_process_before_any_api_call(self):
        import fcntl
        self.prepare()
        args = self.generation_args()
        args.output_dir.mkdir()
        lock_path = args.output_dir / ".generation.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch("prepare_sft.http_json") as request:
                with self.assertRaisesRegex(ValueError, "Another generate/preflight process"):
                    self.run_generate(args)
            request.assert_not_called()
        self.assertTrue(lock_path.exists(), "keep the stable lock inode in place")
        args.preflight_only = True
        self.run_generate(args)
        self.assertTrue(json.loads((args.output_dir / "preflight_report.json").read_text())["passed"])

    def test_export_rejects_changed_student_prompt_after_generation(self):
        self.prepare()
        args = self.generation_args()
        self.run_generate(args)
        settings = json.loads((args.output_dir / "run_manifest.json").read_text())["settings"]
        self.assertEqual(settings["student_prompt_sha256"], sft.digest_text(sft.PROMPT))
        export = argparse.Namespace(input_dir=args.output_dir, output_dir=self.root / "export",
                                    allow_partial=False, val_ratio=0, seed=42)
        with patch("prepare_sft.PROMPT", sft.PROMPT + " Extra student instructions."):
            with self.assertRaisesRegex(ValueError, "Student prompt has changed"):
                sft.export_command(export)
        self.assertFalse(export.output_dir.exists())
        with redirect_stdout(io.StringIO()):
            sft.export_command(export)
        manifest = json.loads((export.output_dir / "sft_manifest.json").read_text())
        self.assertEqual(manifest["student_prompt_sha256"], settings["student_prompt_sha256"])

    def test_independent_qa_validation_preserves_all_sft_training_episodes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.prepare()
        args = self.generation_args()
        val_path = args.input_dir / "val.parquet"
        pq.write_table(pa.Table.from_pylist([{"question": "A separate fixture validation question"}]), val_path)
        manifest_path = args.input_dir / "selection_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["external_validation"] = {"path": str(val_path), "sha256": sft.fingerprints(val_path)[0]}
        sft.save_json(manifest_path, manifest)
        self.run_generate(args)
        export = argparse.Namespace(input_dir=args.output_dir, output_dir=self.root / "export",
                                    allow_partial=False, val_ratio=None, seed=42)
        with redirect_stdout(io.StringIO()):
            sft.export_command(export)
        exported = json.loads((export.output_dir / "sft_manifest.json").read_text())
        self.assertEqual(exported["val_ratio"], 0)
        self.assertEqual(exported["counts"]["train"]["episodes"], 4)
        self.assertEqual(exported["counts"]["val"]["episodes"], 0)
        val_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "validation file has changed"):
            sft.export_command(export)

    def test_torn_checkpoint_is_repaired_without_losing_good_rows(self):
        path = self.root / "attempts.jsonl"
        path.write_bytes(b'{"status":"accepted"}\n{"status":')
        self.assertEqual(sft.load_attempts(path), [{"status": "accepted"}])
        self.assertEqual(path.read_bytes(), b'{"status":"accepted"}\n')
        path.write_bytes(b'broken\n{"status":"accepted"}\n')
        with self.assertRaisesRegex(ValueError, "Corrupt checkpoint"):
            sft.load_attempts(path)


if __name__ == "__main__":
    unittest.main()
