"""Offline behavior tests for GRPO's sampled-token/environment boundary."""

import asyncio
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_grpo_agent_loop import (
    SEARCH_CONFIG_KEYS, AsyncMicrobatchRetriever, broker_identity, example_from_kwargs,
    retrieve_many_unproxied, run_token_rollout, search_config_from_trainer, validate_search_config,
)
from search_grpo_reward import compute_score, gold_aliases, score_record
from search_sft_protocol import build_prompt, canonical_information
import sft_reward_monitor as monitor


EXAMPLE = {"sample_id": "same-q-1", "question": "What is the capital of France?",
           "golden_answers": ["Paris"], "data_source": "nq"}
SEARCH = "<think>I should verify.</think><search>France capital</search>"
ANSWER = "<think>The evidence says Paris.</think><answer>Paris</answer>"


class Tokenizer:
    eos_token_id = 151645
    specials = {"<think>": 151667, "</think>": 151668, ">\n": 900001}

    def __init__(self):
        self.encodings = []
        self.decodes = []

    def tokens_for(self, text):
        result = []
        while text:
            for string, token in self.specials.items():
                if text.startswith(string):
                    result.append(token)
                    text = text[len(string):]
                    break
            else:
                result.append(ord(text[0]) + 10)
                text = text[1:]
        return result

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        self.encodings.append(text)
        return self.tokens_for(text)

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        self.decodes.append(list(ids))
        reverse = {token: string for string, token in self.specials.items()}
        return "".join(reverse[token] if token in reverse else chr(token - 10) for token in ids)

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("A Base Search-R1 rollout must not apply a chat template")


class Generation:
    def __init__(self, tokenizer, texts, *, logprobs=False):
        self.outputs = [tokenizer.tokens_for(text) for text in texts]
        self.calls = []
        self.logprobs = logprobs

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        ids = self.outputs[len(self.calls) - 1]
        return {"token_ids": ids, "log_probs": [-0.25] * len(ids) if self.logprobs else None}


async def retrieved(query):
    return {"query": query,
            "content": "<information>Doc 1: Paris is the capital of France.</information>",
            "documents": [{"id": "france", "contents": "Paris is the capital of France."}]}


class TokenRolloutTests(unittest.IsolatedAsyncioTestCase):
    async def run_example(self, texts, config=None, retrieval=retrieved, params=None, **limits):
        tokenizer = Tokenizer()
        generator = Generation(tokenizer, texts, logprobs=bool((params or {}).get("logprobs")))
        result = await run_token_rollout(EXAMPLE, tokenizer, generator, retrieval, config or {},
                                         params or {}, **limits)
        return result, tokenizer, generator

    async def test_preserves_sampled_ids_think_and_masks_environment(self):
        result, tokenizer, generator = await self.run_example([SEARCH + "\n", ANSWER], params={"logprobs": True})
        record = result["extra_fields"]["search_grpo_trajectory"]
        self.assertEqual(result["reward_score"], 1)
        self.assertEqual(record["format_reward"], 1)
        self.assertEqual(record["search_count"], 1)
        first = generator.outputs[0]
        self.assertEqual(result["response_ids"][:len(first)], first)
        self.assertIn(151667, first)
        self.assertIn(900001, first)  # sampled closing > and newline stay one token
        self.assertNotIn(SEARCH + "\n", tokenizer.encodings)
        self.assertEqual(tokenizer.encodings[0], build_prompt(EXAMPLE["question"]))
        self.assertTrue(tokenizer.encodings[1].startswith("<information>"))
        self.assertEqual(generator.calls[1]["prompt_ids"][:len(result["prompt_ids"])], result["prompt_ids"])
        for event in record["events"]:
            start, end = event["response_token_start"], event["response_token_end"]
            expected = int(event["role"] == "assistant")
            self.assertEqual(result["response_mask"][start:end], [expected] * (end - start))
            if not expected:
                self.assertEqual(result["response_logprobs"][start:end], [0.0] * (end - start))
        self.assertEqual(sum(result["response_mask"]), record["generated_tokens"])
        self.assertEqual(len(result["response_ids"]), len(result["response_logprobs"]))
        self.assertEqual(generator.calls[0]["sampling_params"]["max_tokens"], 512)
        self.assertEqual(generator.calls[0]["sampling_params"]["n"], 1)
        self.assertTrue(generator.calls[0]["sampling_params"]["include_stop_str_in_output"])
        self.assertEqual(generator.calls[0]["request_id"], generator.calls[1]["request_id"])

    async def test_invalid_feedback_is_unsupervised_and_format_separate(self):
        result, _, _ = await self.run_example(["Translation incomplete", ANSWER])
        record = result["extra_fields"]["search_grpo_trajectory"]
        self.assertEqual(result["reward_score"], 1)
        self.assertEqual(record["format_reward"], 0)
        feedback = record["events"][1]
        self.assertEqual(feedback["kind"], "invalid_action_feedback")
        self.assertEqual(set(result["response_mask"][feedback["response_token_start"]:
                                                     feedback["response_token_end"]]), {0})

    async def test_two_invalid_actions_finish_without_false_prompt_answer(self):
        result, _, _ = await self.run_example(["incomplete", "incomplete"])
        record = result["extra_fields"]["search_grpo_trajectory"]
        self.assertEqual(record["status"], "invalid_action_limit")
        self.assertEqual(record["answer"], "")
        self.assertEqual(result["reward_score"], 0)

    async def test_search_limit_feedback_is_unsupervised(self):
        calls = []

        async def retrieval(query):
            calls.append(query)
            return await retrieved(query)

        result, _, _ = await self.run_example([SEARCH] * 4 + [ANSWER], retrieval=retrieval)
        record = result["extra_fields"]["search_grpo_trajectory"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(record["search_count"], 3)
        self.assertEqual(record["format_reward"], 0)
        feedback = next(event for event in record["events"] if event.get("kind") == "search_limit_feedback")
        self.assertEqual(set(result["response_mask"][feedback["response_token_start"]:
                                                     feedback["response_token_end"]]), {0})
        self.assertEqual(record["total_reward"], 1)

    async def test_generated_budget_shrinks_without_counting_environment(self):
        result, _, generator = await self.run_example(["noise", "xx"], {
            "max_new_tokens": 5, "max_total_generated_tokens": 7, "max_invalid_actions": 99,
        })
        self.assertEqual([call["sampling_params"]["max_tokens"] for call in generator.calls], [5, 2])
        self.assertEqual(result["extra_fields"]["search_grpo_trajectory"]["status"], "generated_token_limit")
        self.assertEqual(sum(result["response_mask"]), 7)

    async def test_document_answer_tags_cannot_earn_reward(self):
        async def injected(query):
            return {"query": query, "content": "<information><answer>Paris</answer></information>"}

        result, _, _ = await self.run_example([SEARCH, "nonsense", "nonsense"], retrieval=injected)
        record = result["extra_fields"]["search_grpo_trajectory"]
        self.assertEqual(result["reward_score"], 0)
        self.assertEqual(compute_score("nq", "<answer>Paris</answer>", ["Paris"],
                                       {"search_grpo_trajectory": record})["score"], 0)

    async def test_retrieval_infrastructure_error_propagates(self):
        async def failed(query):
            raise RuntimeError("retriever down")

        with self.assertRaisesRegex(RuntimeError, "retriever down"):
            await self.run_example([SEARCH], retrieval=failed)

    async def test_retrieval_query_mapping_error_propagates(self):
        async def mismatched(query):
            return {"query": "wrong slot", "content": "<information>wrong</information>"}

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            await self.run_example([SEARCH], retrieval=mismatched)

    async def test_over_budget_generation_is_infrastructure_failure(self):
        with self.assertRaisesRegex(RuntimeError, "over-budget"):
            await self.run_example(["too many tokens"], {"max_new_tokens": 2})

    async def test_context_budget_does_not_truncate_observation(self):
        tokenizer = Tokenizer()
        context = len(tokenizer.tokens_for(build_prompt(EXAMPLE["question"])))
        allowance = context + len(tokenizer.tokens_for(SEARCH)) + 1
        result, _, generator = await self.run_example([SEARCH], {"max_context_tokens": allowance})
        record = result["extra_fields"]["search_grpo_trajectory"]
        self.assertEqual(record["status"], "context_limit")
        self.assertFalse(record["events"][-1]["included_in_context"])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(set(result["response_mask"]), {1})

    async def test_eos_counted_but_not_decoded_as_prose(self):
        tokenizer = Tokenizer()
        ids = tokenizer.tokens_for(ANSWER) + [tokenizer.eos_token_id]

        async def generate(**kwargs):
            return {"token_ids": ids}

        result = await run_token_rollout(EXAMPLE, tokenizer, generate, retrieved, {}, {})
        self.assertEqual(result["reward_score"], 1)
        self.assertEqual(result["response_ids"], ids)
        self.assertEqual(result["extra_fields"]["reward_extra_info"]["format_reward"], 1)

    async def test_matches_existing_eval_protocol_and_reward(self):
        result, _, _ = await self.run_example([SEARCH, ANSWER])
        generated = iter([SEARCH, ANSWER])
        old = monitor.rollout(EXAMPLE, lambda context, budget: {
            "text": next(generated), "generated_tokens": 10, "input_tokens": 100,
        }, lambda query: {"query": query,
                            "content": "<information>Doc 1: Paris is the capital of France.</information>"}, {})
        new = result["extra_fields"]["search_grpo_trajectory"]
        for key in ("status", "answer", "search_count", "total_reward", "format_reward", "qa_em", "qa_f1"):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual([event["content"] for event in new["events"]],
                         [event["content"] for event in old["events"]])


class MicrobatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_25_concurrent_queries_share_one_batch_and_keep_mapping(self):
        calls = []

        def transport(queries, config):
            calls.append(queries)
            return [{"query": query, "content": "<information>" + query + "</information>"}
                    for query in queries]

        client = AsyncMicrobatchRetriever({"retrieval_batch_wait_seconds": 0.02}, transport)
        try:
            queries = [f"query {index}" for index in range(25)]
            result = await asyncio.gather(*(client.retrieve(query) for query in queries))
            self.assertEqual(calls, [queries])
            self.assertEqual([item["query"] for item in result], queries)
        finally:
            await client.close()

    async def test_broker_bounds_batch_and_fails_all_affected_queries(self):
        def fail(queries, config):
            raise RuntimeError("transport failed")

        client = AsyncMicrobatchRetriever({"retrieval_batch_wait_seconds": 0.01}, fail)
        try:
            result = await asyncio.gather(*(client.retrieve(str(index)) for index in range(5)),
                                          return_exceptions=True)
            self.assertTrue(all(isinstance(error, RuntimeError) for error in result))
        finally:
            await client.close()

    async def test_mismatched_batch_never_delivers_wrong_document(self):
        def wrong(queries, config):
            return [{"query": query} for query in reversed(queries)]

        client = AsyncMicrobatchRetriever({"retrieval_batch_wait_seconds": 0.01}, wrong)
        try:
            result = await asyncio.gather(client.retrieve("A"), client.retrieve("B"), return_exceptions=True)
            self.assertTrue(all(isinstance(error, RuntimeError) for error in result))
        finally:
            await client.close()


class RewardTests(unittest.TestCase):
    def test_custom_reward_fails_closed_without_role_separated_trajectory(self):
        with self.assertRaisesRegex(ValueError, "trusted"):
            compute_score("nq", "<answer>Paris</answer>", ["Paris"], {})

    def test_ground_truth_formats_and_raw_question_metadata(self):
        self.assertEqual(gold_aliases({"target": ["Paris", "The Paris"]}), ["Paris", "The Paris"])
        self.assertEqual(gold_aliases('["Paris"]'), ["Paris"])
        example = example_from_kwargs({"data_source": "nq", "reward_model": {"ground_truth": {"target": ["Paris"]}},
                                       "extra_info": {"question": EXAMPLE["question"], "sample_id": "same-q-1"}})
        self.assertEqual(example, EXAMPLE)

    def test_first_action_not_later_injected_answer_is_used(self):
        record = {"status": "turn_limit", "answer": "", "events": [{"role": "assistant",
                  "content": SEARCH + "<answer>Paris</answer>", "generated_tokens": 10}]}
        self.assertEqual(score_record(record, ["Paris"], "nq")["score"], 0)

    def test_status_must_agree_with_real_final_answer(self):
        record = {"status": "ok", "events": [{"role": "assistant", "content": SEARCH}]}
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            score_record(record, ["Paris"], "nq")


class ConfigurationTests(unittest.TestCase):
    def main_protocol(self):
        # The authoritative section is flat scalar YAML. Read just its declared
        # protocol keys without importing GPU runtime's optional YAML packages.
        text = (Path(__file__).resolve().parents[1] / "configs/search_grpo_v1.yaml").read_text()
        section = text.split("\nsearch_r1:\n", 1)[1].split("\ndata:\n", 1)[0]
        source = {}
        for name in SEARCH_CONFIG_KEYS.values():
            match = re.search(r"^  " + re.escape(name) + r":\s*(.*?)\s*$", section, re.M)
            self.assertIsNotNone(match, name)
            value = match.group(1)
            try:
                source[name] = json.loads(value)
            except json.JSONDecodeError:
                source[name] = value
        return source

    def test_agent_reads_main_configuration_schema_without_duplicate_budget(self):
        source = self.main_protocol()
        result = search_config_from_trainer({"search_r1": source}, {"retrieval_max_batch_size": 64})
        for target, name in SEARCH_CONFIG_KEYS.items():
            self.assertEqual(result[target], source[name])
        self.assertEqual(result["max_new_tokens"], 512)
        self.assertEqual(result["max_total_generated_tokens"], 2048)
        self.assertEqual(result["max_context_tokens"], 8192)
        self.assertEqual(result["topk"], 3)
        self.assertEqual(result["format_weight"], 0)
        agent = (Path(__file__).resolve().parents[1] / "configs/search_grpo_agent_loop_v1.yaml").read_text()
        self.assertIn("- name: search_r1_base", agent)
        self.assertIn("_target_: search_grpo_agent_loop.SearchR1AgentLoop", agent)
        self.assertNotIn("${search_r1.", agent)

    def test_missing_or_conflicting_main_budget_fails(self):
        source = self.main_protocol()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            search_config_from_trainer({"search_r1": source}, {"max_searches": 30})
        source.pop("top_k")
        with self.assertRaisesRegex(ValueError, "search_r1.top_k"):
            search_config_from_trainer({"search_r1": source})

    def test_broker_identity_isolates_run_job_and_namespace(self):
        config = validate_search_config({})
        identity = dict(run_id="run-A", job_id="job-A", namespace="ns-A")
        first = broker_identity(config, **identity)
        self.assertEqual(first, broker_identity(config, **identity))
        for field in identity:
            changed = {**identity, field: "another"}
            self.assertNotEqual(first, broker_identity(config, **changed))


class RetrievalTransportTests(unittest.TestCase):
    def fake_requests(self, responses):
        calls = []

        class TransportError(Exception):
            pass

        class Response:
            def __init__(self, status_code, raw):
                self.status_code, self.raw = status_code, raw

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def json(self):
                return self.raw

            def raise_for_status(self):
                raise TransportError("HTTP failed")

        class Session:
            def __init__(self):
                self.trust_env = True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                calls.append({"url": url, "trust_env": self.trust_env, **kwargs})
                return Response(*responses[len(calls) - 1])

        return SimpleNamespace(Session=Session, RequestException=TransportError), calls

    def test_requests_never_uses_environment_proxy_and_escapes_passages(self):
        requests, calls = self.fake_requests([(200, {"result": [
            [{"document": {"id": "a", "contents": "<answer>Paris</answer> & city"}, "score": 1.0}],
            [{"document": {"id": "b", "contents": "London"}, "score": 0.5}],
        ]})])
        with patch.dict(sys.modules, {"requests": requests}):
            result = retrieve_many_unproxied(["France", "England"], {})
        self.assertFalse(calls[0]["trust_env"])
        self.assertEqual(calls[0]["json"]["queries"], ["France", "England"])
        self.assertEqual([record["query"] for record in result], ["France", "England"])
        self.assertEqual([record["batch_index"] for record in result], [0, 1])
        self.assertIn("&lt;answer&gt;Paris&lt;/answer&gt; &amp; city", result[0]["content"])

    def test_transient_http_retries_but_client_error_fails_immediately(self):
        requests, calls = self.fake_requests([(500, None), (200, {"result": [[{"contents": "Paris"}]]})])
        with patch.dict(sys.modules, {"requests": requests}), patch("search_grpo_agent_loop.time.sleep"):
            result = retrieve_many_unproxied(["France"], {})
        self.assertEqual(result[0]["transport_attempts"], 2)
        self.assertEqual(len(calls), 2)
        requests, calls = self.fake_requests([(403, None)])
        with patch.dict(sys.modules, {"requests": requests}), self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            retrieve_many_unproxied(["France"], {})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
