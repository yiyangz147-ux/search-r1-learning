"""Token-in/token-out raw-Base Search-R1 agent for pinned verl v0.6.1.

Only sampled assistant tokens receive response_mask=1.  Retrieval, invalid-action
feedback and search-budget feedback receive zero.  No generated token is ever
decoded and re-encoded.  The optional pure-Python core supports offline tests;
production requires verl and Ray plus the documented per-action max_tokens fix
in v0.6.1's vLLM server adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import time
from uuid import uuid4

from search_grpo_reward import gold_aliases, score_record
from search_sft_protocol import build_prompt, canonical_information
from sft_reward_monitor import (
    INVALID_FEEDBACK, SEARCH_LIMIT_FEEDBACK, parse_action, valid_step_format, validate_config,
)

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("verl"):
        raise
    # Offline protocol tests do not require torch/CUDA/Ray/verl.
    AgentLoopBase = object
    AgentLoopOutput = None

    def register(name):
        return lambda cls: cls


DEFAULT_SEARCH_CONFIG = {
    "retrieval_batch_wait_seconds": 0.5,
    "retrieval_max_batch_size": 64,
}

# Main configuration is the authority. The agent YAML only identifies the
# class and optional transport batching knobs, avoiding duplicate budget values.
SEARCH_CONFIG_KEYS = {
    "max_new_tokens": "max_new_tokens_per_action",
    "max_total_generated_tokens": "max_generated_tokens_per_question",
    "max_context_tokens": "max_context_tokens",
    "max_searches": "max_searches",
    "max_turns": "max_turns",
    "max_invalid_actions": "max_invalid_actions",
    "topk": "top_k",
    "passage_chars": "max_chars_per_passage",
    "retriever_url": "retrieval_url",
    "retrieval_timeout_seconds": "retrieval_timeout_seconds",
    "retrieval_retries": "retrieval_retries",
    "format_weight": "format_reward_weight",
}


def validate_search_config(config):
    result = validate_config({**DEFAULT_SEARCH_CONFIG, **dict(config or {})})
    if (not isinstance(result["retrieval_batch_wait_seconds"], (int, float))
            or result["retrieval_batch_wait_seconds"] < 0):
        raise ValueError("retrieval_batch_wait_seconds must be nonnegative")
    maximum = result["retrieval_max_batch_size"]
    if type(maximum) is not int or not 1 <= maximum <= 64:
        raise ValueError("retrieval_max_batch_size must be 1..64")
    return result


def search_config_from_trainer(config, overrides=None):
    """Map the readable ``search_r1`` schema into the existing eval protocol."""
    source = config.get("search_r1")
    if source is None:
        raise ValueError("Trainer config lacks authoritative search_r1 protocol settings")
    result = {}
    for target, name in SEARCH_CONFIG_KEYS.items():
        if name not in source:
            raise ValueError(f"Trainer config is missing search_r1.{name}")
        result[target] = source[name]
    for key, value in dict(overrides or {}).items():
        if key in result and value != result[key]:
            raise ValueError(f"Agent override {key} disagrees with main search_r1 config")
        result[key] = value
    return validate_search_config(result)


def retrieve_many_unproxied(queries, config):
    """One native local-service batch with explicit requests proxy bypass.

    Rendering, HTML escaping, passage truncation and retry policy match the
    existing evaluator's ``retrieve_many``. Sessions are per serial broker call;
    trust_env=False prevents inherited HTTP(S)_PROXY/netrc from changing routing.
    """
    import requests

    config = validate_search_config(config)
    if not queries:
        return []
    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Retriever queries must be nonempty strings")
    payload = {"queries": list(queries), "topk": config["topk"], "return_scores": True}
    raw = None
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(config["retrieval_retries"] + 1):
            try:
                with session.post(config["retriever_url"], json=payload,
                                  timeout=config["retrieval_timeout_seconds"]) as response:
                    if response.status_code >= 400:
                        if response.status_code < 500 and response.status_code != 429:
                            raise RuntimeError(f"GRPO retriever HTTP {response.status_code}")
                        response.raise_for_status()
                    raw = response.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == config["retrieval_retries"]:
                    raise RuntimeError("GRPO retriever transport/HTTP/JSON failure, retries exhausted") from None
            time.sleep(min(2 ** attempt, 8))
    batches = raw.get("result") if isinstance(raw, dict) else None
    if not isinstance(batches, list) or len(batches) != len(queries):
        raise RuntimeError("GRPO retriever returned an invalid result batch")
    result = []
    for batch_index, (query, hits) in enumerate(zip(queries, batches)):
        if not isinstance(hits, list) or not hits:
            raise RuntimeError("GRPO retriever returned an invalid/empty result batch")
        documents, lines = [], []
        for index, hit in enumerate(hits[:config["topk"]], 1):
            if not isinstance(hit, dict):
                raise RuntimeError("GRPO retriever returned a non-object passage")
            document = hit.get("document", hit)
            content = document.get("contents") if isinstance(document, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("GRPO retrieved passage lacks contents")
            shown = content[:config["passage_chars"]]
            documents.append({"id": str(document.get("id", "")), "contents": shown,
                              "score": hit.get("score")})
            lines.append(f"Doc {index}: {shown}")
        content = "<information>" + html.escape("\n".join(lines), quote=False) + "</information>"
        result.append({"query": query, "content": content, "documents": documents,
                       "request": payload, "batch_index": batch_index, "transport_attempts": attempt + 1})
    return result


class AsyncMicrobatchRetriever:
    """One serial HTTP stream coalesces concurrent searches, preserving order.

    In production this lives in one named async Ray actor shared across all
    agent-loop workers.  A process-local queue alone would fragment the FAISS
    batches and can miss its >=20-query BLAS fast path.
    """

    def __init__(self, config, transport=retrieve_many_unproxied):
        self.config = validate_search_config(config)
        self.transport = transport
        self.queue = asyncio.Queue()
        self.task = None

    async def retrieve(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must be nonempty")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((query, future))
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._drain())
        return await future

    async def _drain(self):
        while True:
            first = await self.queue.get()
            pending = [first]
            deadline = asyncio.get_running_loop().time() + self.config["retrieval_batch_wait_seconds"]
            while len(pending) < self.config["retrieval_max_batch_size"]:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            try:
                results = await asyncio.to_thread(
                    self.transport, [query for query, _ in pending], self.config)
                if not isinstance(results, list) or len(results) != len(pending):
                    raise RuntimeError("Retriever batch cardinality mismatch")
                for (query, _), result in zip(pending, results):
                    if not isinstance(result, dict) or result.get("query") != query:
                        raise RuntimeError("Retriever query/result mapping mismatch")
                for (_, future), result in zip(pending, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as error:
                # Infrastructure failures propagate to every affected rollout.
                for _, future in pending:
                    if not future.done():
                        future.set_exception(error)

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            if not future.done():
                future.cancel()


class RetrievalBroker:
    """Ray actor implementation; declared lazily to keep CPU tests lightweight."""

    def __init__(self, config):
        self.retriever = AsyncMicrobatchRetriever(config)

    async def retrieve(self, query):
        return await self.retriever.retrieve(query)


def broker_identity(config, *, run_id, job_id, namespace):
    """Never let independent runs/jobs share a named query queue."""
    relevant = {key: config[key] for key in (
        "retriever_url", "topk", "passage_chars", "retrieval_timeout_seconds",
        "retrieval_retries", "retrieval_batch_wait_seconds", "retrieval_max_batch_size",
    )}
    identity = {"transport": relevant, "run_id": str(run_id), "job_id": str(job_id),
                "namespace": str(namespace)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return "search-r1-retrieval-" + digest


def shared_retrieval_broker(config, *, run_id):
    import ray

    context = ray.get_runtime_context()
    namespace = context.namespace
    job_id = context.get_job_id()
    name = broker_identity(config, run_id=run_id, job_id=job_id, namespace=namespace)
    # Include both Ray job ID and namespace even if a pre-existing Ray cluster
    # supplies one common namespace. No detached actor survives this job.
    actor_class = ray.remote(num_cpus=0.1, max_concurrency=4096)(RetrievalBroker)
    return actor_class.options(
        name=name, namespace=namespace, get_if_exists=True,
    ).remote(config)


def _tokens(value, label):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(type(token) is not int or token < 0 for token in value):
        raise RuntimeError(f"{label} must be a list of nonnegative token IDs")
    return list(value)


def _encode(tokenizer, text):
    return _tokens(tokenizer.encode(text, add_special_tokens=False), "Tokenizer output")


def _packet_field(packet, key, default=None):
    return packet.get(key, default) if isinstance(packet, dict) else getattr(packet, key, default)


def _decode_action(tokenizer, ids):
    # Match evaluation: EOS is counted and retained for training, but excluded
    # from parser text. Qwen3's <think> special tokens must remain visible.
    eos = getattr(tokenizer, "eos_token_id", None)
    text_ids = ids[:-1] if ids and ids[-1] == eos else ids
    return tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _observation_text(content, preceding_text):
    text = canonical_information(content)
    # A single Qwen token can contain both the closing '>' and newline. Keep
    # that sampled token and let it own the separator instead of duplicating it.
    if preceding_text.endswith("\n"):
        text = text[1:]
    return text


async def run_token_rollout(example, tokenizer, generate, retrieve, config, sampling_params,
                            *, prompt_limit=None, response_limit=None, request_id=None):
    """One trajectory; n=5 grouping belongs to verl, outside this function."""
    config = validate_search_config(config)
    prompt = build_prompt(example["question"])
    prompt_ids = _encode(tokenizer, prompt)
    if prompt_limit is not None and len(prompt_ids) > prompt_limit:
        raise ValueError("Raw Base prompt exceeds configured rollout.prompt_length")
    request_id = request_id or uuid4().hex
    response_ids, response_mask, response_logprobs, events = [], [], [], []
    have_logprobs = bool(sampling_params.get("logprobs", False))
    generated = searches = invalid = 0
    status, answer = "turn_limit", ""
    generate_seconds = tool_seconds = 0.0
    for turn in range(config["max_turns"]):
        context_remaining = config["max_context_tokens"] - len(prompt_ids) - len(response_ids)
        storage_remaining = (response_limit - len(response_ids)
                             if response_limit is not None else context_remaining)
        remaining_generated = config["max_total_generated_tokens"] - generated
        if remaining_generated <= 0:
            status = "generated_token_limit"
            break
        if min(context_remaining, storage_remaining) <= 0:
            status = "context_limit"
            break
        budget = min(config["max_new_tokens"], remaining_generated, context_remaining, storage_remaining)
        params = {**sampling_params, "max_tokens": budget, "n": 1,
                  "stop": ["</search>", "</answer>"], "include_stop_str_in_output": True,
                  "skip_special_tokens": False}
        started = time.monotonic()
        packet = await generate(request_id=request_id, prompt_ids=prompt_ids + response_ids,
                                sampling_params=params)
        generate_seconds += time.monotonic() - started
        ids = _tokens(_packet_field(packet, "token_ids"), "vLLM generated output")
        if not ids or len(ids) > budget:
            raise RuntimeError("vLLM returned empty/over-budget generation; check max_tokens adapter patch")
        probabilities = _packet_field(packet, "log_probs")
        if have_logprobs and (probabilities is None or len(probabilities) != len(ids)):
            raise RuntimeError("vLLM rollout log probabilities are missing/misaligned")
        response_ids.extend(ids)
        response_mask.extend([1] * len(ids))
        if have_logprobs:
            response_logprobs.extend(float(value) for value in probabilities)
        generated += len(ids)
        text = _decode_action(tokenizer, ids)
        action = parse_action(text)
        events.append({"role": "assistant", "content": text, "turn": turn,
                       "generated_tokens": len(ids), "input_tokens": len(prompt_ids) + len(response_ids) - len(ids),
                       "format_valid": valid_step_format(text), "action": action,
                       "response_token_start": len(response_ids) - len(ids),
                       "response_token_end": len(response_ids)})
        if action is None:
            invalid += 1
            if invalid >= config["max_invalid_actions"]:
                status = "invalid_action_limit"
                break
            observation = {"role": "environment", "kind": "invalid_action_feedback",
                           "content": _observation_text(INVALID_FEEDBACK, text)}
        elif action["action"] == "answer":
            answer, status = action["value"], "ok"
            break
        elif searches >= config["max_searches"]:
            observation = {"role": "environment", "kind": "search_limit_feedback",
                           "content": _observation_text(SEARCH_LIMIT_FEEDBACK, text)}
        else:
            started = time.monotonic()
            retrieved = await retrieve(action["value"])
            tool_seconds += time.monotonic() - started
            if not isinstance(retrieved, dict) or retrieved.get("query") != action["value"]:
                raise RuntimeError("Retriever result does not match the requested query")
            searches += 1
            observation = {"role": "environment", "kind": "retrieval", "retrieval": retrieved,
                           "content": _observation_text(retrieved["content"], text)}
        observation_ids = _encode(tokenizer, observation["content"])
        capacity = min(config["max_context_tokens"] - len(prompt_ids) - len(response_ids),
                       response_limit - len(response_ids) if response_limit is not None else config["max_context_tokens"])
        if len(observation_ids) > capacity:
            # Do not silently truncate documents or overrun verl's padded tensor.
            # The event remains available for audit but its tokens aren't fed in.
            observation["included_in_context"] = False
            events.append(observation)
            status = "context_limit"
            break
        observation["included_in_context"] = True
        observation["response_token_start"] = len(response_ids)
        response_ids.extend(observation_ids)
        response_mask.extend([0] * len(observation_ids))
        if have_logprobs:
            response_logprobs.extend([0.0] * len(observation_ids))
        observation["response_token_end"] = len(response_ids)
        events.append(observation)
    record = {
        "sample_id": example["sample_id"], "question": example["question"],
        "data_source": example["data_source"], "initial_prompt": prompt,
        "events": events, "status": status, "answer": answer,
        "search_count": searches, "invalid_action_count": invalid,
        "generated_tokens": generated,
    }
    scores = score_record(record, example["golden_answers"], example["data_source"])
    record.update(scores)
    return {
        "prompt_ids": prompt_ids, "response_ids": response_ids, "response_mask": response_mask,
        "response_logprobs": response_logprobs if have_logprobs else None,
        "reward_score": float(scores["score"]), "num_turns": 1 + len(events),
        "metrics": {"generate_sequences": generate_seconds, "tool_calls": tool_seconds},
        "extra_fields": {"search_grpo_trajectory": record, "reward_extra_info": scores},
    }


def example_from_kwargs(kwargs):
    extra = kwargs.get("extra_info") or {}
    question = extra.get("question", kwargs.get("question"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Search-R1 raw Base agent requires extra_info.question")
    reward_model = kwargs.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth", extra.get("golden_answers"))
    return {
        "question": question,
        "sample_id": str(extra.get("sample_id", extra.get("id", extra.get("index", "")))),
        "data_source": kwargs["data_source"], "golden_answers": gold_aliases(ground_truth),
    }


@register("search_r1_base")
class SearchR1AgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, search_config=None, **kwargs):
        if getattr(cls, "_search_class_initialized", False):
            return
        if AgentLoopOutput is None:
            raise RuntimeError("SearchR1AgentLoop production use requires pinned verl v0.6.1")
        cls.search_config = search_config_from_trainer(config, search_config)
        cls.prompt_limit = config.actor_rollout_ref.rollout.prompt_length
        cls.response_limit = config.actor_rollout_ref.rollout.response_length
        cls.retrieval_broker = shared_retrieval_broker(
            cls.search_config, run_id=config.trainer.experiment_name)
        cls._search_class_initialized = cls._class_initialized = True

    async def run(self, sampling_params, **kwargs):
        async def retrieve(query):
            return await self.retrieval_broker.retrieve.remote(query)

        output = await run_token_rollout(
            example_from_kwargs(kwargs), self.tokenizer, self.server_manager.generate,
            retrieve, self.search_config, sampling_params,
            prompt_limit=self.prompt_limit, response_limit=self.response_limit,
        )
        return AgentLoopOutput(**output)
