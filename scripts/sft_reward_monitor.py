#!/usr/bin/env python3
"""Generate held-out RAG trajectories during SFT and report diagnostic rewards.

This module never backpropagates a reward: SFT still optimizes masked token CE.
``total_reward`` is the Search-R1-style binary normalized answer EM; the stricter
think/action format reward is reported separately, without a shaping bonus.
Only generated actions are scored; the prompt's example answer is never scored.

The caller owns distributed synchronization, sharding and W&B logging. Every
rank calls this module on its unwrapped model and separate output file. There
are deliberately no distributed collectives in generation. Retrieval failures
raise, so infrastructure failures cannot silently become low model rewards.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from qa_metrics import metric_spec, score_answer


DEFAULTS = {
    "max_new_tokens": 512,
    "max_searches": 3,
    "max_turns": 6,
    "max_invalid_actions": 2,
    "max_context_tokens": 8192,
    "max_total_generated_tokens": 2048,
    "topk": 3,
    "passage_chars": 1800,
    "retriever_url": "http://127.0.0.1:8000/retrieve",
    "retrieval_timeout_seconds": 180,
    "retrieval_retries": 2,
    "progress_every": 1,
}
ACTION_RE = re.compile(r"<(?P<action>search|answer)>(?P<value>.*?)</(?P=action)>", re.S)
STRICT_STEP_RE = re.compile(
    r"\s*<think>(?P<thought>.*?)</think>\s*"
    r"<(?P<action>search|answer)>(?P<value>.*?)</(?P=action)>\s*", re.S)
PROTOCOL_RE = re.compile(r"</?(?:think|search|answer|information)>")
INVALID_FEEDBACK = (
    "<information>Your previous response did not contain a valid complete action. "
    "Reason inside <think> and </think>, then use <search>query</search> or "
    "<answer>answer</answer>.</information>"
)
SEARCH_LIMIT_FEEDBACK = (
    "<information>The search budget is exhausted. Use the available information "
    "and give your final answer inside <answer> and </answer>.</information>"
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def validate_config(config):
    """Allow the caller's extra metadata, but validate every rollout setting."""
    result = {**DEFAULTS, **config}
    for key in ("max_new_tokens", "max_searches", "max_turns", "max_invalid_actions",
                "max_context_tokens", "max_total_generated_tokens", "topk",
                "passage_chars", "progress_every"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"Reward monitor {key} must be a positive integer")
    if type(result["retrieval_retries"]) is not int or result["retrieval_retries"] < 0:
        raise ValueError("Reward monitor retrieval_retries must be a nonnegative integer")
    if (not isinstance(result["retrieval_timeout_seconds"], (int, float))
            or result["retrieval_timeout_seconds"] <= 0):
        raise ValueError("Reward monitor retrieval_timeout_seconds must be positive")
    if result.get("format_weight", 0) != 0:
        raise ValueError("This monitor keeps total_reward=answer_reward; format_weight must be zero")
    url = urlsplit(result["retriever_url"])
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("Retriever URL must be HTTP(S) without credentials/query/fragment")
    return result


def parse_action(text):
    """Use the first complete action, as the official Search-R1 agent does.

    Extra prose and broken think tags do not erase a recoverable answer. They
    instead lower the independent format reward. Empty actions are invalid.
    """
    match = ACTION_RE.search(text)
    if match is None:
        return None
    value = html.unescape(match.group("value")).strip()
    if not value or PROTOCOL_RE.search(value):
        return None
    if match.group("action") == "search" and len(value) > 500:
        return None
    return {"action": match.group("action"), "value": value,
            "start": match.start(), "end": match.end()}


def valid_step_format(text):
    """A step has exactly one nonempty think block followed by one action."""
    match = STRICT_STEP_RE.fullmatch(text)
    if match is None:
        return False
    thought, value = match.group("thought"), match.group("value")
    if not thought.strip() or not html.unescape(value).strip():
        return False
    if PROTOCOL_RE.search(thought) or PROTOCOL_RE.search(value):
        return False
    return parse_action(text) is not None


def score_trajectory(answer, gold, data_source, *, completed, all_steps_valid):
    """Keep official-style EM reward distinct from stricter trajectory format."""
    answer = answer if completed else ""
    search_r1 = score_answer(answer, gold, "search-r1")
    qa = score_answer(answer, gold, data_source)
    return {
        "total_reward": search_r1["em"],
        "answer_reward": search_r1["em"],
        "format_reward": int(completed and all_steps_valid),
        "qa_em": qa["em"], "qa_f1": qa["f1"],
    }


def retrieve(query, config):
    """Call the existing local E5 service; never route it through HTTP_PROXY."""
    payload = {"queries": [query], "topk": config["topk"], "return_scores": True}
    opener = build_opener(ProxyHandler({}))
    raw = None
    for attempt in range(config["retrieval_retries"] + 1):
        request = Request(config["retriever_url"], data=_canonical(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with opener.open(request, timeout=config["retrieval_timeout_seconds"]) as response:
                raw = json.load(response)
            break
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"Reward monitor retriever HTTP {error.code}") from None
            if attempt == config["retrieval_retries"]:
                raise RuntimeError(f"Reward monitor retriever HTTP {error.code}, retries exhausted") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            if attempt == config["retrieval_retries"]:
                raise RuntimeError("Reward monitor retriever transport/JSON failure, retries exhausted") from None
        time.sleep(min(2 ** attempt, 8))
    batches = raw.get("result") if isinstance(raw, dict) else None
    if not isinstance(batches, list) or len(batches) != 1 or not batches[0]:
        raise RuntimeError("Reward monitor retriever returned an invalid/empty result batch")
    documents, lines = [], []
    for index, hit in enumerate(batches[0][:config["topk"]], 1):
        if not isinstance(hit, dict):
            raise RuntimeError("Reward monitor retriever returned a non-object passage")
        document = hit.get("document", hit)
        content = document.get("contents") if isinstance(document, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Reward monitor retrieved passage lacks contents")
        shown = content[:config["passage_chars"]]
        documents.append({"id": str(document.get("id", "")), "contents": shown,
                          "score": hit.get("score")})
        lines.append(f"Doc {index}: {shown}")
    information = "<information>" + html.escape("\n".join(lines), quote=False) + "</information>"
    return {"query": query, "content": information, "documents": documents,
            "request": payload, "transport_attempts": attempt + 1}


def rollout(example, generate_fn, retrieve_fn, config, *, build_prompt_fn=None,
            canonical_information_fn=None, canonical_assistant_fn=None):
    """Run one question; injected generation/retrieval make protocol tests offline.

    ``generate_fn(context, budget)`` returns text/generated_tokens/input_tokens.
    It may instead return ``context_limit=True`` without generating any tokens.
    Gold answers are read only after the generation loop for scoring.
    """
    if (build_prompt_fn is None or canonical_information_fn is None
            or canonical_assistant_fn is None):
        from search_sft_protocol import build_prompt, canonical_information, canonical_assistant
        build_prompt_fn = build_prompt_fn or build_prompt
        canonical_information_fn = canonical_information_fn or canonical_information
        canonical_assistant_fn = canonical_assistant_fn or canonical_assistant
    config = validate_config(config)
    context = build_prompt_fn(example["question"])
    initial_prompt = context
    events = []
    answer, status = "", "turn_limit"
    searches = generated = invalid = 0
    all_steps_valid = True
    started = time.monotonic()
    for turn in range(config["max_turns"]):
        budget = min(config["max_new_tokens"], config["max_total_generated_tokens"] - generated)
        if budget <= 0:
            status = "generated_token_limit"
            break
        packet = generate_fn(context, budget)
        if packet.get("context_limit"):
            status = "context_limit"
            break
        text = packet["text"]
        generated += packet["generated_tokens"]
        action = parse_action(text)
        step_valid = valid_step_format(text)
        all_steps_valid = all_steps_valid and step_valid
        event = {"role": "assistant", "content": text, "turn": turn,
                 "generated_tokens": packet["generated_tokens"],
                 "input_tokens": packet["input_tokens"], "format_valid": step_valid,
                 "action": action}
        events.append(event)
        # Match the preparation serializer, including when the token completing
        # </search> already contains a newline. The observation owns one newline.
        context += canonical_assistant_fn(text) if text.strip() else text
        if action is None:
            invalid += 1
            if invalid >= config["max_invalid_actions"]:
                status = "invalid_action_limit"
                break
            observation = {"role": "environment", "content": canonical_information_fn(INVALID_FEEDBACK),
                           "kind": "invalid_action_feedback"}
        elif action["action"] == "answer":
            answer, status = action["value"], "ok"
            break
        elif searches >= config["max_searches"]:
            all_steps_valid = False
            observation = {"role": "environment", "content": canonical_information_fn(SEARCH_LIMIT_FEEDBACK),
                           "kind": "search_limit_feedback"}
        else:
            # An infrastructure error propagates. It is not an incorrect answer.
            retrieved = retrieve_fn(action["value"])
            searches += 1
            observation = {"role": "environment", "content": canonical_information_fn(retrieved["content"]),
                           "kind": "retrieval", "retrieval": retrieved}
        events.append(observation)
        context += observation["content"]
    completed = status == "ok"
    scores = score_trajectory(answer, example["golden_answers"], example["data_source"],
                              completed=completed, all_steps_valid=all_steps_valid)
    return {
        "sample_id": example["sample_id"], "question": example["question"],
        "data_source": example["data_source"], "golden_answers": example["golden_answers"],
        "initial_prompt": initial_prompt, "events": events, "status": status, "answer": answer,
        "search_count": searches, "generated_tokens": generated, "model_calls": sum(
            event["role"] == "assistant" for event in events),
        "elapsed_seconds": time.monotonic() - started, **scores,
    }


def aggregate_records(records, elapsed_seconds):
    """Flat W&B scalars. Weight shard means by reward/num_examples to combine."""
    if not records:
        raise ValueError("Cannot aggregate an empty reward monitor shard")
    count = len(records)
    metrics = {"reward/num_examples": count, "reward/elapsed_seconds": elapsed_seconds}
    for key in ("total_reward", "answer_reward", "format_reward", "qa_em", "qa_f1"):
        metrics[f"reward/{key}"] = sum(row[key] for row in records) / count
    metrics["reward/answer_rate"] = sum(row["status"] == "ok" for row in records) / count
    metrics["reward/search_rate"] = sum(row["search_count"] > 0 for row in records) / count
    metrics["reward/mean_searches"] = sum(row["search_count"] for row in records) / count
    metrics["reward/mean_generated_tokens"] = sum(row["generated_tokens"] for row in records) / count
    return metrics


@contextmanager
def _preserve_training_state(model, torch, device):
    """Monitoring cannot change dropout state, cache policy, or training RNG."""
    states = [(module, module.training) for module in model.modules()]
    python_rng = random.getstate()
    try:
        import numpy as np
        numpy_rng = np.random.get_state()
    except ImportError:
        np = None
        numpy_rng = None
    old_use_cache = getattr(model.config, "use_cache", None)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            model.config.use_cache = True
            yield
    finally:
        for module, training in states:
            module.training = training
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        random.setstate(python_rng)
        if np is not None:
            np.random.set_state(numpy_rng)


def run_reward_eval(model, tokenizer, examples, config, outputpath):
    """Autoregressively evaluate one shard and persist reproducible local traces.

    CUDA execution uses BF16 autocast, including the trainable FP32 tied head.
    The caller must pass an unwrapped PEFT model, not DDP/FSDP. Generation uses
    ``synced_gpus=False`` so uneven trajectories do not enter NCCL collectives.
    Existing output files are never overwritten; give each step/rank a new path.
    """
    import torch
    from transformers import GenerationConfig, StopStringCriteria, StoppingCriteriaList
    from search_sft_protocol import PROTOCOL_VERSION
    from prepare_data import PROMPT

    config = validate_config(config)
    examples = list(examples)
    if not examples:
        raise ValueError("Reward monitor needs at least one held-out example per shard")
    identifiers = [row["sample_id"] for row in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Reward monitor examples contain duplicate sample IDs")
    outputpath = Path(outputpath)
    outputpath.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = outputpath.with_suffix(".manifest.json")
    summary_path = outputpath.with_suffix(".summary.json")
    if any(path.exists() for path in (outputpath, manifest_path, summary_path)):
        raise FileExistsError(f"Reward monitor snapshot already exists: {outputpath}")
    model_device = model.get_input_embeddings().weight.device
    if model_device.type not in ("cuda", "cpu"):
        raise ValueError("Reward monitor requires a loaded CPU/CUDA model")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Reward monitor tokenizer needs an EOS token")
    generation_config = GenerationConfig(
        do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=eos,
        bos_token_id=tokenizer.bos_token_id,
    )
    stops = StoppingCriteriaList([StopStringCriteria(
        tokenizer=tokenizer, stop_strings=["</search>", "</answer>"])])
    manifest = {
        "status": "running", "started_at": _now(), "config": config,
        "sample_ids": identifiers, "examples_sha256": hashlib.sha256(_canonical(examples).encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_code_sha256": hashlib.sha256(
            Path(__file__).with_name("search_sft_protocol.py").read_bytes()).hexdigest(),
        "monitor_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reward_definition": "total_reward=answer_reward=Search-R1 normalized exact-match; no format bonus",
        "format_reward_definition": "1 only for completed answer with every assistant step exactly nonempty think then search/answer; no invalid/budget-exceeding actions",
        "reward_metric": metric_spec("search-r1"),
        "qa_metric": {source: metric_spec(source) for source in sorted({row["data_source"] for row in examples})},
        "gold_policy": "Gold aliases are used after generation only; no reward gradients or teacher answer prefills",
        "distributed_policy": "Caller shards examples; all generation local with synced_gpus=False",
    }
    _write_json(manifest_path, manifest)

    def generate(context, budget):
        encoded = tokenizer(context, add_special_tokens=False, return_tensors="pt").to(model_device)
        width = encoded["input_ids"].shape[1]
        available = config["max_context_tokens"] - width
        if available <= 0:
            return {"context_limit": True}
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if model_device.type == "cuda" else nullcontext()
        with autocast:
            output = model.generate(
                **encoded, generation_config=generation_config,
                max_new_tokens=min(budget, available), stopping_criteria=stops, use_cache=True,
                return_dict_in_generate=False, output_scores=False,
                synced_gpus=False, logits_to_keep=1,
            )
        ids = output[0, width:].detach().cpu().tolist()
        # Keep special <think> tokens. Remove only terminal EOS before parsing.
        text_ids = ids[:-1] if ids and ids[-1] == eos else ids
        text = tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return {"text": text, "generated_tokens": len(ids), "input_tokens": width}

    started = time.monotonic()
    records = []
    try:
        with outputpath.open("x", encoding="utf-8") as handle, _preserve_training_state(model, torch, model_device):
            for index, example in enumerate(examples, 1):
                record = rollout(example, generate, lambda query: retrieve(query, config), config)
                handle.write(_canonical(record) + "\n")
                handle.flush()
                records.append(record)
                if index % config["progress_every"] == 0 or index == len(examples):
                    print(_canonical({"event": "sft_reward_progress", "output": str(outputpath),
                                      "completed": index, "total": len(examples),
                                      "sample_id": record["sample_id"], "status": record["status"],
                                      "answer_reward": record["answer_reward"], "format_reward": record["format_reward"],
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        metrics = aggregate_records(records, time.monotonic() - started)
        _write_json(summary_path, metrics)
        manifest.update(status="complete", completed_at=_now(), num_completed=len(records), metrics=metrics)
        _write_json(manifest_path, manifest)
        return metrics
    except Exception as error:
        manifest.update(status="failed", failed_at=_now(), num_completed=len(records),
                        error_type=type(error).__name__, error=str(error))
        _write_json(manifest_path, manifest)
        raise
