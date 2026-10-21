#!/usr/bin/env python3
"""Evaluate one GPU shard of the reserved GRPO set with Search-R1 rollouts.

The model receives only the Base prompt and its actual action/observation
history. Gold answers remain in local terminal records for scoring. This is
greedy HF + PEFT inference, not the older fixed, single-retrieval RAG baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from prepare_data import PROMPT
from qa_metrics import answer_diagnostics, metric_spec, score_answer


DEFAULTS = {
    "seed": 1, "batch_size": 4, "max_searches": 3, "topk": 3,
    "passage_chars": 1800, "max_new_tokens": 512,
    "max_context_tokens": 8192, "max_total_generated_tokens": 2048,
    "retriever_url": "http://127.0.0.1:8000/retrieve",
    "retrieval_timeout_seconds": 180, "retrieval_retries": 2,
}
STOPS = ("</search>", "</answer>")
ACTION_RE = re.compile(
    r"\s*(?:<think>(?P<thought>.*?)</think>\s*)?"
    r"<(?P<action>search|answer)>(?P<value>[^<>]+)</(?P=action)>\s*", re.S)
PROTOCOL_RE = re.compile(r"</?(?:think|search|answer|information)>")


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def append_record(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def worker_lock(path):
    with Path(path).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process already owns this evaluation worker") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_checkpoint(path):
    """Repair only an interrupted last JSON line, preserving all earlier rows."""
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as handle:
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                return records
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if handle.read(1):
                    raise ValueError("Corrupt prediction record before the final line") from None
                handle.truncate(position)
                return records
            if not isinstance(record, dict):
                raise ValueError("Prediction checkpoint contains a non-object record")
            if not line.endswith(b"\n"):
                handle.seek(0, 2)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            records.append(record)


def load_configuration(path):
    original = read_json(path)
    if not isinstance(original, dict):
        raise ValueError("Evaluation config must be a JSON object")
    for key in ("model_path", "adapter_path", "data_file", "output_dir", "training_prepared_dir"):
        if key not in original and key == "training_prepared_dir":
            continue
        value = original.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Configuration requires {key}")
        original[key] = str(Path(value).expanduser().resolve())
    config = {**DEFAULTS, **original}
    for key in ("batch_size", "topk", "passage_chars", "max_new_tokens",
                "max_context_tokens", "max_total_generated_tokens"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "max_searches", "retrieval_retries"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not isinstance(config["retrieval_timeout_seconds"], (int, float)) or config["retrieval_timeout_seconds"] <= 0:
        raise ValueError("retrieval_timeout_seconds must be positive")
    url = urlsplit(config["retriever_url"])
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment):
        raise ValueError("Retriever URL must be HTTP(S), without credentials, query, or fragment")
    if not Path(config["model_path"]).is_dir() or not Path(config["adapter_path"]).is_dir():
        raise ValueError("The local Base model and trained adapter must both exist")
    return original, config


def load_examples(path):
    examples, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("partition") != "grpo":
                raise ValueError(f"Expected original reserved-GRPO rows at line {line_number}")
            sid, source, question, gold = (row.get(key) for key in
                                            ("sample_id", "data_source", "question", "golden_answers"))
            if not isinstance(sid, str) or not sid or sid in seen:
                raise ValueError("Reserved-GRPO sample identifiers must be unique")
            if source not in ("nq", "hotpotqa") or not isinstance(question, str) or not question.strip():
                raise ValueError("Invalid reserved-GRPO source or question")
            if not isinstance(gold, list) or not gold or any(not isinstance(item, str) or not item.strip() for item in gold):
                raise ValueError("Reserved-GRPO examples require nonempty original gold aliases")
            seen.add(sid)
            examples.append({"sample_id": sid, "data_source": source, "question": question,
                             "golden_answers": gold, "partition": "grpo",
                             "source_index": len(examples), "source_line": line_number})
    if len(examples) != 2000 or Counter(row["data_source"] for row in examples) != {"nq": 1000, "hotpotqa": 1000}:
        raise ValueError("Expected the original 2000 reserved questions: NQ1000 + HotpotQA1000")
    return examples


def verify_prepared_run(original, config):
    """Validate the root preparation manifest before loading any GPU runtime."""
    output = Path(config["output_dir"])
    run_path = output / "run_manifest.json"
    manifest = read_json(run_path)
    if manifest.get("config_sha256") != digest(original) or manifest.get("data_sha256") != sha256(config["data_file"]):
        raise ValueError("Evaluation configuration or reserved dataset differs from its prepared run")
    script_root = Path(__file__).resolve().parent
    code_hashes = manifest.get("code_sha256", {})
    for name in (Path(__file__).name, "prepare_data.py", "qa_metrics.py"):
        if code_hashes.get(name) != sha256(script_root / name):
            raise ValueError(f"Prepared evaluation code hash mismatch: {name}")
    for filename, expected in manifest.get("asset_sha256", {}).items():
        path = Path(filename)
        if not path.is_absolute() or not path.is_file() or sha256(path) != expected:
            raise ValueError("A model, adapter, or tokenizer asset changed after evaluation preparation")
    return manifest, sha256(run_path)


def parse_action(text):
    match = ACTION_RE.fullmatch(text)
    if not match or PROTOCOL_RE.search(match.group("thought") or ""):
        raise ValueError("Expected one optional complete think block followed by exactly one search/answer action")
    value = html.unescape(match.group("value")).strip()
    if not value or PROTOCOL_RE.search(value):
        raise ValueError("Action content is empty or contains protocol tags")
    if match.group("action") == "search" and len(value) > 500:
        raise ValueError("Search query exceeds the 500-character protocol limit")
    return match.group("action"), value


def retrieve_batch(queries, config):
    payload = {"queries": queries, "topk": config["topk"], "return_scores": True}
    body = canonical(payload).encode()
    # The configured retriever is a direct local service; do not send its
    # questions through the user's external HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    raw = None
    for attempt in range(config["retrieval_retries"] + 1):
        try:
            request = Request(config["retriever_url"], data=body,
                              headers={"Content-Type": "application/json"}, method="POST")
            with opener.open(request, timeout=config["retrieval_timeout_seconds"]) as response:
                raw = json.load(response)
            break
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"Retriever HTTP {error.code}") from None
            if attempt == config["retrieval_retries"]:
                raise RuntimeError(f"Retriever HTTP {error.code} after transport retries") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            if attempt == config["retrieval_retries"]:
                raise RuntimeError("Retriever transport/JSON failure after retries") from None
        time.sleep(min(2 ** attempt, 8))
    batches = raw.get("result") if isinstance(raw, dict) else None
    if not isinstance(batches, list) or len(batches) != len(queries):
        raise RuntimeError("Retriever batch response does not match the requested queries")
    results = []
    for query_index, (query, hits) in enumerate(zip(queries, batches)):
        try:
            if not isinstance(hits, list) or not hits:
                raise ValueError("Retriever returned no passages")
            documents, lines = [], []
            for index, hit in enumerate(hits[:config["topk"]], 1):
                if not isinstance(hit, dict):
                    raise ValueError("Retriever returned a non-object passage")
                document = hit.get("document", hit)
                contents = document.get("contents") if isinstance(document, dict) else None
                if not isinstance(contents, str) or not contents.strip():
                    raise ValueError("Retrieved passage lacks contents")
                shown = contents[:config["passage_chars"]]
                documents.append({"id": str(document.get("id", "")), "contents": shown,
                                  "score": hit.get("score")})
                lines.append(f"Doc {index}: {shown}")
            content = "<information>" + html.escape("\n".join(lines), quote=False) + "</information>"
            results.append({"query": query, "content": content, "documents": documents,
                            "retrieval_raw": raw, "query_index_in_batch": query_index, "request": payload,
                            "transport_attempts": attempt + 1})
        except ValueError as error:
            results.append({"query": query, "error": str(error), "request": payload,
                            "retrieval_raw": raw, "query_index_in_batch": query_index,
                            "transport_attempts": attempt + 1})
    return results


class StopWorker(RuntimeError):
    pass


def new_state(example, worker, binding):
    question = example["question"]
    question = question if question.endswith("?") else question + "?"
    return {"example": example, "prompt": PROMPT.format(question=question),
            "started_clock": time.monotonic(), "started_at": now(), "events": [],
            "search_count": 0, "worker_index": worker, "binding_sha256": binding,
            "usage": {"prompt_tokens": 0, "generated_tokens": 0, "model_calls": 0,
                      "model_attempts": 0, "retrieval_calls": 0, "oom_retries": 0}}


def finish(state, path, status, answer="", error_type=None, error=None):
    record = {**state["example"], "worker_index": state["worker_index"],
              "binding_sha256": state["binding_sha256"], "status": status,
              "answer": answer, "em": 0, "f1": 0.0,
              "events": state["events"], "search_count": state["search_count"],
              "usage": dict(state["usage"]), "started_at": state["started_at"],
              "finished_at": now(), "elapsed_seconds": round(time.monotonic() - state["started_clock"], 3)}
    if status == "ok":
        record.update(score_answer(answer, record["golden_answers"], record["data_source"]))
        record["answer_diagnostics"] = answer_diagnostics(answer, record["golden_answers"], record["data_source"])
    if error_type:
        record["error_type"] = error_type
    if error:
        record["error"] = error
    append_record(path, record)
    state["finished"] = True
    print(canonical({key: record[key] for key in
                     ("worker_index", "sample_id", "source_index", "data_source", "status", "em", "search_count")}),
          flush=True)


def generation_packets(states, model, tokenizer, stop_criteria, config, torch, amount):
    """Yield completed local generations; split OOM batches without dropping IDs."""
    for state in states:
        state["usage"]["model_attempts"] += 1
    encoded = output = None
    failure = None
    started = time.monotonic()
    try:
        encoded = tokenizer([state["prompt"] for state in states], add_special_tokens=False,
                            padding=True, return_tensors="pt").to("cuda:0")
        width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=amount, do_sample=False, use_cache=True,
                eos_token_id=151643, pad_token_id=151643,
                stopping_criteria=stop_criteria, return_dict_in_generate=True,
                output_scores=False, logits_to_keep=1,
            )
        token_rows = output.sequences[:, width:].detach().cpu().tolist()
    except torch.cuda.OutOfMemoryError:
        failure = {"error_type": "OutOfMemoryError", "error": "Single-example generation exhausted GPU memory",
                   "status": "oom"}
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": "Local model generation failed", "status": "model_error"}
    finally:
        del encoded, output
    # Leave the exception handler before retrying, so its traceback cannot hold
    # the failed generation's CUDA tensors alive during the smaller retry.
    if failure and failure["status"] == "oom":
        gc.collect()
        torch.cuda.empty_cache()
        for state in states:
            state["usage"]["oom_retries"] += 1
        if len(states) > 1:
            middle = len(states) // 2
            yield from generation_packets(states[:middle], model, tokenizer, stop_criteria, config, torch, amount)
            yield from generation_packets(states[middle:], model, tokenizer, stop_criteria, config, torch, amount)
            return
        yield states[0], failure
        raise StopWorker("Stopped after checkpointing a single-example OOM; other questions remain pending")
    if failure:
        for state in states:
            yield state, failure
        raise StopWorker("Stopped after checkpointing a local model failure; see worker log")
    elapsed = round(time.monotonic() - started, 3)
    for state, ids in zip(states, token_rows):
        eos = ids.index(151643) if 151643 in ids else None
        visible_ids = ids[:eos] if eos is not None else ids
        raw = tokenizer.decode(visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        ends = [raw.find(stop) + len(stop) for stop in STOPS if stop in raw]
        end = min(ends) if ends else None
        # A stop string may end inside a token. Preserve any overhang in audit
        # data, while exposing exactly the intended stopped continuation.
        text = raw[:end] if end is not None else raw
        generated = len(visible_ids) + int(eos is not None and end is None)
        yield state, {"content": text, "raw_decoded": raw, "stop_overhang": raw[end:] if end is not None else "",
                      "input_tokens": state["input_tokens"], "generated_tokens": generated,
                      "token_ids": visible_ids, "eos_without_action": eos is not None and end is None,
                      "action_stop_found": end is not None, "generation_limit": amount,
                      "batch_size": len(states), "batch_elapsed_seconds": elapsed}


def rollout_chunk(examples, worker, binding, model, tokenizer, stops, config, torch, path):
    states = [new_state(row, worker, binding) for row in examples]
    while any(not state.get("finished") for state in states):
        groups = defaultdict(list)
        for state in states:
            if state.get("finished"):
                continue
            size = len(tokenizer(state["prompt"], add_special_tokens=False)["input_ids"])
            state["input_tokens"] = size
            context_left = config["max_context_tokens"] - size
            total_left = config["max_total_generated_tokens"] - state["usage"]["generated_tokens"]
            if context_left <= 0:
                finish(state, path, "context_budget_exceeded")
                continue
            if total_left <= 0:
                finish(state, path, "generation_budget_exceeded")
                continue
            amount = min(config["max_new_tokens"], context_left, total_left)
            state["limit_reason"] = ("generation_budget_exceeded" if amount == total_left
                                     else "context_budget_exceeded" if amount == context_left
                                     else "unfinished_generation")
            groups[amount].append(state)
        searches = []
        for amount, group in groups.items():
            for state, packet in generation_packets(group, model, tokenizer, stops, config, torch, amount):
                if packet.get("error_type"):
                    finish(state, path, packet["status"], error_type=packet["error_type"], error=packet["error"])
                    continue
                state["usage"]["model_calls"] += 1
                state["usage"]["prompt_tokens"] += packet["input_tokens"]
                state["usage"]["generated_tokens"] += packet["generated_tokens"]
                state["events"].append({"role": "assistant", **packet})
                if not packet["action_stop_found"]:
                    finish(state, path, "invalid_action" if packet["eos_without_action"] else state["limit_reason"],
                           error="Generation ended without a complete search/answer action")
                    continue
                try:
                    action, value = parse_action(packet["content"])
                except ValueError as error:
                    finish(state, path, "invalid_action", error=str(error))
                    continue
                state["events"][-1].update({"action": action, "value": value})
                if action == "answer":
                    finish(state, path, "ok", answer=value)
                elif state["search_count"] >= config["max_searches"]:
                    finish(state, path, "search_budget_exceeded")
                else:
                    state["prompt"] += packet["content"] + "\n"
                    searches.append((state, value))
        if searches:
            for state, _ in searches:
                state["usage"]["retrieval_calls"] += 1
                state["search_count"] += 1
            try:
                retrievals = retrieve_batch([query for _, query in searches], config)
            except (ValueError, RuntimeError, OSError) as error:
                for state, query in searches:
                    state["events"].append({"role": "environment", "query": query, "error": str(error),
                                            "retriever_url": config["retriever_url"]})
                    finish(state, path, "retrieval_error", error_type=type(error).__name__, error=str(error))
                continue
            for (state, _), result in zip(searches, retrievals):
                state["events"].append({"role": "environment", "retriever_url": config["retriever_url"], **result})
                if result.get("error"):
                    finish(state, path, "retrieval_error", error=result["error"])
                else:
                    state["prompt"] += result["content"] + "\n"


def run(args):
    original, config = load_configuration(args.config)
    all_examples = load_examples(config["data_file"])
    _, prepared_sha = verify_prepared_run(original, config)
    examples = [row for row in all_examples if row["source_index"] % args.num_workers == args.worker_index]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", visible):
        raise ValueError("Each worker must receive exactly one explicit GPU UUID in CUDA_VISIBLE_DEVICES")
    selected = config.get("selected_gpu_uuids")
    if selected is not None and (len(selected) != args.num_workers or selected[args.worker_index] != visible):
        raise ValueError("Worker GPU UUID differs from its configured device")
    root = Path(__file__).resolve().parent
    binding = {"configuration": config, "prepared_run_manifest_sha256": prepared_sha,
               "data_sha256": sha256(config["data_file"]), "worker_index": args.worker_index,
               "num_workers": args.num_workers, "gpu_uuid": visible,
               "code_sha256": {name: sha256(root / name) for name in
                               (Path(__file__).name, "prepare_data.py", "qa_metrics.py")},
               "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
               "decoding": "greedy; complete Search-R1 actions; no chat template",
               "metric_specs": {name: metric_spec(name) for name in ("nq", "hotpotqa")}}
    binding_sha = digest(binding)
    output = Path(config["output_dir"])
    predictions = output / f"worker-{args.worker_index}.predictions.jsonl"
    manifest_path = output / f"worker-{args.worker_index}.manifest.json"
    with worker_lock(output / f".worker-{args.worker_index}.lock"):
        if manifest_path.exists():
            if read_json(manifest_path).get("binding_sha256") != binding_sha:
                raise ValueError("Worker resume refused: data, config, assets, code, GPU, or shard changed")
        else:
            if predictions.exists():
                raise ValueError("Worker predictions exist without their binding manifest")
            write_json(manifest_path, {"format_version": 1, "created_at": now(),
                                       "binding": binding, "binding_sha256": binding_sha})
        previous = load_checkpoint(predictions)
        expected = {row["sample_id"]: row for row in examples}
        for record in previous:
            if record.get("sample_id") not in expected or record.get("binding_sha256") != binding_sha:
                raise ValueError("Prediction checkpoint contains a foreign sample or evaluation binding")
            sample = expected[record["sample_id"]]
            if any(record.get(key) != sample[key] for key in
                   ("source_index", "data_source", "question", "golden_answers")):
                raise ValueError("Prediction checkpoint sample differs from the original reserved dataset")
        completed = {row["sample_id"] for row in previous}
        pending = [row for row in examples if row["sample_id"] not in completed]
        if not pending:
            print(canonical({"worker_index": args.worker_index, "selected": len(examples), "pending": 0}), flush=True)
            return
        import torch
        from peft import PeftModel
        from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                                  StopStringCriteria, StoppingCriteriaList, set_seed)

        if torch.cuda.device_count() != 1:
            raise ValueError("Evaluation workers may see exactly one selected GPU")
        torch.cuda.set_device(0)
        if "3090" not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
            raise ValueError("This recipe requires the selected BF16-capable RTX 3090")
        set_seed(config["seed"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True,
                                                  local_files_only=True, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.eos_token_id != 151643:
            raise ValueError("Expected the original Qwen3-4B-Base fast tokenizer with EOS151643")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
                                                   trust_remote_code=False, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa", low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, config["adapter_path"], is_trainable=False)
        model.to("cuda:0").eval()
        model.config.use_cache = True
        model.generation_config = GenerationConfig(
            do_sample=False, use_cache=True, eos_token_id=151643, pad_token_id=151643,
            bos_token_id=tokenizer.bos_token_id,
        )
        if config["max_context_tokens"] > model.config.max_position_embeddings:
            raise ValueError("Requested context limit exceeds the Base model's native context window")
        stops = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings=list(STOPS))])
        runtime = {name: importlib.metadata.version(name) for name in
                   ("torch", "transformers", "peft", "tokenizers")}
        record = read_json(manifest_path)
        if record.get("runtime_versions") not in (None, runtime):
            raise ValueError("Worker runtime changed; use a fresh evaluation output directory")
        record.update({"runtime_versions": runtime, "last_started_at": now(),
                       "model_class": type(base).__name__, "adapter_class": type(model).__name__,
                       "adapter_enabled": True, "active_adapters": model.active_adapters,
                       "selected": len(examples), "already_completed": len(completed)})
        write_json(manifest_path, record)
        for offset in range(0, len(pending), config["batch_size"]):
            rollout_chunk(pending[offset:offset + config["batch_size"]], args.worker_index, binding_sha,
                          model, tokenizer, stops, config, torch, predictions)
        record = read_json(manifest_path)
        record.update({"complete": True, "finished_at": now(), "terminal_records": len(examples)})
        write_json(manifest_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, choices=range(4), required=True)
    parser.add_argument("--num-workers", type=int, choices=(4,), default=4)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f"Evaluation worker stopped: {error}\n")


if __name__ == "__main__":
    main()
