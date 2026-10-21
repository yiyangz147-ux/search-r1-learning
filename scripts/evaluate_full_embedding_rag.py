#!/usr/bin/env python3
"""Resume a bound, independent GPU shard of full-embedding LoRA RAG evaluation.

Generation is batched; each question keeps its own search, token, invalid-action
and turn budgets. The protocol and rewards match ``sft_reward_monitor``. A
retriever/model infrastructure exception fails the worker without manufacturing
an incorrect answer. Completed records are fsynced and can be resumed.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from sft_reward_monitor import (
    INVALID_FEEDBACK, SEARCH_LIMIT_FEEDBACK, parse_action, retrieve,
    score_trajectory, valid_step_format, validate_config,
)
from search_sft_protocol import build_prompt, canonical_assistant, canonical_information


TERMINAL_STATUSES = frozenset({"ok", "turn_limit", "context_limit",
                               "invalid_action_limit", "generated_token_limit"})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def now():
    return datetime.now(timezone.utc).isoformat()


def binding_digest(manifest):
    return hashlib.sha256(canonical(manifest).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_example(example, index):
    result = dict(example)
    result["sample_id"] = str(example.get("sample_id", example.get("id", "")))
    result["golden_answers"] = example.get("golden_answers", example.get("gold_answers"))
    result["sample_index"] = index
    if (not result["sample_id"] or not isinstance(result.get("question"), str)
            or not result["question"].strip() or not isinstance(result.get("data_source"), str)
            or not isinstance(result["golden_answers"], list) or not result["golden_answers"]
            or any(not isinstance(answer, str) for answer in result["golden_answers"])):
        raise ValueError(f"Malformed evaluation example at index {index}")
    return result


@dataclass
class Trajectory:
    example: dict
    context: str = ""
    initial_prompt: str = ""
    events: list = field(default_factory=list)
    searches: int = 0
    generated: int = 0
    invalid: int = 0
    turn: int = 0
    all_steps_valid: bool = True
    started: float = field(default_factory=time.monotonic)
    diagnostics: dict = field(default_factory=lambda: {
        "action_stop": 0, "eos": 0, "new_token_limit": 0,
        "context_limited_steps": 0, "context_limit": 0,
    })

    def __post_init__(self):
        self.context = self.initial_prompt = build_prompt(self.example["question"])

    def finish(self, status, answer=""):
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Unrecognized terminal status: {status}")
        score = score_trajectory(answer, self.example["golden_answers"],
                                 self.example["data_source"], completed=status == "ok",
                                 all_steps_valid=self.all_steps_valid)
        return {
            "id": self.example["sample_id"], "sample_id": self.example["sample_id"],
            "sample_index": self.example["sample_index"],
            "question": self.example["question"], "data_source": self.example["data_source"],
            "golden_answers": self.example["golden_answers"],
            "initial_prompt": self.initial_prompt, "events": self.events,
            "status": status, "answer": answer, "search_count": self.searches,
            "generated_tokens": self.generated, "model_calls": self.turn,
            "elapsed_seconds": time.monotonic() - self.started,
            "generation_diagnostics": self.diagnostics, "completed_at": now(), **score,
        }


def retrieve_many(queries, config):
    """Use the local service's native query batch without changing passage text.

    As in the training monitor, local retrieval bypasses environment proxies,
    transient transport/server errors are retried, and empty/corrupted service
    responses fail evaluation instead of producing an incorrect-answer reward.
    """
    if not queries:
        return []
    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("Retriever queries must be nonempty strings")
    config = validate_config(config)
    payload = {"queries": list(queries), "topk": config["topk"], "return_scores": True}
    opener = build_opener(ProxyHandler({}))
    raw = None
    for attempt in range(config["retrieval_retries"] + 1):
        request = Request(config["retriever_url"], data=canonical(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with opener.open(request, timeout=config["retrieval_timeout_seconds"]) as response:
                raw = json.load(response)
            break
        except HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"Evaluation retriever HTTP {error.code}") from None
            if attempt == config["retrieval_retries"]:
                raise RuntimeError(f"Evaluation retriever HTTP {error.code}, retries exhausted") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            if attempt == config["retrieval_retries"]:
                raise RuntimeError("Evaluation retriever transport/JSON failure, retries exhausted") from None
        time.sleep(min(2 ** attempt, 8))
    batches = raw.get("result") if isinstance(raw, dict) else None
    if not isinstance(batches, list) or len(batches) != len(queries):
        raise RuntimeError("Evaluation retriever returned an invalid result batch")
    results = []
    for batch_index, (query, hits) in enumerate(zip(queries, batches)):
        if not isinstance(hits, list) or not hits:
            raise RuntimeError("Evaluation retriever returned an invalid/empty result batch")
        documents, lines = [], []
        for index, hit in enumerate(hits[:config["topk"]], 1):
            if not isinstance(hit, dict):
                raise RuntimeError("Evaluation retriever returned a non-object passage")
            document = hit.get("document", hit)
            content = document.get("contents") if isinstance(document, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Evaluation retrieved passage lacks contents")
            shown = content[:config["passage_chars"]]
            documents.append({"id": str(document.get("id", "")), "contents": shown,
                              "score": hit.get("score")})
            lines.append(f"Doc {index}: {shown}")
        information = "<information>" + html.escape("\n".join(lines), quote=False) + "</information>"
        results.append({"query": query, "content": information, "documents": documents,
                        "request": payload, "batch_index": batch_index,
                        "transport_attempts": attempt + 1})
    return results


def batched_rollouts(examples, generate_many, retrieve_one, config, *, on_record=None,
                     retrieve_many=None):
    """Dynamically refill independent trajectories; injectable pure I/O for tests.

    ``generate_many(contexts, budgets)`` returns one packet per context, with
    the same packet fields as the training monitor. ``retrieve_one`` uses that
    monitor's unchanged service adapter. An optional ``retrieve_many(queries)``
    performs one native HTTP query batch per generation round instead. Gold
    aliases are accessed for scoring only after a trajectory terminates.
    """
    config = validate_config(config)
    batch_size = config.get("batch_size", 8)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    waiting = deque(examples)
    active, records = [], []

    def finish(state, status, answer=""):
        record = state.finish(status, answer)
        if on_record is not None:
            on_record(record)
        records.append(record)

    while active or waiting:
        while waiting and len(active) < batch_size:
            active.append(Trajectory(waiting.popleft()))
        ready = []
        for state in active:
            if state.generated >= config["max_total_generated_tokens"]:
                finish(state, "generated_token_limit")
            else:
                ready.append(state)
        if not ready:
            active = []
            continue
        budgets = [min(config["max_new_tokens"],
                       config["max_total_generated_tokens"] - state.generated) for state in ready]
        packets = generate_many([state.context for state in ready], budgets)
        if not isinstance(packets, list) or len(packets) != len(ready):
            raise RuntimeError("Generator did not return exactly one packet per trajectory")
        active, continuations, search_slots, search_queries = [], [], [], []
        for state, packet, budget in zip(ready, packets, budgets):
            if packet.get("context_limit"):
                state.diagnostics["context_limit"] += 1
                finish(state, "context_limit")
                continue
            count = packet.get("generated_tokens")
            if (not isinstance(packet.get("text"), str) or type(count) is not int
                    or count < 1 or count > budget
                    or type(packet.get("input_tokens")) is not int
                    or packet["input_tokens"] < 1):
                raise RuntimeError("Generator returned a malformed or over-budget packet")
            text = packet["text"]
            state.generated += count
            stop_reason = packet.get("stop_reason")
            if stop_reason in state.diagnostics:
                state.diagnostics[stop_reason] += 1
            if packet.get("context_limited"):
                state.diagnostics["context_limited_steps"] += 1
            action, valid = parse_action(text), valid_step_format(text)
            state.all_steps_valid = state.all_steps_valid and valid
            event = {"role": "assistant", "content": text, "turn": state.turn,
                     "generated_tokens": count, "input_tokens": packet["input_tokens"],
                     "format_valid": valid, "action": action,
                     "stop_reason": stop_reason, "context_limited": bool(packet.get("context_limited"))}
            state.events.append(event)
            state.turn += 1
            state.context += canonical_assistant(text) if text.strip() else text
            if action is None:
                state.invalid += 1
                if state.invalid >= config["max_invalid_actions"]:
                    finish(state, "invalid_action_limit")
                    continue
                observation = {"role": "environment", "content": canonical_information(INVALID_FEEDBACK),
                               "kind": "invalid_action_feedback"}
            elif action["action"] == "answer":
                finish(state, "ok", action["value"])
                continue
            elif state.searches >= config["max_searches"]:
                state.all_steps_valid = False
                observation = {"role": "environment", "content": canonical_information(SEARCH_LIMIT_FEEDBACK),
                               "kind": "search_limit_feedback"}
            else:
                search_slots.append(len(continuations))
                search_queries.append(action["value"])
                observation = None
            continuations.append((state, observation))
        if search_queries:
            retrieved_batch = (retrieve_many(search_queries) if retrieve_many is not None
                               else [retrieve_one(query) for query in search_queries])
            if not isinstance(retrieved_batch, list) or len(retrieved_batch) != len(search_queries):
                raise RuntimeError("Retriever did not return exactly one result per search query")
            for slot, retrieved in zip(search_slots, retrieved_batch):
                state, _ = continuations[slot]
                state.searches += 1
                observation = {"role": "environment", "content": canonical_information(retrieved["content"]),
                               "kind": "retrieval", "retrieval": retrieved}
                continuations[slot] = (state, observation)
        for state, observation in continuations:
            state.events.append(observation)
            state.context += observation["content"]
            if state.turn >= config["max_turns"]:
                finish(state, "turn_limit")
            else:
                active.append(state)
    return records


def validate_record(record, examples_by_index, worker_index, num_workers, run_binding):
    index = record.get("sample_index")
    if type(index) is not int or index not in examples_by_index or index % num_workers != worker_index:
        raise ValueError("Progress record belongs to a different example or shard")
    example = examples_by_index[index]
    if (record.get("run_binding_sha256") != run_binding or record.get("worker_index") != worker_index
            or record.get("sample_id") != example["sample_id"] or record.get("id") != example["sample_id"]
            or record.get("question") != example["question"]
            or record.get("golden_answers") != example["golden_answers"]
            or record.get("data_source") != example["data_source"]
            or record.get("status") not in TERMINAL_STATUSES
            or not isinstance(record.get("answer"), str) or not isinstance(record.get("events"), list)):
        raise ValueError("Progress record has incompatible data, binding, or terminal status")
    assistant = [event for event in record["events"] if event.get("role") == "assistant"]
    valid = all(valid_step_format(event.get("content", "")) for event in assistant)
    # An over-budget search is syntactically valid but invalidates the trajectory.
    valid = valid and not any(event.get("kind") == "search_limit_feedback" for event in record["events"])
    expected = score_trajectory(record["answer"], example["golden_answers"], example["data_source"],
                                completed=record["status"] == "ok", all_steps_valid=valid)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Progress record has inconsistent reward {key}")
    if record["status"] == "ok":
        action = parse_action(assistant[-1]["content"]) if assistant else None
        if action is None or action["action"] != "answer" or action["value"] != record["answer"]:
            raise ValueError("Completed progress record lacks its claimed final answer")
    elif record["answer"]:
        raise ValueError("Unfinished progress record cannot contain a final answer")
    if (record.get("model_calls") != len(assistant)
            or record.get("generated_tokens") != sum(event["generated_tokens"] for event in assistant)
            or record.get("search_count") != sum(event.get("kind") == "retrieval" for event in record["events"])):
        raise ValueError("Progress counters are inconsistent with saved events")
    return index


def load_progress(path, examples_by_index, worker_index, num_workers, run_binding):
    """Repair only an incomplete final write; reject corruption/duplicate results."""
    path = Path(path)
    if not path.exists():
        return {}
    completed = {}
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            terminated = line.endswith(b"\n")
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if terminated:
                    raise ValueError("Malformed complete progress line; refusing to erase it") from None
                stream.seek(offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            index = validate_record(record, examples_by_index, worker_index, num_workers, run_binding)
            if index in completed:
                raise ValueError("Duplicate sample in progress journal")
            completed[index] = record
            if not terminated:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                break
    return completed


class Journal:
    def __init__(self, path, completed, worker_index, run_binding, total):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed = completed
        self.worker_index, self.run_binding, self.total = worker_index, run_binding, total
        self.stream = self.path.open("a", encoding="utf-8")

    def append(self, record):
        index = record["sample_index"]
        if index in self.completed:
            raise ValueError("Refusing to append a duplicate completed question")
        record = {**record, "worker_index": self.worker_index, "run_binding_sha256": self.run_binding}
        self.stream.write(canonical(record) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.completed[index] = record
        print(canonical({"event": "full_embedding_eval_progress", "worker": self.worker_index,
                         "completed": len(self.completed), "total": self.total,
                         "sample_id": record["sample_id"], "status": record["status"],
                         "total_reward": record["total_reward"], "format_reward": record["format_reward"],
                         "searches": record["search_count"], "tokens": record["generated_tokens"],
                         "elapsed_seconds": round(record["elapsed_seconds"], 2)}), flush=True)

    def close(self):
        self.stream.close()


class BatchedGenerator:
    """Left-pad only equal per-row generation budgets, with exact row stop lengths.

    A stopping criterion records each row's first terminal length. This keeps
    padding after a stopped row out of its trace and generated-token budget.
    The longest left-padded prompt plus its shared generation budget must fit
    the context limit. OOM retries split only the current inference microbatch.
    """

    def __init__(self, model, tokenizer, config):
        import torch
        from transformers import GenerationConfig, StopStringCriteria, StoppingCriteriaList
        self.torch = torch
        self.StoppingCriteriaList = StoppingCriteriaList
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.device = model.get_input_embeddings().weight.device
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Evaluation tokenizer needs an EOS token")
        tokenizer.pad_token_id = self.eos
        tokenizer.padding_side = "left"
        self.generation_config = GenerationConfig(
            do_sample=False, use_cache=True, eos_token_id=self.eos,
            pad_token_id=self.eos, bos_token_id=tokenizer.bos_token_id)
        self.stop_strings = StopStringCriteria(tokenizer=tokenizer, stop_strings=["</search>", "</answer>"])

    def __call__(self, contexts, budgets):
        microbatch_size = self.config.get("generation_microbatch_size", 8)
        if type(microbatch_size) is not int or microbatch_size < 1:
            raise ValueError("generation_microbatch_size must be a positive integer")
        encoded = self.tokenizer(contexts, add_special_tokens=False, padding=False)
        groups, packets = {}, [None] * len(contexts)
        for index, (ids, budget) in enumerate(zip(encoded["input_ids"], budgets)):
            available = self.config["max_context_tokens"] - len(ids)
            if available <= 0:
                packets[index] = {"context_limit": True}
            else:
                effective = min(budget, available)
                groups.setdefault(effective, []).append((index, ids, effective < budget))
        for budget, rows in groups.items():
            # The active trajectory pool may be larger than a model batch. Keep
            # inference memory bounded while collecting all search actions into
            # one native retrieval batch after this generation round completes.
            for start in range(0, len(rows), microbatch_size):
                for index, packet in self._generate(rows[start:start + microbatch_size], budget):
                    packets[index] = packet
        return packets

    def _generate(self, rows, budget):
        try:
            return self._generate_once(rows, budget)
        except self.torch.cuda.OutOfMemoryError:
            if len(rows) == 1:
                raise
        # Leave the exception handler before retrying: its traceback otherwise
        # retains the failed generate() frame and its large KV-cache tensors.
        gc.collect()
        self.torch.cuda.empty_cache()
        middle = len(rows) // 2
        print(canonical({"event": "full_embedding_eval_oom_split", "batch": len(rows),
                         "left": middle, "right": len(rows) - middle}), flush=True)
        return self._generate(rows[:middle], budget) + self._generate(rows[middle:], budget)

    def _generate_once(self, rows, budget):
        torch, eos, stop_strings = self.torch, self.eos, self.stop_strings
        encoded = self.tokenizer.pad({"input_ids": [row[1] for row in rows]},
                                     padding=True, return_attention_mask=True, return_tensors="pt").to(self.device)
        width = encoded["input_ids"].shape[1]
        if width + budget > self.config["max_context_tokens"]:
            raise RuntimeError("Padded generation batch exceeds its context budget")
        lengths, reasons = [None] * len(rows), [None] * len(rows)

        class TrackRowStops:
            def __call__(self, input_ids, scores, **kwargs):
                stop = stop_strings(input_ids, scores, **kwargs)
                eos_hit = input_ids[:, -1] == eos
                values = (stop | eos_hit).detach().cpu().tolist()
                eos_values = eos_hit.detach().cpu().tolist()
                for index, hit in enumerate(values):
                    if hit and lengths[index] is None:
                        lengths[index] = input_ids.shape[1] - width
                        reasons[index] = "eos" if eos_values[index] else "action_stop"
                return torch.tensor([value is not None for value in lengths], dtype=torch.bool,
                                    device=input_ids.device)

        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with torch.inference_mode(), autocast:
            output = self.model.generate(
                **encoded, generation_config=self.generation_config, max_new_tokens=budget,
                stopping_criteria=self.StoppingCriteriaList([TrackRowStops()]), use_cache=True,
                return_dict_in_generate=False, output_scores=False, synced_gpus=False, logits_to_keep=1)
        output = output[:, width:].detach().cpu().tolist()
        result = []
        for row, ids, length, reason in zip(rows, output, lengths, reasons):
            ids = ids[:length] if length is not None else ids
            text_ids = ids[:-1] if ids and ids[-1] == eos else ids
            result.append((row[0], {
                "text": self.tokenizer.decode(text_ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False),
                "generated_tokens": len(ids), "input_tokens": len(row[1]),
                "stop_reason": reason or "new_token_limit", "context_limited": row[2],
            }))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()
    from prepare_full_embedding_eval import verify_run

    config, manifest, source_examples = verify_run(args.config)
    if args.num_workers != config["num_workers"] or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("Worker rank/count differ from the bound run")
    selected = config["selected_gpu_uuids"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != selected[args.worker_index]:
        raise ValueError("Each independent worker must see exactly its configured GPU UUID")
    run_binding = binding_digest(manifest)
    examples = [normalize_example(row, index) for index, row in enumerate(source_examples)]
    examples_by_index = {row["sample_index"]: row for row in examples}
    shard = [row for row in examples if row["sample_index"] % args.num_workers == args.worker_index]
    directory = Path(config["output_dir"]) / "workers"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"worker-{args.worker_index:02d}.jsonl"
    status_path = directory / f"worker-{args.worker_index:02d}.status.json"
    lock = directory / f"worker-{args.worker_index:02d}.lock"
    # flock is released by the OS on process exit, including SIGKILL. A stale
    # path cannot block a legitimate resume; a live duplicate process does.
    import fcntl
    with lock.open("a") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process already owns this evaluation worker") from None
        completed = load_progress(path, examples_by_index, args.worker_index, args.num_workers, run_binding)
        pending = [row for row in shard if row["sample_index"] not in completed]
        started = time.monotonic()
        status = {"worker_index": args.worker_index, "pid": os.getpid(), "status": "running",
                  "run_binding_sha256": run_binding, "started_at": now(),
                  "total": len(shard), "resumed_completed": len(completed)}
        atomic_json(status_path, status)
        journal = Journal(path, completed, args.worker_index, run_binding, len(shard))
        try:
            if pending:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
                from full_embedding_lora import assert_full_embedding_tied, load_full_embedding_adapter
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise RuntimeError("Worker requires exactly one visible CUDA device")
                set_seed(config["seed"])
                torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
                torch.backends.cuda.matmul.allow_tf32 = True
                tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True,
                                                            local_files_only=True)
                base = AutoModelForCausalLM.from_pretrained(
                    config["model_path"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                    low_cpu_mem_usage=True, device_map={"": 0}, local_files_only=True)
                model = load_full_embedding_adapter(base, config["adapter_path"], is_trainable=False)
                model.requires_grad_(False)
                model.eval()
                model.config.use_cache = True
                assert_full_embedding_tied(model, require_trainable=False)
                print(canonical({"event": "full_embedding_eval_model_loaded", "worker": args.worker_index,
                                 "pending": len(pending), "gpu": selected[args.worker_index],
                                 "batch_size": config["batch_size"], "full_embedding_tied": True}), flush=True)
                batched_rollouts(pending, BatchedGenerator(model, tokenizer, config),
                                 lambda query: retrieve(query, config), config, on_record=journal.append,
                                 retrieve_many=lambda queries: retrieve_many(queries, config))
            status.update(status="complete", completed=len(completed), completed_at=now(),
                          elapsed_seconds=time.monotonic() - started)
            atomic_json(status_path, status)
        except BaseException as error:
            status.update(status="failed", completed=len(completed), failed_at=now(),
                          error_type=type(error).__name__, error=str(error),
                          elapsed_seconds=time.monotonic() - started)
            atomic_json(status_path, status)
            raise
        finally:
            journal.close()


if __name__ == "__main__":
    main()
