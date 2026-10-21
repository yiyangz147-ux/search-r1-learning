"""Auditable GRPO metrics for the pinned, patched verl trainer.

No torch/verl imports are needed here.  Metrics are independently recomputed
from role-separated assistant actions; decoded mixed-role text is never scored.
The raw records stay on disk, while only numeric aggregates go to W&B.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean

from search_grpo_reward import TERMINAL_STATUSES, gold_aliases, score_record


SCHEMA_VERSION = 1



def chunked_logprobs_entropy(logits, labels, *, calculate_entropy=False, chunk_size=128,
                            checkpoint_chunks=True, logprobs_fn=None, entropy_fn=None):
    """Bound padded actor vocabulary temporaries to a token chunk.

    Preserve the caller's native LP/entropy kernels and autocast context. Split
    each batch row explicitly: reshaping a sliced B>1 tensor could copy the
    entire vocabulary matrix. Per-chunk nonreentrant checkpointing recomputes
    only deterministic vocabulary operations during backward, never the model.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if logits.ndim not in (2, 3) or tuple(logits.shape[:-1]) != tuple(labels.shape):
        raise ValueError("Expected [tokens,vocab] or [batch,tokens,vocab] with aligned labels")
    if logits.shape[-2] == 0:
        raise ValueError("Empty token dimension")
    if logprobs_fn is None or (calculate_entropy and entropy_fn is None):
        # Production passes both actual pinned verl functions. This fallback
        # also makes small CPU contracts independent of the RL installation.
        def fallback_logprobs(value, target, inplace_backward=False):
            if value.dtype in (torch.float32, torch.float64):
                return value.gather(-1, target.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(value, dim=-1)
            return torch.nn.functional.log_softmax(value, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        def fallback_entropy(value):
            probabilities = torch.nn.functional.softmax(value, dim=-1)
            return torch.logsumexp(value, dim=-1) - torch.sum(probabilities * value, dim=-1)
        logprobs_fn = logprobs_fn or fallback_logprobs
        entropy_fn = entropy_fn or fallback_entropy

    def statistics(value, target):
        # Keep a batch dimension so native BF16 fallback performs one matrix
        # log-softmax, rather than one Python/kernel invocation per token.
        value = value.unsqueeze(0)
        target = target.unsqueeze(0)
        # Native Flash-CE may otherwise overwrite a shared logits view while
        # entropy checkpoint recomputation still needs its original values.
        log_probs = logprobs_fn(value, target, inplace_backward=False).squeeze(0)
        if calculate_entropy:
            return log_probs, entropy_fn(value).squeeze(0)
        return log_probs

    squeeze = logits.ndim == 2
    batch_logits = logits.unsqueeze(0) if squeeze else logits
    batch_labels = labels.unsqueeze(0) if squeeze else labels
    all_log_probs, all_entropy = [], []
    for row, target in zip(batch_logits.unbind(0), batch_labels.unbind(0), strict=True):
        row_log_probs, row_entropy = [], []
        for value, target_chunk in zip(row.split(chunk_size, dim=0), target.split(chunk_size, dim=0), strict=True):
            if checkpoint_chunks and torch.is_grad_enabled() and value.requires_grad:
                result = checkpoint(statistics, value, target_chunk, use_reentrant=False, preserve_rng_state=False)
            else:
                result = statistics(value, target_chunk)
            if calculate_entropy:
                log_probs, entropy = result
                row_entropy.append(entropy)
            else:
                log_probs = result
            row_log_probs.append(log_probs)
        all_log_probs.append(torch.cat(row_log_probs, dim=0))
        if calculate_entropy:
            all_entropy.append(torch.cat(row_entropy, dim=0))
    log_probs = torch.stack(all_log_probs, dim=0)
    entropy = torch.stack(all_entropy, dim=0) if calculate_entropy else None
    if squeeze:
        log_probs = log_probs.squeeze(0)
        entropy = entropy.squeeze(0) if entropy is not None else None
    return entropy, log_probs


def trim_micro_batch_padding(micro_batch):
    """Optional exact-context trim for a text-only single-sequence forward.

    Keep all observation tokens and the existing position IDs.  The caller
    restores the original response width before any masked RL losses run.
    Returning None leaves unsupported layouts on the unmodified native path.
    """
    if "multi_modal_inputs" in micro_batch:
        return None
    input_ids = micro_batch["input_ids"]
    attention_mask = micro_batch["attention_mask"]
    position_ids = micro_batch["position_ids"]
    responses = micro_batch["responses"]
    if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or position_ids.ndim != 2
            or tuple(position_ids.shape) != tuple(input_ids.shape)
            or tuple(attention_mask.shape) != tuple(input_ids.shape)):
        return None
    original_response = int(responses.shape[-1])
    original_prompt = int(input_ids.shape[-1]) - original_response
    valid = attention_mask[0].nonzero(as_tuple=False).flatten()
    if not valid.numel():
        return None
    start, end = int(valid[0].item()), int(valid[-1].item()) + 1
    # This transformation removes only edge padding, never holes or tool state.
    if (start == 0 and end == input_ids.shape[-1]) or not start < original_prompt < end:
        return None
    if not bool((attention_mask[0, start:end] == 1).all().item()):
        return None
    response_length = end - original_prompt
    trimmed = dict(micro_batch)
    for key in ("input_ids", "attention_mask", "position_ids"):
        trimmed[key] = micro_batch[key][:, start:end]
    trimmed["responses"] = responses[:, :response_length]
    return trimmed, original_response


def restore_response_padding(entropy, log_probs, original_response):
    """Restore verl's response coordinates without detaching autograd."""
    from torch.nn import functional as F
    padding = int(original_response) - int(log_probs.shape[-1])
    if padding < 0:
        raise ValueError("Trimmed response exceeds the original storage width")
    log_probs = F.pad(log_probs, (0, padding))
    entropy = F.pad(entropy, (0, padding)) if entropy is not None else None
    return entropy, log_probs


def cast_model_parameters_preserving_buffers(model, dtype):
    """Finish HF weight casting without quantizing reconstructed FP32 RoPE.

    ``Module.to(dtype)`` also casts non-persistent rotary inv_freq buffers.  HF
    recreates those buffers as FP32 on reload; changing them after loading
    changes the model even when every checkpoint Parameter is unchanged.
    """
    before = {name: (str(value.dtype), id(value)) for name, value in model.named_buffers()}
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
    after = {name: (str(value.dtype), id(value)) for name, value in model.named_buffers()}
    if before != after:
        raise ValueError("Parameter-only dtype casting altered a non-parameter buffer")
    return model


def _tensor_sha256(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _canonical_model_name(name):
    return name.removeprefix("base_model.model.")


def audit_actor_parameters(model, config, *, rank=0, output_dir=None, tensor_hasher=None):
    """Measure the actual pre-FSDP actor and refuse an incorrect parameter set.

    The worker passes actor_rollout_ref's config subtree.  The launcher passes
    SEARCH_GRPO_OUTPUT_DIR because trainer.default_local_dir is not in that
    subtree.  No audit flags are copied from the export's declarative LoRA plan.
    """
    destination = output_dir or os.environ.get("SEARCH_GRPO_OUTPUT_DIR")
    if not destination:
        raise ValueError("SEARCH_GRPO_OUTPUT_DIR is required for actual actor parameter audit")
    output = Path(destination)
    rank = int(rank)
    model_path = Path(_get(config, "model.path"))
    manifest_path = model_path / "sft_export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError("SFT export manifest is incomplete")
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    tied = input_weight is output_weight
    frozen = not input_weight.requires_grad and not output_weight.requires_grad
    if not tied or not frozen:
        raise ValueError("Actor must keep the actual shared input/output embedding tied and frozen")
    if str(input_weight.dtype) != "torch.bfloat16":
        raise ValueError("Frozen merged SFT embedding must retain its exported BF16 dtype")
    if _get(config, "model.lora_adapter_path") is not None:
        raise ValueError("Expected a newly initialized GRPO adapter, not an existing adapter")
    if int(_get(config, "model.lora_rank")) != 32 or int(_get(config, "model.lora_alpha")) != 16:
        raise ValueError("GRPO config must use LoRA rank32 alpha16")
    peft_configs = getattr(model, "peft_config", {})
    if set(peft_configs) != {"default"}:
        raise ValueError("Actor must have exactly one newly created default GRPO adapter")
    peft = peft_configs["default"]
    measured_lora = {"r": int(_get(peft, "r")), "alpha": int(_get(peft, "lora_alpha")),
                     "dropout": float(_get(peft, "lora_dropout", 0)),
                     "target_modules": sorted(_get(peft, "target_modules")),
                     "bias": _get(peft, "bias")}
    if measured_lora != {"r": 32, "alpha": 16, "dropout": 0.0,
                         "target_modules": ["q_proj", "v_proj"], "bias": "none"}:
        raise ValueError(f"Actual PEFT adapter differs from the approved GRPO adapter: {measured_lora}")
    pattern = re.compile(r"(?:^|\.)(q_proj|v_proj)\.lora_([AB])\.default\.weight$")
    trainable, base_count, trainable_count, b_nonzero = [], 0, 0, 0
    parameter_dtypes = Counter()
    for name, parameter in model.named_parameters():
        size = int(parameter.numel())
        parameter_dtypes[str(parameter.dtype)] += size
        match = pattern.search(name)
        if parameter.requires_grad:
            if match is None:
                raise ValueError(f"Unexpected trainable actor Parameter: {name}")
            shape = list(parameter.shape)
            if len(shape) != 2 or shape[0 if match[2] == "A" else 1] != 32:
                raise ValueError(f"Actual LoRA rank is not32: {name} {shape}")
            entry = {"name": name, "shape": shape, "numel": size, "dtype": str(parameter.dtype)}
            trainable.append(entry)
            trainable_count += size
            if match[2] == "B":
                count = int(parameter.detach().count_nonzero().item())
                b_nonzero += count
                entry["initial_nonzero"] = count
        else:
            if match is not None:
                raise ValueError(f"Expected q/v LoRA Parameter was incorrectly frozen: {name}")
            base_count += size
    if trainable_count != 11796480 or not trainable:
        raise ValueError(f"Expected exactly11796480 trainable q/v LoRA parameters, got {trainable_count}")
    if not any(".lora_B." in row["name"] for row in trainable) or b_nonzero:
        raise ValueError("New GRPO LoRA B matrices must actually be zero before loading a training checkpoint")
    buffers = {_canonical_model_name(name): str(value.dtype) for name, value in model.named_buffers()}
    expected_buffers = manifest.get("non_parameter_buffer_dtypes")
    if not isinstance(expected_buffers, dict) or buffers != expected_buffers:
        raise ValueError(f"Actor buffer dtypes differ from the merged SFT export: {buffers} vs {expected_buffers}")
    rotary = {name: dtype for name, dtype in buffers.items() if "inv_freq" in name}
    if not rotary or any(dtype != "torch.float32" for dtype in rotary.values()):
        raise ValueError("Actual Qwen3 rotary inv_freq buffers must remain FP32")
    expected_hash = manifest.get("shared_embedding", {}).get("tensor_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("SFT export is missing its full embedding tensor SHA256")
    embedding_hash = (tensor_hasher or _tensor_sha256)(input_weight) if rank == 0 else None
    if rank == 0 and embedding_hash != expected_hash:
        raise ValueError("Actual actor input embedding tensor SHA256 differs from the complete SFT export")
    result = {"schema_version": SCHEMA_VERSION, "rank": rank,
              "measurement_stage": "after_new_PEFT_before_FSDP_and_checkpoint_resume",
              "embedding_frozen": frozen, "only_qv_lora_trainable": True,
              "base_all_frozen": True, "shared_embedding_tied": tied,
              "trainable_count": trainable_count, "base_parameter_count": base_count,
              "trainable_parameters": trainable, "parameter_count_by_dtype": dict(parameter_dtypes),
              "lora": measured_lora, "initial_lora_b_nonzero": b_nonzero,
              "embedding_shape": list(input_weight.shape), "embedding_dtype": str(input_weight.dtype),
              "embedding_tensor_sha256": embedding_hash, "expected_embedding_tensor_sha256": expected_hash,
              "embedding_hash_verified_on_this_rank": rank == 0,
              "non_parameter_buffer_dtypes": buffers, "rotary_buffer_dtypes": rotary,
              "sft_export_manifest": str(manifest_path),
              "sft_export_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    _atomic_json(output / f"runtime_parameter_audit_rank{rank}.json", result)
    if rank == 0:
        _atomic_json(output / "runtime_parameter_audit.json", result)
    return result


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return _plain(value.item())
    return value


def _get(config, dotted, default=None):
    value = config
    for key in dotted.split("."):
        if hasattr(value, "get"):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
        if value is default:
            break
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(path, (_json(value) + "\n").encode())


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(_json(value) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_audit(directory, split, step, rows):
    """Keep resume attempts instead of silently replacing an earlier audit."""
    directory = Path(directory)
    path = directory / f"{split}-step{step:06d}.jsonl"
    content = "".join(_json(row) + "\n" for row in rows).encode()
    attempt = 1
    while path.exists() and path.read_bytes() != content:
        attempt += 1
        path = directory / f"{split}-step{step:06d}-attempt{attempt:03d}.jsonl"
    if not path.exists():
        _atomic_bytes(path, content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "count": len(rows)}


def _finite_number(value, label):
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} is not a finite number: {value!r}")
    return float(value)


def _same_number(actual, expected, label):
    if abs(_finite_number(actual, label) - _finite_number(expected, label)) > 1e-6:
        raise ValueError(f"{label} disagrees with independently recomputed reward")


def _verify_response_mask(record, mask):
    mask = _plain(mask)
    if not isinstance(mask, list) or any(value not in (0, 1) for value in mask):
        raise ValueError("response_mask must be a binary token vector")
    expected = [0] * len(mask)
    cursor = 0
    generated = 0
    for event in record["events"]:
        if event["role"] == "environment" and not event.get("included_in_context", True):
            continue
        start, end = event.get("response_token_start"), event.get("response_token_end")
        if (not isinstance(start, int) or not isinstance(end, int)
                or start != cursor or not start < end <= len(mask)):
            raise ValueError("Trajectory token spans are missing, discontinuous or out of bounds")
        if event["role"] == "assistant":
            if event.get("generated_tokens") != end - start:
                raise ValueError("Assistant token span disagrees with its generated token count")
            expected[start:end] = [1] * (end - start)
            generated += end - start
        cursor = end
    if expected != mask:
        raise ValueError("response_mask supervises observation/padding or omits assistant tokens")
    if record.get("generated_tokens") != generated:
        raise ValueError("Trajectory generated token count disagrees with assistant spans")


def audit_batch(batch, score_values, *, step, split):
    """Return raw auditable records, checking IDs, masks and true EM against verl."""
    scores = _plain(score_values)
    if not isinstance(scores, list):
        raise ValueError("Outcome scores must be a vector")
    non_tensor = batch.non_tensor_batch
    required = ("search_grpo_trajectory", "data_source", "uid", "reward_model", "extra_info")
    for key in required:
        if key not in non_tensor or len(non_tensor[key]) != len(scores):
            raise ValueError(f"Missing/misaligned batch metadata: {key}")
    if "response_mask" not in batch.batch or len(batch.batch["response_mask"]) != len(scores):
        raise ValueError("Missing/misaligned generated-token response mask")
    rows = []
    for index, claimed_score in enumerate(scores):
        record = _plain(non_tensor["search_grpo_trajectory"][index])
        info = _plain(non_tensor["extra_info"][index])
        reward_model = _plain(non_tensor["reward_model"][index])
        source = str(non_tensor["data_source"][index])
        uid = str(non_tensor["uid"][index])
        sample_id = info.get("sample_id")
        if not sample_id or record.get("sample_id") != sample_id or not uid:
            raise ValueError("Trusted trajectory sample_id/group uid does not match dataset metadata")
        if source not in ("nq", "hotpotqa") or record.get("data_source") != source:
            raise ValueError("Trajectory source does not match selected NQ/HotpotQA question")
        if record.get("question") != info.get("question"):
            raise ValueError("Trajectory question does not match selected dataset question")
        gold = gold_aliases(reward_model.get("ground_truth"))
        recomputed = score_record(record, gold, source)
        _same_number(claimed_score, recomputed["score"], "verl outcome score")
        for key, value in recomputed.items():
            if key in record:
                _same_number(record[key], value, f"trajectory.{key}")
            if key in non_tensor:
                _same_number(_plain(non_tensor[key][index]), value, f"batch.{key}")
        _verify_response_mask(record, batch.batch["response_mask"][index])
        rows.append({**record, **recomputed, "golden_answers": gold, "group_uid": uid,
                     "global_step": int(step), "split": split, "schema_version": SCHEMA_VERSION})
    return rows


def _aggregate(rows, prefix):
    if not rows:
        raise ValueError("Cannot report an empty Search-R1 evaluation")
    metrics = {
        f"{prefix}/answer_em": mean(row["score"] for row in rows),
        f"{prefix}/total_reward": mean(row["score"] for row in rows),
        f"{prefix}/format_reward": mean(row["format_reward"] for row in rows),
        f"{prefix}/trajectory_count": len(rows),
    }
    if prefix == "train":
        metrics["train/answer_reward"] = metrics["train/answer_em"]
    for key in ("search_count", "invalid_action_count", "model_calls", "generated_tokens"):
        metrics[f"{prefix}/{key}_mean"] = mean(row[key] for row in rows)
    for source in sorted({row["data_source"] for row in rows}):
        selected = [row for row in rows if row["data_source"] == source]
        em = mean(row["score"] for row in selected)
        metrics[f"{prefix}/{source}_answer_em"] = em
        metrics[f"{prefix}/{source}_count"] = len(selected)
        if prefix == "train":
            metrics[f"train/{source}_answer_reward"] = em
    status_counts = Counter(row["status"] for row in rows)
    for status in sorted(TERMINAL_STATUSES):
        metrics[f"{prefix}/status_{status}_fraction"] = status_counts[status] / len(rows)
    return metrics


def _expected_val_ids(trainer):
    rows = getattr(trainer.val_dataset, "rows", None)
    if rows is None:
        raise ValueError("SearchR1BaseDataset.rows required for complete validation ID coverage")
    ids = [row["extra_info"]["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Validation dataset contains duplicate sample IDs")
    return set(ids)


def _checkpoint_candidate(trainer, metrics, audit):
    step = int(trainer.global_steps)
    output = Path(_get(trainer.config, "trainer.default_local_dir"))
    initial = step == 0
    actor_path = (Path(_get(trainer.config, "actor_rollout_ref.model.path")) if initial
                  else output / f"global_step_{step}" / "actor")
    return {"schema_version": SCHEMA_VERSION, "global_step": step,
            "checkpoint_kind": "merged_sft_base_zero_grpo_increment" if initial else "verl_fsdp_actor",
            "actor_checkpoint": str(actor_path), "checkpoint_ready_at_validation": actor_path.exists(),
            "val_answer_em": metrics["val/answer_em"], "val_count": audit["count"],
            "selection_metric": "val/answer_em", "selection_direction": "max",
            "tie_break": "earlier_global_step", "validation_audit": audit, "metrics": metrics}


def record_validation(trainer, rows):
    """Validate full dev coverage, save raw trajectories and update EM-only selection."""
    if int(_get(trainer.config, "actor_rollout_ref.rollout.val_kwargs.n", 1)) != 1:
        raise ValueError("This experiment requires exactly one greedy validation trajectory per question")
    actual = [row["sample_id"] for row in rows]
    expected = _expected_val_ids(trainer)
    if len(actual) != len(trainer.val_dataset) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Validation did not cover every selected dev question exactly once")
    output = Path(_get(trainer.config, "trainer.default_local_dir"))
    audit = _write_audit(output / "audit", "validation", int(trainer.global_steps), rows)
    metrics = _aggregate(rows, "val")
    candidate = _checkpoint_candidate(trainer, metrics, audit)
    best_file = output / "best_checkpoint.json"
    if best_file.exists():
        best = json.loads(best_file.read_text())
        if best.get("selection_metric") != "val/answer_em" or best.get("val_count") != len(rows):
            raise ValueError("Existing best checkpoint uses a different metric or validation selection")
    else:
        best = None
    if (best is None or candidate["val_answer_em"] > best["val_answer_em"]
            or (candidate["val_answer_em"] == best["val_answer_em"]
                and candidate["global_step"] < best["global_step"])):
        _atomic_json(best_file, candidate)
        best = candidate
    if int(trainer.global_steps) == 0 and not (output / "last_checkpoint.json").exists():
        _atomic_json(output / "last_checkpoint.json", candidate)
    metrics["val/best_answer_em"] = best["val_answer_em"]
    metrics["val/best_global_step"] = best["global_step"]
    _append_jsonl(output / "validation_history.jsonl", {
        "global_step": int(trainer.global_steps), "metrics": metrics, "audit": audit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    return metrics


def record_training(trainer, batch, native_metrics):
    """Audit all five candidates per prompt and provide numeric W&B aliases."""
    scores = batch.batch["token_level_scores"].sum(-1)
    rows = audit_batch(batch, scores, step=trainer.global_steps, split="train")
    group_size = int(_get(trainer.config, "actor_rollout_ref.rollout.n"))
    prompt_count = int(_get(trainer.config, "data.train_batch_size"))
    if group_size != 5 or len(rows) != prompt_count * group_size:
        raise ValueError("Training batch must contain the configured number of five-trajectory groups")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_uid"]].append(row)
    if len(groups) != prompt_count or any(len(group) != group_size for group in groups.values()):
        raise ValueError("Missing/duplicate GRPO group members after distributed batch balancing")
    for group in groups.values():
        if len({(row["sample_id"], row["data_source"], row["question"]) for row in group}) != 1:
            raise ValueError("A GRPO uid collides across different questions")
    if len({group[0]["sample_id"] for group in groups.values()}) != prompt_count:
        raise ValueError("One training question was assigned multiple GRPO group uids")
    metrics = _aggregate(rows, "train")
    totals = [sum(row["score"] for row in group) for group in groups.values()]
    metrics.update({
        "train/question_count": len(groups),
        "train/group_all_zero_fraction": sum(total == 0 for total in totals) / len(groups),
        "train/group_all_one_fraction": sum(total == group_size for total in totals) / len(groups),
        "train/group_mixed_fraction": sum(0 < total < group_size for total in totals) / len(groups),
        "train/pass_at_5": sum(total > 0 for total in totals) / len(groups),
        "train/group_zero_variance_fraction": sum(total in (0, group_size) for total in totals) / len(groups),
    })
    output = Path(_get(trainer.config, "trainer.default_local_dir"))
    audit = _write_audit(output / "audit", "train", int(trainer.global_steps), rows)
    all_metrics = {**_plain(native_metrics), **metrics}
    _append_jsonl(output / "metrics.jsonl", {"global_step": int(trainer.global_steps),
                                           "metrics": all_metrics, "audit": audit})
    checkpoint = output / f"global_step_{int(trainer.global_steps)}" / "actor"
    if checkpoint.is_dir():
        _atomic_json(output / "last_checkpoint.json", {
            "schema_version": SCHEMA_VERSION, "global_step": int(trainer.global_steps),
            "checkpoint_kind": "verl_fsdp_actor", "actor_checkpoint": str(checkpoint),
            "checkpoint_ready": True, "training_audit": audit,
        })
    return metrics
