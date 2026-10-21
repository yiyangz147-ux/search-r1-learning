#!/usr/bin/env python3
"""Prepare isolated GRPO splits and export the complete SFT policy as an HF base.

`data` is CPU/data-only. `export-model` uses the existing SFT environment and
requires its full-embedding loader; ordinary PEFT loading would lose the SFT E.
Both publish manifests last and refuse to overwrite an existing destination.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from prepare_lora_sft_data_v2 import QuestionIndex, normalized_question
from search_sft_protocol import PROTOCOL_VERSION, build_prompt


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_rows(rows, label, require_answers=True):
    result, ids = [], set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        sid = row.get("sample_id", row.get("id", f"{label}:{index}"))
        question = row.get("question")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError(f"Missing or repeated sample ID in {label}: {sid}")
        if not isinstance(question, str) or not normalized_question(question):
            raise ValueError(f"Missing question in {label}: {sid}")
        gold = row.get("golden_answers", row.get("gold_answers"))
        if require_answers and (not isinstance(gold, list) or not gold or any(not isinstance(a, str) or not a.strip() for a in gold)):
            raise ValueError(f"Missing answer aliases in {label}: {sid}")
        if require_answers and row.get("data_source") not in ("nq", "hotpotqa"):
            raise ValueError(f"Unexpected source in {label}: {sid}")
        ids.add(sid)
        row["sample_id"] = sid
        if require_answers:
            row["golden_answers"] = gold
        result.append(row)
    if not result:
        raise ValueError(f"Empty dataset: {label}")
    return result


class ProtectedQuestions:
    """Exact normalized-question matches plus recall-complete 3-gram Jaccard."""
    def __init__(self, rows):
        self.rows = rows
        self.index = QuestionIndex(rows)
        self.exact = {}
        for row in rows:
            self.exact.setdefault(normalized_question(row["question"]), []).append(row["sample_id"])

    def matches(self, row):
        matches = {item["sample_id"]: item for item in self.index.matches(row)}
        for sid in self.exact.get(normalized_question(row["question"]), []):
            matches[sid] = {"sample_id": sid, "jaccard": 1.0}
        return sorted(matches.values(), key=lambda x: x["sample_id"])


def ensure_separate(rows, protected, label):
    collisions = [{"sample_id": row["sample_id"], "matches": matches}
                  for row in rows if (matches := protected.matches(row))]
    if collisions:
        raise ValueError(f"{label} overlaps protected questions: {json.dumps(collisions[:10])}")


def choose_validation(reserve, protected, seed, per_source):
    eligible, excluded = [], []
    for row in reserve:
        matches = protected.matches(row)
        if matches:
            excluded.append({"sample_id": row["sample_id"], "matches": matches})
        else:
            eligible.append(row)
    rng = random.Random(seed)
    selected = []
    for source in ("nq", "hotpotqa"):
        # Hash/ID sorting makes selection independent of accidental file reorder.
        candidates = sorted((row for row in eligible if row["data_source"] == source), key=lambda row: row["sample_id"])
        rng.shuffle(candidates)
        accepted, local = [], ProtectedQuestions(selected)
        for row in candidates:
            if local.matches(row):
                continue
            accepted.append(row)
            local = ProtectedQuestions(selected + accepted)
            if len(accepted) == per_source:
                break
        if len(accepted) != per_source:
            raise ValueError(f"Not enough independent {source} reserve questions: need {per_source}")
        selected.extend(accepted)
    rng.shuffle(selected)
    return selected, eligible, excluded


def verl_record(row, split, index):
    """Raw string prompt is consumed by our Base dataset, never a chat template."""
    return {"data_source": row["data_source"], "prompt": build_prompt(row["question"]),
            "ability": "fact-retrieval", "agent_name": "search_r1_base",
            "reward_model": {"style": "rule", "ground_truth": {"target": list(row["golden_answers"])}},
            "extra_info": {"sample_id": row["sample_id"], "question": row["question"],
                           "data_source": row["data_source"], "split": split, "index": index},
            "sample_id": row["sample_id"], "question": row["question"],
            "golden_answers": list(row["golden_answers"])}


def file_records(directory):
    return {str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(directory.rglob("*")) if path.is_file()}


def check_existing(destination, manifest_name, request):
    manifest_path = destination / manifest_name
    if not destination.exists():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"Refusing an existing incomplete destination: {destination}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or manifest.get("request") != request:
        raise ValueError(f"Existing destination belongs to a different preparation: {destination}")
    for name, metadata in manifest["files"].items():
        path = destination / name
        if not path.is_file() or path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Prepared artifact changed: {path}")
    return manifest


def prepare_data(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    inputs = {name: Path(getattr(args, name)).resolve() for name in
              ("grpo_file", "reserve_file", "test_file", "sft_train", "sft_val", "monitor_file")}
    source_hashes = {str(path): sha256(path) for path in inputs.values()}
    request = {"inputs": {key: str(value) for key, value in inputs.items()}, "source_hashes": source_hashes,
               "seed": args.seed, "dev_per_source": args.dev_per_source, "format_version": 1,
               "protocol_version": PROTOCOL_VERSION}
    output = Path(args.output_dir).resolve()
    prior = check_existing(output, "data_manifest.json", request)
    if prior is not None:
        return prior
    for source in inputs.values():
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must be separate from all input files")
    train = canonical_rows(read_rows(inputs["grpo_file"]), "grpo")
    if len(train) != 2000 or Counter(row["data_source"] for row in train) != {"nq": 1000, "hotpotqa": 1000}:
        raise ValueError("GRPO input must be the original NQ 1000 + HotpotQA 1000 questions")
    if any(row.get("partition") != "grpo" for row in train):
        raise ValueError("GRPO input must retain its original partition=grpo")
    test = canonical_rows(read_rows(inputs["test_file"]), "test")
    if len(test) != 1000 or Counter(row["data_source"] for row in test) != {"nq": 500, "hotpotqa": 500}:
        raise ValueError("Test input must be the original NQ 500 + HotpotQA 500 questions")
    sft_train = canonical_rows(read_rows(inputs["sft_train"]), "sft_train", require_answers=False)
    sft_val = canonical_rows(read_rows(inputs["sft_val"]), "sft_val", require_answers=False)
    sft_index = ProtectedQuestions(sft_train + sft_val)
    ensure_separate(train, sft_index, "GRPO train")
    ensure_separate(test, ProtectedQuestions(train + sft_train + sft_val), "Original test")
    reserve = canonical_rows(read_rows(inputs["reserve_file"]), "reserve")
    protected = ProtectedQuestions(train + test + sft_train + sft_val)
    validation, eligible, excluded = choose_validation(reserve, protected, args.seed, args.dev_per_source)
    # Assert pairwise independence of all selected validation rows.
    seen = []
    for row in validation:
        ensure_separate([row], ProtectedQuestions(seen), "Validation duplicates")
        seen.append(row)
    monitor = canonical_rows(read_rows(inputs["monitor_file"]), "monitor")
    test_by_question = {normalized_question(row["question"]): row for row in test}
    exclusions = []
    for row in monitor:
        match = test_by_question.get(normalized_question(row["question"]))
        if match is None or set(match["golden_answers"]) != set(row["golden_answers"]):
            raise ValueError(f"SFT monitor question/gold does not match original test: {row['sample_id']}")
        exclusions.append({"sample_id": match["sample_id"], "monitor_sample_id": row["sample_id"],
                           "data_source": match["data_source"], "reason": "used_in_SFT_reward_monitor"})
    if len(exclusions) != 32 or len({row["sample_id"] for row in exclusions}) != 32:
        raise ValueError("Expected exactly 32 distinct original test monitoring exclusions")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        shutil.copyfile(inputs["grpo_file"], temporary / "train.selected.jsonl")
        shutil.copyfile(inputs["test_file"], temporary / "test.selected.jsonl")
        write_jsonl(temporary / "val.selected.jsonl", validation)
        for split, rows in (("train", train), ("val", validation), ("test", test)):
            pq.write_table(pa.Table.from_pylist([verl_record(row, split, i) for i, row in enumerate(rows)]), temporary / f"{split}.parquet")
        write_json(temporary / "monitor_exclusion.json", exclusions)
        write_json(temporary / "reserve_exclusions.json", excluded)
        manifest = {"complete": True, "created_at": datetime.now(timezone.utc).isoformat(), "request": request,
                    "counts": {"train": len(train), "val": len(validation), "test": len(test),
                               "reserve": len(reserve), "reserve_eligible": len(eligible), "reserve_excluded": len(excluded),
                               "test_excluding_SFT_monitor": len(test) - len(exclusions)},
                    "by_source": {name: dict(Counter(r["data_source"] for r in rows)) for name, rows in
                                  (("train", train), ("val", validation), ("test", test))},
                    "overlap_policy": {"normalized_exact": True, "word_shingle_size": 3, "jaccard_at_least": 0.85,
                                       "test_previously_used_for_SFT_monitor": 32},
                    "data_contract": {"prompt": "raw Base string", "chat_template": False,
                                      "agent_name": "search_r1_base", "ground_truth": "reward_model.ground_truth.target[]"}}
        manifest["files"] = file_records(temporary)
        write_json(temporary / "data_manifest.json", manifest)
        os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def plain_tied_model(model):
    """Remove SFT mixed-precision hooks, then materialize its compute E in BF16."""
    import torch
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if hasattr(embedding, "_full_embedding_cast_handle"):
        embedding._full_embedding_cast_handle.remove()
        del embedding._full_embedding_cast_handle
    if hasattr(embedding, "_full_embedding_compute_dtype"):
        del embedding._full_embedding_compute_dtype
    if hasattr(head, "_full_embedding_original_forward"):
        head.forward = head._full_embedding_original_forward
        del head._full_embedding_original_forward
    # Cast only the FP32 master Parameter. Casting the whole model would also
    # quantize non-persistent FP32 RoPE buffers, which HF reconstructs as FP32.
    embedding.weight.data = embedding.weight.data.to(dtype=torch.bfloat16)
    model.tie_weights()
    model.requires_grad_(False)
    if embedding.weight is not head.weight or embedding.weight.dtype != torch.bfloat16:
        raise ValueError("Export did not retain a standard BF16 tied embedding")
    model.config.torch_dtype = torch.bfloat16
    return model


def numerical_comparison(before, after):
    import torch
    differences = (before.float() - after.float()).abs()
    old_logp = before.float().log_softmax(-1)
    new_logp = after.float().log_softmax(-1)
    old_top = before.float().topk(min(2, before.numel())).values
    new_top = after.float().topk(min(2, after.numel())).values
    return {"max_abs_logit_difference": differences.max().item(),
            "mean_abs_logit_difference": differences.mean().item(),
            "next_token_KL_before_to_after": (old_logp.exp() * (old_logp - new_logp)).sum().item(),
            "before_top1_margin": (old_top[0] - old_top[-1]).item(),
            "after_top1_margin": (new_top[0] - new_top[-1]).item(),
            "before_top1_probability": old_logp.max().exp().item(),
            "after_top1_probability": new_logp.max().exp().item(),
            "next_token_top1_equal": bool(torch.equal(before.argmax(-1), after.argmax(-1))),
            "top10_overlap": len(set(before.flatten().topk(min(10, before.numel())).indices.tolist()) &
                                 set(after.flatten().topk(min(10, after.numel())).indices.tolist()))}


def capture_probe(model, tokenizer, questions, device, max_new_tokens):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from sft_reward_monitor import parse_action, valid_step_format

    class CompleteActionStop(StoppingCriteria):
        def __init__(self, prefix_length):
            self.prefix_length = prefix_length

        def __call__(self, input_ids, scores, **kwargs):
            texts = tokenizer.batch_decode(input_ids[:, self.prefix_length:], skip_special_tokens=False)
            return torch.tensor(["</search>" in text or "</answer>" in text for text in texts], device=input_ids.device)
    result = []
    model.eval()
    for row in questions:
        encoded = tokenizer(build_prompt(row["question"]), return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(device)
        with torch.inference_mode():
            logits = model(**encoded, logits_to_keep=1).logits[0, -1].detach().float().cpu()
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                       use_cache=True, stopping_criteria=StoppingCriteriaList([CompleteActionStop(encoded.input_ids.shape[-1])]))[0, encoded.input_ids.shape[-1]:].detach().cpu().tolist()
        text = tokenizer.decode(generated, skip_special_tokens=False)
        action = parse_action(text)
        result.append({"sample_id": row["sample_id"], "logits": logits, "greedy_token_ids": generated,
                       "greedy_text": text, "first_action": action, "valid_format": valid_step_format(text)})
        print(json.dumps({"probe_sample_complete": row["sample_id"], "new_tokens": len(generated), "first_action": action}), flush=True)
    return result


def compare_probes(before, after):
    if [item["sample_id"] for item in before] != [item["sample_id"] for item in after]:
        raise ValueError("Model probe IDs changed")
    return [{"sample_id": old["sample_id"], **numerical_comparison(old["logits"], new["logits"]),
             "greedy_token_ids_before": old["greedy_token_ids"], "greedy_token_ids_after": new["greedy_token_ids"],
             "greedy_tokens_equal": old["greedy_token_ids"] == new["greedy_token_ids"],
             "greedy_text_before": old["greedy_text"], "greedy_text_after": new["greedy_text"],
             "first_action_before": old["first_action"], "first_action_after": new["first_action"],
             "first_action_equal": (None if old["first_action"] is None or new["first_action"] is None else
                                    all(old["first_action"][k] == new["first_action"][k] for k in ("action", "value"))),
             "valid_format_before": old["valid_format"], "valid_format_after": new["valid_format"]}
            for old, new in zip(before, after)]


def export_model(args):
    def phase(name):
        print(json.dumps({"phase": name, "at": datetime.now(timezone.utc).isoformat()}), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from full_embedding_lora import assert_full_embedding_tied, load_full_embedding_adapter

    torch.set_num_threads(args.cpu_threads)
    base, adapter, output = (Path(getattr(args, name)).resolve() for name in ("base_model", "adapter_path", "output_dir"))
    source_files = [path for root in (base, adapter) for path in root.iterdir() if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja", ".txt", ".model")]
    phase("hash_source_model_and_SFT_checkpoint")
    source_hashes = {str(path): sha256(path) for path in sorted(source_files)}
    validation_file = Path(args.validation_file).resolve()
    source_hashes[str(validation_file)] = sha256(validation_file)
    request = {"base_model": str(base), "adapter_path": str(adapter), "source_hashes": source_hashes,
               "validation_file": str(validation_file), "probe_count": args.probe_count,
               "probe_tokens": args.probe_tokens, "verify_device": args.verify_device,
               "max_allowed_merge_logit_difference": args.max_logit_difference,
               "format_version": 1, "export_dtype": "bfloat16"}
    prior = check_existing(output, "sft_export_manifest.json", request)
    if prior is not None:
        return prior
    if output == base or output == adapter or base in output.parents or adapter in output.parents:
        raise ValueError("Merged output must be separate from both source models")
    questions = canonical_rows(read_rows(validation_file), "validation")[:args.probe_count]
    if len(questions) != args.probe_count:
        raise ValueError("Not enough validation probe questions")
    tokenizer_root = adapter if (adapter / "tokenizer_config.json").is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True, trust_remote_code=False)
    phase("load_original_base_CPU")
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.bfloat16, device_map="cpu",
                                                attn_implementation="eager", local_files_only=True, trust_remote_code=False)
    sample_token_ids = [i for i in (0, 1, 151667, 151668) if i < model.get_input_embeddings().weight.shape[0]]
    original_e_sample = model.get_input_embeddings().weight.detach()[sample_token_ids].float().clone()
    phase("restore_SFT_LoRA_and_full_FP32_master_embedding")
    model = load_full_embedding_adapter(model, adapter, is_trainable=False).eval()
    shared = assert_full_embedding_tied(model, require_trainable=False)
    expected_e_hash = tensor_digest(shared.to(dtype=torch.bfloat16))
    e_sample_delta = (shared.detach()[sample_token_ids].float() - original_e_sample).abs()
    # Hash every target's exact BF16 merge result, without keeping a model copy.
    phase("hash_BF16_embedding_and_all_expected_merged_qv_weights")
    expected_targets = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            delta = module.get_delta_weight("default")
            expected = module.get_base_layer().weight.detach().clone()
            expected.add_(delta)  # Match PEFT safe_merge: add FP32 delta, round once to BF16.
            expected_targets[name.removeprefix("base_model.model.")] = {"merged_sha256": tensor_digest(expected),
                       "delta_abs_max": delta.float().abs().max().item(), "delta_nonzero": bool(torch.count_nonzero(delta)),
                       "dynamic_LoRA_branch_dtype": str(module.lora_A["default"].weight.dtype)}
            del expected, delta
    if not expected_targets or not all(item["delta_nonzero"] for item in expected_targets.values()):
        raise ValueError("SFT LoRA was missing or contains a zero target delta")
    device = torch.device(args.verify_device)
    model.to(device)
    phase("probe_unmerged_SFT_on_" + str(device))
    before = capture_probe(model, tokenizer, questions, device, args.probe_tokens)
    model.to("cpu")
    original_buffer_dtypes = {name.removeprefix("base_model.model."): str(value.dtype) for name, value in model.named_buffers()}
    phase("safe_merge_LoRA_CPU_and_materialize_standard_BF16_tied_embedding")
    merged = plain_tied_model(model.merge_and_unload(safe_merge=True))
    del model
    merged_buffer_dtypes = {name: str(value.dtype) for name, value in merged.named_buffers()}
    if original_buffer_dtypes != merged_buffer_dtypes:
        raise ValueError(f"Merging altered non-parameter buffer dtypes: {original_buffer_dtypes} -> {merged_buffer_dtypes}")
    if tensor_digest(merged.get_input_embeddings().weight) != expected_e_hash:
        raise ValueError("Full SFT embedding changed during merge")
    modules = dict(merged.named_modules())
    for name, entry in expected_targets.items():
        if name not in modules or tensor_digest(modules[name].weight) != entry["merged_sha256"]:
            raise ValueError(f"SFT LoRA merge differs from expected BF16 target: {name}")
    merged.to(device)
    phase("probe_merged_SFT_on_" + str(device))
    after = capture_probe(merged, tokenizer, questions, device, args.probe_tokens)
    merge_comparisons = compare_probes(before, after)
    # BF16 fused weights and dynamic LoRA are mathematically close, not bitwise equal.
    # Refuse gross corruption; preserve top1/greedy disagreement explicitly for review.
    if any(item["max_abs_logit_difference"] > args.max_logit_difference for item in merge_comparisons):
        raise ValueError(f"Merge probe exceeds allowed logit difference: {merge_comparisons}")
    merged.to("cpu")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        phase("save_standard_HF_shards_and_tokenizer")
        merged.save_pretrained(str(temporary), safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(str(temporary))
        del merged
        if device.type == "cuda":
            torch.cuda.empty_cache()
        phase("reload_HF_shards_and_check_every_qv_and_embedding_hash")
        reloaded = AutoModelForCausalLM.from_pretrained(str(temporary), torch_dtype=torch.bfloat16, device_map="cpu",
                                                        attn_implementation="eager", local_files_only=True, trust_remote_code=False)
        if reloaded.get_input_embeddings().weight is not reloaded.get_output_embeddings().weight:
            raise ValueError("Saved HF export did not reload with Weight Tying")
        if tensor_digest(reloaded.get_input_embeddings().weight) != expected_e_hash:
            raise ValueError("Saved HF model lost trained SFT embedding")
        reload_modules = dict(reloaded.named_modules())
        for name, entry in expected_targets.items():
            if tensor_digest(reload_modules[name].weight) != entry["merged_sha256"]:
                raise ValueError(f"Saved HF model lost merged q/v target: {name}")
        reloaded.requires_grad_(False).to(device)
        phase("probe_reloaded_HF_on_" + str(device))
        final = capture_probe(reloaded, tokenizer, questions, device, args.probe_tokens)
        reload_comparisons = compare_probes(after, final)
        if any(item["max_abs_logit_difference"] != 0 or not item["greedy_tokens_equal"] for item in reload_comparisons):
            raise ValueError(f"Saved model reload did not reproduce merged model probes: {reload_comparisons}")
        manifest = {"complete": True, "created_at": datetime.now(timezone.utc).isoformat(), "request": request,
                    "shared_embedding": {"dtype": "bfloat16", "tied": True, "tensor_sha256": expected_e_hash,
                                         "source_master_dtype": "float32", "sample_token_ids": sample_token_ids, "sample_source_vs_base_abs_max": e_sample_delta.max().item()},
                    "non_parameter_buffer_dtypes": merged_buffer_dtypes,
                    "merged_lora_targets": expected_targets,
                    "merge_probe_comparisons": merge_comparisons, "reload_probe_comparisons": reload_comparisons,
                    "merge_probe_summary": {"samples": len(merge_comparisons),
                        "max_abs_logit_difference": max(x["max_abs_logit_difference"] for x in merge_comparisons),
                        "mean_next_token_KL": sum(x["next_token_KL_before_to_after"] for x in merge_comparisons) / len(merge_comparisons),
                        "max_next_token_KL": max(x["next_token_KL_before_to_after"] for x in merge_comparisons),
                        "same_next_token_top1": sum(x["next_token_top1_equal"] for x in merge_comparisons),
                        "same_full_greedy": sum(x["greedy_tokens_equal"] for x in merge_comparisons),
                        "same_complete_first_action": sum(x["first_action_equal"] is True for x in merge_comparisons),
                        "original_dynamic_LoRA_bitwise_equivalence": False,
                        "requires_merged_step_zero_validation_baseline": True},
                    "new_grpo_adapter": {"created_by": "verl actor", "r": 32, "lora_alpha": 16,
                                         "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.0,
                                         "initial_delta": "zero (PEFT default B initialization)", "base_frozen": True},
                    "numerical_note": "BF16 merging and dynamic LoRA differ in rounding; comparison values above are measured, not bitwise-equivalence claims."}
        phase("fingerprint_exported_HF_files")
        manifest["files"] = file_records(temporary)
        write_json(temporary / "sft_export_manifest.json", manifest)
        os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="CPU-only independent train/dev/test preparation")
    for option in ("grpo-file", "reserve-file", "test-file", "sft-train", "sft-val", "monitor-file", "output-dir"):
        data.add_argument("--" + option, required=True)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--dev-per-source", type=int, default=100)
    model = sub.add_parser("export-model", help="Restore SFT embedding + LoRA, merge, save and verify standard HF")
    for option in ("base-model", "adapter-path", "output-dir", "validation-file"):
        model.add_argument("--" + option, required=True)
    model.add_argument("--verify-device", default="cpu")
    model.add_argument("--probe-count", type=int, default=2)
    model.add_argument("--probe-tokens", type=int, default=8)
    model.add_argument("--cpu-threads", type=int, default=8)
    model.add_argument("--max-logit-difference", type=float, default=1.0)
    return result


def main():
    args = parser().parse_args()
    if args.command == "data":
        if args.dev_per_source < 1:
            raise ValueError("dev-per-source must be positive")
        value = prepare_data(args)
        print(json.dumps({"complete": value["complete"], "counts": value["counts"], "output_dir": args.output_dir}))
    else:
        if min(args.probe_count, args.probe_tokens, args.cpu_threads) < 1:
            raise ValueError("Model probe and CPU limits must be positive")
        value = export_model(args)
        print(json.dumps({"complete": value["complete"], "output_dir": args.output_dir,
                          "merge_probe_comparisons": value["merge_probe_comparisons"]}))


if __name__ == "__main__":
    main()
