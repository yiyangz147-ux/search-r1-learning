#!/usr/bin/env python3
"""Apply a small, hash-bound patch to an isolated verl v0.6.1 checkout.

No networking and no modifications to the original SFT/evaluation environment.
Every changed file has a pristine backup, before/after SHA and a unified diff.
Reapplying the same patch is a no-op; unknown upstream changes are rejected.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PATCH_VERSION = "search-grpo-v061-v6"
BASELINE_SHA256 = {
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py":
        "731c21c4e240baf614ca256626f09f755360ade68e3d5fbd6669395d9e56095d",
    "verl/trainer/ppo/ray_trainer.py":
        "423b50445817f6d788fd1a6a3894f448d951f09e3aea8f7db71f8c1dc64b836b",
    "verl/workers/fsdp_workers.py":
        "ac75354db92440f6bed447a8272b77e62d5d64f33430ef43c50c1fc309e9376c",
    "verl/workers/actor/dp_actor.py":
        "ec53e7a8b88e2b9dbdca40af07fd84b1e1b735cdf75378ab284383640b44f5bf",
    "verl/workers/config/model.py":
        "f112b54475d474ea163592baa455c311f5c07b71a7c697b0279043be135425de",
}

# Recognized previous transformations can migrate only through pristine backups.
# v6 adds bounded LP/entropy chunks to the already-patched v5 padded actor path.
SUPPORTED_PREVIOUS_SHA256 = {
    "verl/workers/actor/dp_actor.py": {
        "416f18acf5400fd7523fb020eb188a59c32276e2053b879178da24413acc02f0",
    },
    "verl/trainer/ppo/ray_trainer.py": {
        "c5aca063376b878563d41dac78536bf69265e6acdcdd763341d082a5d033046b",
    },
}


def _replace_once(source, before, after, label):
    count = source.count(before)
    if count != 1:
        raise ValueError(f"Pinned source anchor {label!r}: expected 1 occurrence, found {count}")
    return source.replace(before, after, 1)


def patched_source(relative, source):
    if relative.endswith("vllm_async_server.py"):
        source = _replace_once(source,
            "        self.config.max_model_len = self.config.prompt_length + self.config.response_length\n",
            "        # Search-R1: tensor padding capacities are not the real context window.\n"
            "        if self.config.max_model_len is None:\n"
            "            self.config.max_model_len = self.config.prompt_length + self.config.response_length\n",
            "respect_explicit_context")
        source = _replace_once(source,
            "        max_tokens = self.config.max_model_len - len(prompt_ids)\n"
            "        sampling_params[\"logprobs\"] = 0 if sampling_params.pop(\"logprobs\", False) else None\n",
            "        # Search-R1: retain the per-action cap and never mutate the caller's dict.\n"
            "        sampling_params = dict(sampling_params)\n"
            "        remaining = self.config.max_model_len - len(prompt_ids)\n"
            "        max_tokens = min(remaining, int(sampling_params.pop(\"max_tokens\", remaining)))\n"
            "        if max_tokens <= 0:\n"
            "            raise ValueError(\"No positive generation budget remains within max_model_len\")\n"
            "        sampling_params[\"logprobs\"] = 0 if sampling_params.pop(\"logprobs\", False) else None\n",
            "respect_per_action_budget")
    elif relative == "verl/workers/config/model.py":
        source = _replace_once(source,
            '    target_modules: Optional[str] = "all-linear"\n',
            '    # PEFT accepts a regex string or an explicit module-name list.\n'
            '    # OmegaConf container unions are unsupported; Any preserves both representations.\n'
            '    target_modules: Any = "all-linear"\n',
            "allow_explicit_lora_target_module_list")
    elif relative.endswith("dp_actor.py"):
        source = _replace_once(source,
            "        response_length = micro_batch[\"responses\"].size(-1)\n"
            "        multi_modal_inputs = {}\n",
            "        # Search-R1: avoid a full padded 9216-token forward for microbatch=1.\n"
            "        # Real token IDs, positions and observation context remain unchanged.\n"
            "        if not self.use_remove_padding and not self.use_fused_kernels and self.ulysses_sequence_parallel_size == 1:\n"
            "            from search_grpo_metrics import trim_micro_batch_padding, restore_response_padding\n"
            "            trimmed = trim_micro_batch_padding(micro_batch)\n"
            "            if trimmed is not None:\n"
            "                trimmed_inputs, original_response = trimmed\n"
            "                # No edge padding remains, so the recursive call takes the native path.\n"
            "                entropy, log_probs = self._forward_micro_batch(\n"
            "                    trimmed_inputs, temperature=temperature, calculate_entropy=calculate_entropy\n"
            "                )\n"
            "                return restore_response_padding(entropy, log_probs, original_response)\n\n"
            "        response_length = micro_batch[\"responses\"].size(-1)\n"
            "        multi_modal_inputs = {}\n",
            "microbatch_one_edge_padding_trim")
        source = _replace_once(source,
            '                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])\n'
            '                    if calculate_entropy:\n'
            '                        if not self.config.entropy_checkpointing:\n'
            '                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)\n'
            '                        else:\n'
            '                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)\n',
            '                    # Search-R1: bound vocabulary-sized forward/backward temporaries.\n'
            '                    # This is the actual padded SDPA path, not the unused rmpad selector.\n'
            '                    from search_grpo_metrics import chunked_logprobs_entropy\n'
            '                    entropy, log_probs = chunked_logprobs_entropy(\n'
            '                        logits, micro_batch["responses"], calculate_entropy=calculate_entropy,\n'
            '                        chunk_size=128, checkpoint_chunks=True,\n'
            '                        logprobs_fn=logprobs_from_logits, entropy_fn=verl_F.entropy_from_logits,\n'
            '                    )\n',
            "bounded_padded_logprobs_and_entropy")
    elif relative.endswith("fsdp_workers.py"):
        source = _replace_once(source,
            "            actor_module.to(torch_dtype)\n",
            "            # Search-R1: Module.to(dtype) would quantize reconstructed FP32 RoPE buffers.\n"
            "            from search_grpo_metrics import cast_model_parameters_preserving_buffers\n"
            "            cast_model_parameters_preserving_buffers(actor_module, torch_dtype)\n",
            "preserve_fp32_rotary_buffers")
        source = _replace_once(source,
            "                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))\n\n"
            "        self.use_orig_params = fsdp_config.get(\"use_orig_params\", False)\n",
            "                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))\n\n"
            "        if role == \"actor\" and self._is_actor:\n"
            "            from search_grpo_metrics import audit_actor_parameters\n"
            "            audit_actor_parameters(actor_module, self.config, rank=self.rank)\n\n"
            "        self.use_orig_params = fsdp_config.get(\"use_orig_params\", False)\n",
            "audit_actual_trainable_actor_before_fsdp")
    elif relative.endswith("ray_trainer.py"):
        source = _replace_once(source, "from verl import DataProto\n",
            "from verl import DataProto\n"
            "from search_grpo_metrics import audit_batch as search_grpo_audit_batch\n"
            "from search_grpo_metrics import record_training as search_grpo_record_training\n"
            "from search_grpo_metrics import record_validation as search_grpo_record_validation\n"
            "from search_grpo_metrics import _plain as search_grpo_plain\n",
            "metrics_imports")
        source = _replace_once(source,
            "            lines.append(json.dumps(entry, ensure_ascii=False))\n",
            "            lines.append(json.dumps(search_grpo_plain(entry), ensure_ascii=False))\n",
            "serialize_numpy_tensor_rollout_metadata")
        source = _replace_once(source,
            "    def _validate(self):\n        data_source_lst = []\n",
            "    def _validate(self):\n        search_grpo_validation_rows = []\n        data_source_lst = []\n",
            "validation_collect_init")
        source = _replace_once(source,
            "            sample_scores.extend(scores)\n\n            reward_extra_infos_dict[\"reward\"].extend(scores)\n",
            "            sample_scores.extend(scores)\n"
            "            search_grpo_validation_rows.extend(search_grpo_audit_batch(\n"
            "                test_batch, scores, step=self.global_steps, split=\"validation\"\n"
            "            ))\n\n            reward_extra_infos_dict[\"reward\"].extend(scores)\n",
            "validation_collect_rows")
        source = _replace_once(source,
            "        return metric_dict\n\n    def init_workers(self):\n",
            "        metric_dict.update(search_grpo_record_validation(self, search_grpo_validation_rows))\n"
            "        return metric_dict\n\n    def init_workers(self):\n",
            "validation_aggregate_select")
        source = _replace_once(source,
            "                # TODO: make a canonical logger that supports various backend\n"
            "                logger.log(data=metrics, step=self.global_steps)\n",
            "                # Audit original assistant/environment events before publishing numeric metrics.\n"
            "                metrics.update(search_grpo_record_training(self, batch, metrics))\n"
            "                # TODO: make a canonical logger that supports various backend\n"
            "                logger.log(data=metrics, step=self.global_steps)\n",
            "training_audit_metrics")
    else:
        raise ValueError(f"Unsupported vendor file: {relative}")
    ast.parse(source)
    return source


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def apply_patches(vendor_root):
    root = Path(vendor_root).resolve()
    metadata_dir = root / ".search-grpo-patches" / PATCH_VERSION
    manifest_path = metadata_dir / "manifest.json"
    planned = []
    # Inspect every file before writing any mutation.
    for relative, expected_sha in BASELINE_SHA256.items():
        target = root / relative
        backup = metadata_dir / "original" / relative
        legacy_backups = [root / f".search-grpo-patches/{version}/original" / relative
                          for version in ("search-grpo-v061-v5", "search-grpo-v061-v4", "search-grpo-v061-v3", "search-grpo-v061-v2", "search-grpo-v061-v1")]
        current = target.read_bytes()
        if sha256(current) == expected_sha:
            original = current
        elif backup.exists() and sha256(backup.read_bytes()) == expected_sha:
            original = backup.read_bytes()
        elif any(path.exists() and sha256(path.read_bytes()) == expected_sha for path in legacy_backups):
            original = next(path.read_bytes() for path in legacy_backups
                            if path.exists() and sha256(path.read_bytes()) == expected_sha)
        else:
            raise ValueError(f"Unrecognized verl v0.6.1 file; refusing to patch: {relative}")
        transformed = patched_source(relative, original.decode("utf-8")).encode("utf-8")
        if current not in (original, transformed) and sha256(current) not in SUPPORTED_PREVIOUS_SHA256.get(relative, set()):
            raise ValueError(f"Vendor file changed after previous patch: {relative}")
        planned.append((relative, target, backup, original, transformed, current == transformed))
    metadata_dir.mkdir(parents=True, exist_ok=True)
    records, diff = [], []
    for relative, target, backup, original, transformed, already in planned:
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            if backup.read_bytes() != original:
                raise ValueError(f"Pristine vendor backup was modified: {backup}")
        else:
            backup.write_bytes(original)
            shutil.copystat(target, backup)
        if not already:
            temporary = target.with_name(target.name + ".search-grpo-tmp")
            temporary.write_bytes(transformed)
            shutil.copystat(target, temporary)
            temporary.replace(target)
        records.append({"path": relative, "before_sha256": sha256(original),
                        "after_sha256": sha256(transformed), "backup": str(backup)})
        diff.extend(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                        transformed.decode().splitlines(keepends=True),
                                        fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    diff_path = metadata_dir / "changes.patch"
    diff_path.write_text("".join(diff))
    manifest = {"patch_version": PATCH_VERSION, "upstream": "verl v0.6.1",
                "files": records, "diff": str(diff_path)}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(key) != value for key, value in manifest.items()):
            raise ValueError("Existing patch manifest disagrees with deterministic patch output")
        return old
    manifest["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_patches(args.vendor_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
