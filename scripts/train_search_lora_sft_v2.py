#!/usr/bin/env python3
"""Train q/v LoRA plus the full tied embedding, with generated RAG reward monitoring.

Original v1 training, data, and evaluation artifacts are kept unchanged.
Only masked assistant tokens contribute to SFT; reward is monitoring, not RL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import uuid


DEFAULTS = {
    "wandb_entity": None,
    "seed": 42,
    "max_length": 4096,
    "epochs": 4,
    "lr": 1e-5,
    "betas": [0.9, 0.95],
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "lora_rank": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "per_device_batch": 1,
    "eval_per_device_batch": 1,
    "gradient_accumulation": 8,
    "bf16": True,
    "attn_implementation": "sdpa",
    "gradient_checkpointing": True,
    "tf32": True,
    "dataloader_num_workers": 2,
    "save_total_limit": 2,
    "embedding_lr": 1e-5,
    "full_embedding": True,
    "max_steps": -1,
    "reward_every_steps": 40,
    "reward_on_start": True,
    "reward_examples": 32,
    "reward_max_new_tokens": 512,
    "reward_max_context_tokens": 8192,
    "reward_max_searches": 3,
    "retriever_url": "http://127.0.0.1:8000/retrieve",
    "smoke_train_examples": 0,
    "smoke_eval_examples": 0,
}
REQUIRED = {
    "model_path", "prepared_dir", "output_dir", "run_name", "wandb_project",
    "selected_gpu_uuids",
}
MASK_POLICY = {
    "prompt_masked": True,
    "information_masked": True,
    "padding_label": -100,
    "terminal_eos_supervised": True,
    "training_unit": "complete_episode",
    "chat_template": False,
    "boundary_straddling_tokens_masked": True,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    supplied = read_json(path)
    if not isinstance(supplied, dict):
        raise ValueError("The training config must be a JSON object")
    missing = REQUIRED - supplied.keys()
    unknown = supplied.keys() - REQUIRED - DEFAULTS.keys()
    if missing or unknown:
        raise ValueError(f"Config keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    for name in ("model_path", "prepared_dir", "output_dir"):
        config[name] = str(Path(config[name]).expanduser().resolve())
    for name in ("run_name", "wandb_project"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if config["wandb_entity"] is not None and not isinstance(config["wandb_entity"], str):
        raise ValueError("wandb_entity must be a string or null")
    positive_ints = ("max_length", "lora_rank", "lora_alpha", "per_device_batch",
                     "eval_per_device_batch", "gradient_accumulation", "save_total_limit")
    for name in positive_ints:
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config["dataloader_num_workers"]) is not int or config["dataloader_num_workers"] < 0:
        raise ValueError("dataloader_num_workers must be a nonnegative integer")
    for name in ("bf16", "gradient_checkpointing", "tf32"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be a JSON boolean")
    if not config["bf16"] or config["attn_implementation"] != "sdpa":
        raise ValueError("This recipe requires bf16=true and attn_implementation='sdpa'")
    for name in ("epochs", "lr", "embedding_lr", "max_grad_norm"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    for name in ("weight_decay", "warmup_ratio", "lora_dropout"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or not 0 <= config[name] <= 1:
            raise ValueError(f"{name} must be between zero and one")
    betas = config["betas"]
    if (not isinstance(betas, list) or len(betas) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 1 for x in betas)):
        raise ValueError("betas must contain two numbers in [0, 1)")
    targets = config["target_modules"]
    supported = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if (not isinstance(targets, list) or not targets or len(targets) != len(set(targets))
            or any(x not in supported for x in targets)):
        raise ValueError("target_modules must be unique Qwen attention/MLP projection names")
    selected = config["selected_gpu_uuids"]
    if (not isinstance(selected, list) or not selected or len(selected) != len(set(selected))
            or any(not isinstance(x, str) or not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", x)
                   for x in selected)):
        raise ValueError("selected_gpu_uuids must contain unique full GPU UUIDs")
    model_path, prepared_dir = Path(config["model_path"]), Path(config["prepared_dir"])
    output = Path(config["output_dir"])
    if not model_path.is_dir() or not prepared_dir.is_dir():
        raise ValueError("model_path and prepared_dir must exist locally")
    for protected in (model_path, prepared_dir):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("output_dir must be separate from the model and prepared data directories")
    if config["full_embedding"] is not True:
        raise ValueError("This v2 recipe requires the full tied embedding to be trainable")
    for key in ("reward_every_steps", "reward_examples", "reward_max_new_tokens",
                "reward_max_context_tokens", "reward_max_searches"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("smoke_train_examples", "smoke_eval_examples"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(config["max_steps"]) is not int or config["max_steps"] == 0 or config["max_steps"] < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if type(config["reward_on_start"]) is not bool:
        raise ValueError("reward_on_start must be a boolean")
    return config


def check_launch(config):
    expected = config["selected_gpu_uuids"]
    actual = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if actual != expected:
        raise ValueError("CUDA_VISIBLE_DEVICES must exactly match selected_gpu_uuids, including order")
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Launch this script through the provided single-node torchrun wrapper") from exc
    if world_size != len(expected) or local_world_size != world_size or rank != local_rank or not 0 <= rank < world_size:
        raise ValueError("The single-node torchrun world size/ranks do not match the GPU whitelist")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("This recipe requires online W&B recording; run wandb login before training")
    return rank, local_rank, world_size


class PreparedEpisodeDataset:
    """Return only model inputs; source text and identifiers never enter W&B."""

    def __init__(self, rows):
        self.items = [{name: row[name] for name in ("input_ids", "attention_mask", "labels")}
                      for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # The collator may pad labels; avoid mutating the stored episode arrays.
        return {name: value.copy() for name, value in self.items[index].items()}


def load_prepared(config):
    import pyarrow.parquet as pq

    root = Path(config["prepared_dir"])
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("format_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Prepared manifest must have format_version=1 and complete=true")
    from search_sft_protocol import PROTOCOL_VERSION
    from prepare_data import PROMPT
    if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("prompt_sha256") != hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError("Prepared prompt/protocol differs from the generation monitor")
    if manifest.get("max_length") != config["max_length"]:
        raise ValueError("Prepared max_length differs from the training recipe")
    if manifest.get("expected_new_episodes") != 100:
        raise ValueError("Prepared manifest must target exactly 100 new episodes")
    counts = manifest.get("counts", {})
    if counts.get("new_episodes") != 100 or counts.get("new_by_source") != {"nq": 50, "hotpotqa": 50}:
        raise ValueError("The required 100 new episodes (NQ50 + HotpotQA50) are incomplete")
    if any(manifest.get("mask_policy", {}).get(name) != value for name, value in MASK_POLICY.items()):
        raise ValueError("Prepared mask policy does not match complete-episode Base-model SFT")
    for name in ("config_fingerprint", "prompt_sha256"):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Prepared manifest lacks {name}")
    if canonical_hash(manifest.get("configuration")) != manifest["config_fingerprint"]:
        raise ValueError("Prepared configuration fingerprint is invalid")
    files = manifest.get("files", {})
    for relative, record in files.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Prepared manifest contains an unsafe or missing file")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Prepared file hash/size mismatch: {relative}")
    if not {"train.parquet", "val.parquet", "preparation_report.json"} <= files.keys():
        raise ValueError("Prepared manifest is missing required files")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Prepared manifest is missing its source-file fingerprints")
    for filename, expected_sha in source_hashes.items():
        source = Path(filename)
        if not source.is_absolute() or not source.is_file() or sha256_file(source) != expected_sha:
            raise ValueError("An original prepared-data input is missing or changed; prepare a fresh dataset")
    model_config = read_json(Path(config["model_path"]) / "config.json")
    if model_config.get("model_type") != "qwen3":
        raise ValueError("This recipe expects the local Qwen3-4B-Base model")
    vocab_size = model_config["vocab_size"]
    eos = manifest.get("tokenizer", {}).get("eos_token_id")
    if type(eos) is not int or eos < 0 or eos >= vocab_size:
        raise ValueError("Prepared tokenizer EOS is invalid")
    datasets, stats, seen = {}, {}, set()
    origins = Counter()
    for split, filename, expected_count in (
        ("train", "train.parquet", counts.get("train")),
        ("validation", "val.parquet", counts.get("validation")),
    ):
        rows = pq.read_table(root / filename).to_pylist()
        if not rows or len(rows) != expected_count or len(rows) != files[filename].get("rows"):
            raise ValueError(f"Prepared {split} row count mismatch or empty split")
        split_origins, supervised = Counter(), 0
        max_length = 0
        for number, row in enumerate(rows):
            label = f"{split} row {number}"
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"Missing or duplicate episode identifier at {label}")
            seen.add(sample_id)
            origin = row.get("data_origin")
            if origin not in {"aethersearch", "deepseek_new100"}:
                raise ValueError(f"Unrecognized data origin at {label}")
            ids, attention, labels = (row.get(name) for name in ("input_ids", "attention_mask", "labels"))
            if (not all(isinstance(x, list) for x in (ids, attention, labels)) or not ids
                    or not len(ids) == len(attention) == len(labels) == row.get("length")
                    or len(ids) > config["max_length"]):
                raise ValueError(f"Invalid token-array lengths at {label}")
            if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
                raise ValueError(f"Invalid input token at {label}")
            if any(type(mask) is not int or mask != 1 for mask in attention):
                raise ValueError(f"Prepared episodes must contain no padding at {label}")
            if any(type(target) is not int or target not in (-100, token) for token, target in zip(ids, labels)):
                raise ValueError(f"Invalid supervised labels at {label}")
            active = sum(target != -100 for target in labels)
            if not active or active != row.get("supervised_tokens") or labels[0] != -100:
                raise ValueError(f"Invalid response-only supervision at {label}")
            if ids[-1] != eos or labels[-1] != eos or ids.count(eos) != 1:
                raise ValueError(f"Each episode must supervise exactly one terminal EOS at {label}")
            split_origins[origin] += 1
            origins[origin] += 1
            supervised += active
            max_length = max(max_length, len(ids))
        aether_key = "aether_train" if split == "train" else "aether_validation"
        if split_origins["aethersearch"] != counts.get(aether_key):
            raise ValueError(f"AetherSearch {split} count differs from the manifest")
        datasets[split] = PreparedEpisodeDataset(rows)
        stats[split] = {"episodes": len(rows), "by_origin": dict(split_origins),
                        "supervised_tokens": supervised, "max_length": max_length}
    if origins["deepseek_new100"] != 100:
        raise ValueError("Actual prepared token rows do not contain exactly 100 new episodes")
    if stats["train"]["by_origin"].get("deepseek_new100") != 100 or stats["validation"]["by_origin"].get("deepseek_new100", 0):
        raise ValueError("All 100 new episodes must be in the training split")
    return datasets, manifest, sha256_file(manifest_path), stats


def tokenizer_fingerprint(tokenizer):
    return {
        "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        # Match sft_length.tokenizer_provenance, including its default JSON
        # whitespace; the compact config fingerprint uses a different format.
        "special_tokens_sha256": hashlib.sha256(json.dumps(
            tokenizer.special_tokens_map, ensure_ascii=False, sort_keys=True,
            default=str).encode()).hexdigest(),
        "eos_token_id": tokenizer.eos_token_id,
    }


@contextmanager
def output_lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.training.lock"
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another training process already owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def select_checkpoint(output, requested):
    if requested is None:
        return None
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt",
                "adapter_config.json", "training_binding.json",
                "full_tied_embedding.safetensors", "full_tied_embedding.json")

    def complete(path):
        return (path.is_dir() and all((path / name).is_file() for name in required)
                and any((path / name).is_file() for name in (
                    "adapter_model.safetensors", "adapter_model.bin")))

    if requested == "latest":
        candidates = []
        for path in output.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if suffix.isdigit() and complete(path):
                candidates.append((int(suffix), path))
        if not candidates:
            raise ValueError("No complete resumable checkpoint exists in output_dir")
        checkpoint = max(candidates)[1]
    else:
        checkpoint = Path(requested).expanduser().resolve()
    if checkpoint.parent != output or not checkpoint.is_dir():
        raise ValueError("Resume checkpoint must be a checkpoint directory directly inside output_dir")
    if not all((checkpoint / name).is_file() for name in required):
        raise ValueError("Resume requires adapter, trainer, optimizer, scheduler, and binding checkpoint state")
    if not ((checkpoint / "adapter_model.safetensors").is_file()
            or (checkpoint / "adapter_model.bin").is_file()):
        raise ValueError("Resume checkpoint is missing adapter weights")
    return str(checkpoint)


def run_training(config, datasets, prepared_manifest, prepared_sha, stats, requested_resume):
    # Imports are deliberately deferred: --help requires no training runtime.
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments, set_seed)

    rank, local_rank, world_size = check_launch(config)
    if torch.cuda.device_count() != world_size:
        raise ValueError("CUDA sees a different number of devices from the configured UUID whitelist")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU/runtime does not support BF16")
    if "3090" not in torch.cuda.get_device_name(local_rank):
        raise ValueError("This server recipe only permits the selected RTX 3090 devices")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    # Generation has unequal lengths; coordinate its results on a separate CPU
    # group rather than leaving an NCCL collective waiting for retrieval.
    reward_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
    output = Path(config["output_dir"])
    lock = output_lock(output) if rank == 0 else None
    owned_wandb_run = None
    wandb_module = None
    lock_entered = False
    owned_manifest = False
    succeeded = False
    try:
        set_seed(config["seed"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True,
                                                  local_files_only=True, trust_remote_code=False)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required by the prepared-data fingerprint")
        actual_tokenizer = tokenizer_fingerprint(tokenizer)
        if any(prepared_manifest["tokenizer"].get(name) != value for name, value in actual_tokenizer.items()):
            raise ValueError("The model tokenizer differs from the tokenizer used to prepare labels")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        versions = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "tokenizers", "peft", "accelerate", "pyarrow", "numpy", "wandb")}
        model_files = {}
        for name in ("config.json", "model.safetensors.index.json"):
            path = Path(config["model_path"]) / name
            if path.is_file():
                model_files[name] = sha256_file(path)
        binding = {
            "configuration": config,
            "configuration_sha256": canonical_hash(config),
            "prepared_manifest_sha256": prepared_sha,
            "prepared_files": prepared_manifest["files"],
            "training_script_sha256": sha256_file(Path(__file__)),
            "support_code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in (
                "full_embedding_lora.py", "sft_reward_monitor.py", "search_sft_protocol.py", "qa_metrics.py")},
            "tokenizer": actual_tokenizer,
            "base_model_metadata_sha256": model_files,
            "runtime_versions": versions,
            "world_size": world_size,
            "global_batch_size": world_size * config["per_device_batch"] * config["gradient_accumulation"],
        }
        binding_sha = canonical_hash(binding)
        payload = [None]
        if rank == 0:
            try:
                lock.__enter__()
                lock_entered = True
                checkpoint = select_checkpoint(output, requested_resume)
                manifest_path = output / "run_manifest.json"
                if checkpoint:
                    existing = read_json(manifest_path)
                    if existing.get("binding_sha256") != binding_sha:
                        raise ValueError("Resume refused: recipe, data, code, GPU mapping, tokenizer, or runtime changed")
                    wandb_id = existing.get("wandb_run_id")
                    if not isinstance(wandb_id, str) or not wandb_id:
                        raise ValueError("Resume manifest is missing its W&B run ID")
                    checkpoint_binding = read_json(Path(checkpoint) / "training_binding.json")
                    if checkpoint_binding != {"binding_sha256": binding_sha, "wandb_run_id": wandb_id}:
                        raise ValueError("Resume checkpoint does not belong to this exact training run")
                else:
                    if output.exists() and any(output.iterdir()):
                        raise ValueError("output_dir is nonempty; choose a fresh run directory or resume a valid checkpoint")
                    output.mkdir(parents=True, exist_ok=True)
                    wandb_id = uuid.uuid4().hex[:8]
                    existing = {"format_version": 1, "created_at_utc": utc_now(),
                                "binding": binding, "binding_sha256": binding_sha,
                                "wandb_run_id": wandb_id, "data_statistics": stats,
                                "status": "initialized", "base_weight_hash_scope": "metadata/index only"}
                    write_json(manifest_path, existing)
                if os.environ.get("WANDB_RUN_ID", wandb_id) != wandb_id:
                    raise ValueError("WANDB_RUN_ID conflicts with this run's saved W&B ID")
                existing["last_started_at_utc"] = utc_now()
                existing["resume_checkpoint"] = checkpoint
                existing["status"] = "starting"
                write_json(manifest_path, existing)
                owned_manifest = True
                payload[0] = {"checkpoint": checkpoint, "wandb_id": wandb_id}
            except Exception as exc:
                payload[0] = {"error": str(exc)}
        dist.broadcast_object_list(payload, src=0)
        if "error" in payload[0]:
            raise ValueError(payload[0]["error"])
        checkpoint, wandb_id = payload[0]["checkpoint"], payload[0]["wandb_id"]

        # Disable optional code/model uploads and gradient watching. Only scalars
        # and the explicit non-text metadata below are sent to W&B.
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB_DISABLE_CODE"] = "true"
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation=config["attn_implementation"],
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=config["lora_rank"],
            lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"], bias="none",
        ))
        from full_embedding_lora import configure_full_embedding_training, FullEmbeddingTrainer
        embedding_info = configure_full_embedding_training(model)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        if trainable <= 0 or trainable >= total:
            raise ValueError("Expected q/v LoRA plus a trainable tied embedding and frozen decoder base weights")
        if config["smoke_train_examples"]:
            # Include longest sequences to exercise actual peak-memory pressure.
            datasets["train"].items.sort(key=lambda row: len(row["input_ids"]), reverse=True)
            datasets["train"].items = datasets["train"].items[:config["smoke_train_examples"]]
        if config["smoke_eval_examples"]:
            datasets["validation"].items = datasets["validation"].items[:config["smoke_eval_examples"]]
        steps_per_epoch = math.ceil(math.ceil(len(datasets["train"]) / (world_size * config["per_device_batch"]))
                                    / config["gradient_accumulation"])
        max_steps = config["max_steps"] if config["max_steps"] > 0 else math.ceil(config["epochs"] * steps_per_epoch)
        training_args = TrainingArguments(
            output_dir=str(output), run_name=config["run_name"],
            num_train_epochs=config["epochs"], max_steps=max_steps, learning_rate=config["lr"],
            adam_beta1=config["betas"][0], adam_beta2=config["betas"][1],
            weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"],
            max_grad_norm=config["max_grad_norm"], lr_scheduler_type="cosine",
            optim="adamw_torch", per_device_train_batch_size=config["per_device_batch"],
            per_device_eval_batch_size=config["eval_per_device_batch"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            bf16=True, fp16=False, tf32=config["tf32"],
            gradient_checkpointing=config["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
            label_names=["labels"], remove_unused_columns=False,
            average_tokens_across_devices=True, prediction_loss_only=True,
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
            logging_steps=1, logging_first_step=True, eval_on_start=True,
            save_total_limit=config["save_total_limit"], save_safetensors=True,
            save_only_model=False, load_best_model_at_end=False,
            metric_for_best_model="eval_loss", greater_is_better=False,
            dataloader_num_workers=config["dataloader_num_workers"],
            dataloader_pin_memory=True, dataloader_drop_last=False,
            seed=config["seed"], data_seed=config["seed"],
            report_to=["wandb"], push_to_hub=False,
        )
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, padding=True,
                                        label_pad_token_id=-100, pad_to_multiple_of=8)
        from transformers import TrainerCallback

        class SaveBindingCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                if state.is_world_process_zero:
                    write_json(Path(args.output_dir) / f"checkpoint-{state.global_step}" / "training_binding.json",
                               {"binding_sha256": binding_sha, "wandb_run_id": wandb_id})

        class SupervisedPositionTrainer(FullEmbeddingTrainer):
            """Avoid computing vocabulary logits at ignored prompt/retrieval positions."""
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs["labels"]
                positions = (labels[:, 1:] != -100).any(dim=0).nonzero(as_tuple=True)[0]
                if not positions.numel():
                    raise ValueError("Batch has no supervised causal targets")
                selected_labels = labels[:, positions + 1]
                outputs = model(**{key: value for key, value in inputs.items() if key != "labels"},
                                logits_to_keep=positions, use_cache=False)
                loss_sum = torch.nn.functional.cross_entropy(
                    outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                    selected_labels.reshape(-1), ignore_index=-100, reduction="sum")
                if num_items_in_batch is not None:
                    loss = loss_sum / num_items_in_batch
                    if self.args.average_tokens_across_devices:
                        loss = loss * self.accelerator.num_processes
                else:
                    loss = loss_sum / (selected_labels != -100).sum()
                return (loss, {"loss": loss, "logits": outputs.logits}) if return_outputs else loss

            def create_optimizer(self):
                if self.optimizer is None:
                    embedding = self.model.get_input_embeddings().weight
                    other = [p for p in self.model.parameters() if p.requires_grad and p is not embedding]
                    self.optimizer = torch.optim.AdamW([
                        {"params": other, "lr": config["lr"]},
                        {"params": [embedding], "lr": config["embedding_lr"]},
                    ], betas=tuple(config["betas"]), weight_decay=config["weight_decay"], foreach=False)
                return self.optimizer

        trainer = SupervisedPositionTrainer(model=model, args=training_args, train_dataset=datasets["train"],
                          eval_dataset=datasets["validation"], data_collator=collator,
                          processing_class=tokenizer, callbacks=[SaveBindingCallback()])
        # compute_loss handles a global accumulated supervised-token denominator.
        trainer.model_accepts_loss_kwargs = True
        from sft_reward_monitor import run_reward_eval
        reward_file = Path(config["prepared_dir"]) / "reward_monitor.jsonl"
        reward_examples = [json.loads(line) for line in reward_file.read_text().splitlines() if line.strip()]
        if config["reward_examples"] < world_size or len(reward_examples) < config["reward_examples"]:
            raise ValueError("Reward evaluation file has insufficient held-out examples")
        # The prepared file is balanced/interleaved; taking a prefix is only for smoke checks.
        reward_examples = reward_examples[:config["reward_examples"]]
        reward_config = {
            "retriever_url": config["retriever_url"], "max_searches": config["reward_max_searches"],
            "max_new_tokens": config["reward_max_new_tokens"],
            "max_context_tokens": config["reward_max_context_tokens"],
            "topk": 3, "passage_chars": 1800, "format_weight": 0.0,
        }

        class GeneratedRewardCallback(TrainerCallback):
            def __init__(self):
                self.last_step = None

            def evaluate_rewards(self, state):
                step = state.global_step
                if step == self.last_step:
                    return
                result = None
                try:
                    raw_model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
                    shard = reward_examples[rank * len(reward_examples) // world_size:(rank + 1) * len(reward_examples) // world_size]
                    path = output / "reward_evaluations" / f"step-{step:06d}" / f"rank-{rank}.jsonl"
                    torch.cuda.empty_cache()
                    metrics = run_reward_eval(raw_model, tokenizer, shard, reward_config, path)
                    result = {"ok": True, "count": len(shard), "metrics": metrics,
                              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
                except Exception as exc:
                    result = {"ok": False, "error_type": type(exc).__name__, "rank": rank}
                    import traceback
                    traceback.print_exc()
                gathered = [None] * world_size
                dist.all_gather_object(gathered, result, group=reward_group)
                if any(not item["ok"] for item in gathered):
                    raise RuntimeError(f"Generated reward evaluation failed: {[x for x in gathered if not x['ok']]}")
                # All shard sizes are equal for the full recipe; weighted means
                # also cover smaller smoke configurations.
                total_count = sum(item["count"] for item in gathered)
                keys = set(gathered[0]["metrics"]) - {"reward/num_examples", "reward/elapsed_seconds"}
                metrics = {key: sum(item["metrics"][key] * item["count"] for item in gathered) / total_count
                           for key in keys if all(isinstance(item["metrics"].get(key), (int, float)) for item in gathered)}
                metrics["reward/num_examples"] = total_count
                metrics["reward/elapsed_seconds"] = max(item["metrics"]["reward/elapsed_seconds"] for item in gathered)
                metrics["reward/evaluated_examples"] = total_count
                metrics["reward/global_step"] = step
                metrics["memory/peak_allocated_mib_max"] = max(item["peak_allocated_mib"] for item in gathered)
                metrics["memory/peak_reserved_mib_max"] = max(item["peak_reserved_mib"] for item in gathered)
                if rank == 0:
                    write_json(output / "reward_evaluations" / f"step-{step:06d}" / "metrics.json", metrics)
                    print(json.dumps({"generated_reward": metrics}, ensure_ascii=False), flush=True)
                trainer.log(metrics)
                if step > 0:
                    # A small diagnostic confirms that optimizer updates can
                    # separate the formerly identical think-token rows.
                    weight = trainer.model.get_input_embeddings().weight
                    with torch.no_grad():
                        row_difference = (weight[151667] - weight[151668]).abs().max().item()
                        projected_difference = (weight[151667].to(torch.bfloat16)
                                                - weight[151668].to(torch.bfloat16)).abs().max().item()
                    trainer.log({"embedding/think_row_max_abs_difference_fp32": row_difference,
                                 "embedding/think_row_max_abs_difference_bf16": projected_difference})
                self.last_step = step
                torch.cuda.empty_cache()

            def on_train_begin(self, args, state, control, **kwargs):
                if config["reward_on_start"] and state.global_step == 0:
                    self.evaluate_rewards(state)

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % config["reward_every_steps"] == 0 or state.global_step == state.max_steps:
                    self.evaluate_rewards(state)

        trainer.add_callback(GeneratedRewardCallback())
        wandb_payload = [None]
        if rank == 0:
            try:
                import wandb
                wandb_module = wandb
                if wandb.run is not None:
                    raise ValueError("A W&B run is already active in this training process")
                # HF's W&B callback also writes model/generation config keys
                # such as max_length. Keep our actual training recipe separate.
                safe_config = {
                    "recipe": {name: config[name] for name in DEFAULTS if name != "wandb_entity"},
                    "training_max_length": config["max_length"],
                }
                safe_config.update({"base_model": Path(config["model_path"]).name,
                                    "configuration_sha256": binding["configuration_sha256"],
                                    "prepared_manifest_sha256": prepared_sha,
                                    "training_script_sha256": binding["training_script_sha256"],
                                    "binding_sha256": binding_sha, "data_statistics": stats,
                                    "prepared_counts": prepared_manifest["counts"],
                                    "selected_gpu_uuids": config["selected_gpu_uuids"],
                                    "world_size": world_size, "global_batch_size": binding["global_batch_size"],
                                    "trainable_parameters": trainable, "total_parameters": total,
                                    "embedding_training": embedding_info,
                                    "reward_definition": "total_reward=official Search-R1 normalized answer EM; format_reward is separate, not added to SFT loss",
                                    "runtime_versions": versions, "training_unit": "complete_episode"})
                owned_wandb_run = wandb.init(
                    project=config["wandb_project"], entity=config["wandb_entity"],
                    name=config["run_name"], id=wandb_id,
                    resume="must" if checkpoint else "never", mode="online",
                    dir=str(output), config=safe_config,
                    job_type="preflight" if config["smoke_train_examples"] else "sft",
                    settings=wandb.Settings(init_timeout=120, disable_git=True),
                )
                if owned_wandb_run is None or owned_wandb_run.id != wandb_id:
                    raise ValueError("W&B did not initialize this run's required online ID")
                record = read_json(output / "run_manifest.json")
                record.update({"status": "training", "wandb_url": owned_wandb_run.url,
                               "trainable_parameters": trainable, "total_parameters": total})
                write_json(output / "run_manifest.json", record)
                print(json.dumps({"run_name": config["run_name"], "wandb_url": owned_wandb_run.url,
                                  "episodes": prepared_manifest["counts"], "trainable_parameters": trainable,
                                  "global_batch_size": binding["global_batch_size"]}, ensure_ascii=False), flush=True)
                wandb_payload[0] = {"ok": True}
            except Exception:
                # Do not echo service/client exceptions: authentication details
                # belong in the user's secure login flow, never our run manifest.
                wandb_payload[0] = {"error": "W&B online initialization failed; verify wandb login and network connectivity"}
        dist.broadcast_object_list(wandb_payload, src=0)
        if "error" in wandb_payload[0]:
            raise RuntimeError(wandb_payload[0]["error"])
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # Final train-time epoch evaluation has already run on every rank.
        # Avoid a second evaluation collective during rank-dependent teardown.
        final_metrics = next((row for row in reversed(trainer.state.log_history) if "eval_loss" in row), {})
        trainer.save_model(str(output / "final_adapter"))
        trainer.save_state()
        if rank == 0:
            tokenizer.save_pretrained(output / "final_adapter")
            summary = {"completed_at_utc": utc_now(), "global_step": trainer.state.global_step,
                       "train_metrics": train_result.metrics, "final_eval_metrics": final_metrics,
                       "best_checkpoint": trainer.state.best_model_checkpoint,
                       "best_eval_loss": trainer.state.best_metric,
                       "final_adapter": str(output / "final_adapter"),
                       "binding_sha256": binding_sha, "wandb_run_id": wandb_id,
                       "trainable_parameters": trainable, "total_parameters": total}
            write_json(output / "training_summary.json", summary)
            record = read_json(output / "run_manifest.json")
            record.update({"status": "complete", "completed_at_utc": summary["completed_at_utc"],
                           "global_step": trainer.state.global_step})
            write_json(output / "run_manifest.json", record)
        dist.barrier()
        succeeded = True
    except BaseException as exc:
        if rank == 0 and owned_manifest:
            try:
                record = read_json(output / "run_manifest.json")
                record.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               "stopped_at_utc": utc_now()})
                # Keep service exceptions out of persisted metadata.
                write_json(output / "run_manifest.json", record)
            except Exception:
                pass
        raise
    finally:
        try:
            if rank == 0 and owned_wandb_run is not None and wandb_module is not None:
                if wandb_module.run is owned_wandb_run:
                    owned_wandb_run.finish(exit_code=0 if succeeded else 1)
        finally:
            if rank == 0 and lock_entered:
                lock.__exit__(None, None, None)
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local JSON recipe; no secrets")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", default=None,
                        help="Resume an exact matching checkpoint path, or use 'latest'")
    args = parser.parse_args()
    config = load_config(args.config)
    check_launch(config)
    datasets, manifest, manifest_sha, statistics = load_prepared(config)
    run_training(config, datasets, manifest, manifest_sha, statistics, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
