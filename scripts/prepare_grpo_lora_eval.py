#!/usr/bin/env python3
"""Bind the user's reserved GRPO 2000 to a reproducible LoRA evaluation.

CPU preparation only: no model weights, CUDA, retrieval, or inference requests.
The original GRPO partition and files are never changed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from prepare_lora_sft_data import QuestionIndex


TARGETS = {"nq": 1000, "hotpotqa": 1000}
PATH_FIELDS = ("data_file", "model_path", "adapter_path", "training_prepared_dir", "output_dir")
CODE_FILES = ("prepare_grpo_lora_eval.py", "report_grpo_lora_eval.py", "evaluate_search_lora.py",
              "prepare_data.py", "qa_metrics.py", "prepare_lora_sft_data.py", "sft_length.py")
SUBSET = "user-selected-GRPO-reserved-set"


def fail(message):
    raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path, tolerate_partial_tail=False):
    content = Path(path).read_bytes()
    if tolerate_partial_tail and content and not content.endswith(b"\n"):
        # A reporter can observe a writer midway through a UTF-8 record.
        # Only newline-committed records are safe to decode and summarize.
        boundary = content.rfind(b"\n")
        content = content[:boundary + 1] if boundary >= 0 else b""
    text = content.decode("utf-8")
    lines = text.splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_partial_tail and index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
        if not isinstance(value, dict):
            fail(f"Expected a JSON object: {path}:{index + 1}")
        result.append(value)
    return result


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path):
    """Preserve unknown inference settings in the immutable configuration binding."""
    path = Path(path).resolve()
    config = read_json(path)
    if not isinstance(config, dict):
        fail("Configuration must be a JSON object")
    for field in PATH_FIELDS:
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            fail(f"Configuration requires {field}")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        config[field] = str(candidate.resolve())
    workers = config.get("num_workers", 4)
    if type(workers) is not int or not 1 <= workers <= 4:
        fail("num_workers must be an integer from 1 to 4")
    # Do not insert defaults: configuration remains identical to the worker's input.
    output = Path(config["output_dir"])
    for field in PATH_FIELDS[:-1]:
        source = Path(config[field])
        if output == source or output in source.parents or source in output.parents:
            fail(f"Evaluation output_dir must be separate from {field}")
    return config


def load_grpo(path):
    rows = read_jsonl(path)
    if len(rows) != 2000:
        fail("The selected GRPO reserved set must contain exactly 2000 rows")
    ids = set()
    for index, row in enumerate(rows):
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in ids:
            fail(f"Missing or duplicate sample_id at row {index}")
        ids.add(sid)
        if row.get("partition") != "grpo" or row.get("source_split") != "train":
            fail(f"Expected unchanged partition=grpo and source_split=train: {sid}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            fail(f"Invalid question: {sid}")
        answers = row.get("golden_answers")
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) or not answer.strip() for answer in answers)):
            fail(f"Invalid golden answer aliases: {sid}")
    if dict(Counter(row.get("data_source") for row in rows)) != TARGETS:
        fail("GRPO source quotas must be exactly NQ 1000 and HotpotQA 1000")
    return rows


def selected_hash(rows):
    """Hash canonical complete original records in their unchanged source order."""
    return digest_text("".join(canonical(row) + "\n" for row in rows))


def record_asset(path, assets, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        fail(f"Required evaluation asset is missing: {path}")
    value = sha256(path)
    if expected is not None and value != expected:
        fail(f"Evaluation asset checksum mismatch: {path}")
    assets[str(path)] = value
    return value


def model_assets(config, assets):
    model, adapter = Path(config["model_path"]), Path(config["adapter_path"])
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        record_asset(model / filename, assets)
    optional = ("model.safetensors.index.json", "generation_config.json", "special_tokens_map.json",
                "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")
    for filename in optional:
        if (model / filename).is_file():
            record_asset(model / filename, assets)
    record_asset(adapter / "adapter_config.json", assets)
    weights = [adapter / filename for filename in ("adapter_model.safetensors", "adapter_model.bin")
               if (adapter / filename).is_file()]
    if not weights:
        fail("Adapter directory has no adapter_model.safetensors or adapter_model.bin")
    for path in weights:
        record_asset(path, assets)
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "added_tokens.json", "generation_config.json", "vocab.json", "merges.txt"):
        if (adapter / filename).is_file():
            record_asset(adapter / filename, assets)
    adapter_config = read_json(adapter / "adapter_config.json")
    if adapter_config.get("peft_type") != "LORA":
        fail("The selected checkpoint is not a LoRA adapter")


def prepared_overlap(config, evaluation, assets, data_sha):
    import pyarrow.parquet as pq

    prepared = Path(config["training_prepared_dir"])
    manifest_path = prepared / "manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha = record_asset(manifest_path, assets)
    if manifest.get("complete") is not True or manifest.get("counts", {}).get("new_episodes") != 100:
        fail("Training prepared manifest must bind the completed AetherSearch + new100 mixture")
    if data_sha not in manifest.get("source_hashes", {}).values():
        fail("This exact GRPO file was not recorded as a protected source in SFT preparation")
    files = manifest.get("files", {})
    for required in ("train.parquet", "val.parquet"):
        if required not in files:
            fail(f"Training prepared manifest does not bind {required}")
    for filename, metadata in files.items():
        path = (prepared / filename).resolve()
        if not path.is_relative_to(prepared.resolve()):
            fail("Training prepared manifest contains an escaping path")
        record_asset(path, assets, metadata["sha256"])
        if path.stat().st_size != metadata["bytes"]:
            fail(f"Training prepared byte count differs: {filename}")
    partitions = {}
    for split, filename, count_field in (("train", "train.parquet", "train"),
                                          ("validation", "val.parquet", "validation")):
        values = pq.read_table(prepared / filename, columns=["sample_id", "question", "data_origin"]).to_pylist()
        if len(values) != manifest["counts"][count_field] or len(values) != files[filename]["rows"]:
            fail(f"Training prepared row count differs: {filename}")
        partitions[split] = values
    new_train = [row for row in partitions["train"] if row["data_origin"] == "deepseek_new100"]
    if len(new_train) != 100:
        fail("Prepared training data does not contain exactly 100 new teacher episodes")
    accepted_paths = [Path(path) for path in manifest.get("source_hashes", {}) if Path(path).name == "accepted.jsonl"]
    if len(accepted_paths) != 1:
        fail("Training manifest must unambiguously identify the original new100 accepted.jsonl")
    original = accepted_paths[0]
    record_asset(original, assets, manifest["source_hashes"][str(original)])
    new100 = read_jsonl(original)
    if len(new100) != 100 or any(row.get("status") != "accepted" for row in new100):
        fail("Original new100 source is not a complete accepted 100-episode file")
    new_questions = {row["sample_id"]: row["question"] for row in new100}
    if new_questions != {row["sample_id"]: row["question"] for row in new_train}:
        fail("Original new100 questions differ from their prepared training rows")
    groups = {**partitions, "original_new100": new100}
    overlap_counts = {}
    for label, values in groups.items():
        index = QuestionIndex(values)
        collisions = []
        for row in evaluation:
            matches = index.matches(row)
            if matches:
                collisions.append({"evaluation_sample_id": row["sample_id"], "matches": matches})
        overlap_counts[label] = {"source_rows": len(values), "overlapping_evaluation_questions": len(collisions)}
        if collisions:
            fail(f"GRPO evaluation overlaps {label} at normalized/word-3gram Jaccard >= 0.85: {collisions[0]['evaluation_sample_id']}")
    return manifest_sha, overlap_counts


def verify_run(config, manifest):
    """Used by periodic reporting; no model or inference runtime is imported."""
    if manifest.get("complete") is not True or manifest.get("subset") != SUBSET:
        fail("Evaluation preparation manifest is incomplete or is a different subset")
    if manifest.get("selected") != 2000 or manifest.get("sources") != TARGETS:
        fail("Evaluation manifest does not bind the expected 2000-question source quotas")
    if manifest.get("num_workers") != config.get("num_workers", 4):
        fail("Evaluation worker count changed")
    if not set(CODE_FILES).issubset(manifest.get("code_sha256", {})) or not manifest.get("asset_sha256"):
        fail("Evaluation manifest is missing code/asset checksum bindings")
    if manifest.get("configuration") != config or manifest.get("config_sha256") != digest_text(canonical(config)):
        fail("Evaluation configuration changed; use a fresh evaluation output directory")
    if manifest.get("data_sha256") != sha256(config["data_file"]):
        fail("Original GRPO evaluation file changed")
    for filename, expected in manifest.get("code_sha256", {}).items():
        if sha256(Path(__file__).with_name(filename)) != expected:
            fail(f"Evaluation code changed: {filename}")
    for path, expected in manifest.get("asset_sha256", {}).items():
        if sha256(path) != expected:
            fail(f"An evaluation/model/training asset changed: {path}")


def prepare(config):
    data_sha = sha256(config["data_file"])
    data = load_grpo(config["data_file"])
    assets = {}
    record_asset(Path(config["data_file"]), assets, data_sha)
    model_assets(config, assets)
    prepared_sha, overlap = prepared_overlap(config, data, assets, data_sha)
    code = {filename: sha256(Path(__file__).with_name(filename)) for filename in CODE_FILES}
    manifest = {"format_version": 1, "complete": True, "subset": SUBSET,
                "configuration": config, "config_sha256": digest_text(canonical(config)),
                "data_sha256": data_sha, "selected_sha256": selected_hash(data),
                "selected_hash_policy": "SHA256 of compact canonical original JSON records plus newline, in unchanged original order",
                "selected": len(data), "sources": TARGETS, "num_workers": config.get("num_workers", 4),
                "original_partition": "grpo", "original_source_split": "train",
                "official_test_set": False, "gold_answers_sent_to_model": False,
                "asset_sha256": assets, "code_sha256": code,
                "training_prepared_manifest_sha256": prepared_sha,
                "overlap_checks": {"normalization": "Unicode NFKC + casefold + remove Unicode punctuation + collapse whitespace",
                                   "features": "word 3-grams; entire word sequence for shorter questions",
                                   "jaccard_threshold": 0.85, "results": overlap,
                                   "limitation": "semantic paraphrases are not guaranteed excluded"},
                "base_weight_hash_scope": "model config/index and tokenizer assets; full base weight shards are not rehashed",
                "prepared_at": datetime.now(timezone.utc).isoformat(), "inference_started": False}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare_eval.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("Another evaluation preparation is running")
        path = output / "run_manifest.json"
        if path.exists():
            previous = read_json(path)
            comparable = {key: value for key, value in manifest.items() if key != "prepared_at"}
            if {key: value for key, value in previous.items() if key != "prepared_at"} != comparable:
                fail("Existing evaluation output belongs to a different bound run; choose a fresh output_dir")
            verify_run(config, previous)
            phase = "existing_preparation_verified"
        else:
            unexpected = [item for item in output.iterdir() if item.name != ".prepare_eval.lock"]
            if unexpected:
                fail("Evaluation output is nonempty without a run manifest; choose a fresh output_dir")
            for source, expected in assets.items():
                if sha256(source) != expected:
                    fail(f"An input changed during preparation: {source}")
            atomic_json(path, manifest)
            phase = "prepared"
    print(canonical({"阶段": phase, "评测集": SUBSET, "题数": len(data), "来源": TARGETS,
                     "原始分区": "grpo", "输出目录": str(output), "已启动推理": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"评测准备失败：{error}", file=sys.stderr)
        sys.exit(1)
