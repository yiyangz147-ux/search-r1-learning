#!/usr/bin/env python3
"""Bind a held-out question set to LoRA + full tied-embedding RAG evaluation.

This command is CPU-only. It fingerprints the final checkpoint, original Base,
question/gold records, training inputs and evaluator before publishing a run.
Workers only need stat checks for large weights after this one full SHA pass.
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

from prepare_lora_sft_data_v2 import QuestionIndex, normalized_question
from qa_metrics import normalize_answer
from sft_reward_monitor import validate_config as validate_rollout_config


SUBSET = "previous-Base-question-set-full-embedding-SFT-RAG"
PATH_FIELDS = ("data_file", "model_path", "adapter_path", "training_prepared_dir", "output_dir")
CODE_FILES = ("prepare_full_embedding_eval.py", "report_full_embedding_eval.py",
              "evaluate_full_embedding_rag.py", "sft_reward_monitor.py", "search_sft_protocol.py",
              "qa_metrics.py", "full_embedding_lora.py", "prepare_lora_sft_data_v2.py",
              "prepare_data.py", "sft_length.py")
OPTIONAL_CODE_FILES = ("run_full_embedding_eval.py",)
LARGE_BYTES = 32 * 1024 * 1024
DEFAULTS = {"seed": 42, "batch_size": 8, "num_workers": 4, "max_new_tokens": 512,
            "max_context_tokens": 8192, "max_total_generated_tokens": 2048, "max_turns": 6,
            "max_invalid_actions": 2, "max_searches": 3, "topk": 3, "passage_chars": 1800,
            "retriever_url": "http://127.0.0.1:8000/retrieve", "retrieval_timeout_seconds": 180,
            "retrieval_retries": 2, "require_no_training_overlap": True}


def fail(message):
    raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def read_jsonl(path, tolerate_partial_tail=False):
    content = Path(path).read_bytes()
    if tolerate_partial_tail and content and not content.endswith(b"\n"):
        boundary = content.rfind(b"\n")
        content = content[:boundary + 1] if boundary >= 0 else b""
    rows = []
    for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique_object)
        if not isinstance(row, dict):
            fail(f"Expected a JSON object: {path}:{number}")
        rows.append(row)
    return rows


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path):
    path = Path(path).resolve()
    raw = read_json(path)
    if not isinstance(raw, dict):
        fail("Configuration must be a JSON object")
    config = {**DEFAULTS, **raw}
    for field in PATH_FIELDS:
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            fail(f"Configuration requires {field}")
        candidate = Path(value).expanduser()
        config[field] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    if config.get("reward_monitor_file") is not None:
        candidate = Path(config["reward_monitor_file"]).expanduser()
        config["reward_monitor_file"] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    for field in ("expected_count", "batch_size", "num_workers"):
        if type(config.get(field)) is not int or config[field] < 1:
            fail(f"{field} must be a positive integer")
    if config["num_workers"] > 4:
        fail("This evaluation supports at most four approved GPUs")
    uuids = config.get("selected_gpu_uuids")
    if (not isinstance(uuids, list) or len(uuids) != config["num_workers"]
            or len(set(uuids)) != len(uuids) or any(not isinstance(x, str) or not x.startswith("GPU-") for x in uuids)):
        fail("selected_gpu_uuids must contain one distinct UUID per worker")
    if type(config["seed"]) is not int:
        fail("seed must be an integer")
    if type(config["require_no_training_overlap"]) is not bool:
        fail("require_no_training_overlap must be boolean")
    expected = config.get("expected_data_sha256")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)):
        fail("expected_data_sha256 must bind the selected source file")
    sources = config.get("expected_source_counts")
    if sources is not None and (not isinstance(sources, dict) or set(sources) - {"nq", "hotpotqa"}
                                or any(type(n) is not int or n < 0 for n in sources.values())
                                or sum(sources.values()) != config["expected_count"]):
        fail("expected_source_counts must describe the complete selected set")
    validate_rollout_config(config)
    output = Path(config["output_dir"])
    for field in PATH_FIELDS[:-1]:
        source = Path(config[field])
        if output == source or output in source.parents or source in output.parents:
            fail(f"output_dir must be separate from {field}")
    return config


def normalize_examples(rows):
    result, ids = [], set()
    for index, original in enumerate(rows):
        row = dict(original)
        sid = row.get("sample_id", row.get("id"))
        if not isinstance(sid, str) or not sid.strip() or sid in ids:
            fail(f"Missing or duplicate sample ID at row {index}")
        if row.get("id", sid) != sid:
            fail(f"Conflicting id/sample_id: {sid}")
        question = row.get("question")
        gold = row.get("golden_answers", row.get("gold_answers"))
        source = row.get("data_source")
        if not isinstance(question, str) or not question.strip():
            fail(f"Missing question: {sid}")
        if (not isinstance(gold, list) or not gold
                or any(not isinstance(answer, str) or not answer.strip() for answer in gold)):
            fail(f"Missing nonempty gold aliases: {sid}")
        if row.get("gold_answers", gold) != gold:
            fail(f"Conflicting golden_answers/gold_answers: {sid}")
        if source not in ("nq", "hotpotqa"):
            fail(f"Unsupported question data_source: {sid}")
        ids.add(sid)
        row.update(sample_id=sid, id=sid, golden_answers=gold, gold_answers=gold)
        result.append(row)
    return result


def selected_hash(rows):
    return digest_text("".join(canonical(row) + "\n" for row in rows))


def load_examples(config):
    if sha256(config["data_file"]) != config["expected_data_sha256"]:
        fail("Question source SHA256 differs from expected_data_sha256")
    rows = normalize_examples(read_jsonl(config["data_file"]))
    if len(rows) != config["expected_count"]:
        fail("Question count differs from expected_count")
    if config.get("expected_source_counts") is not None:
        if dict(Counter(row["data_source"] for row in rows)) != config["expected_source_counts"]:
            fail("Question source quotas differ from expected_source_counts")
    return rows


def record_asset(path, assets, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        fail(f"Required asset is missing: {path}")
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        fail(f"An input changed while fingerprinting: {path}")
    if expected is not None and digest != expected:
        fail(f"Asset checksum mismatch: {path}")
    assets[str(path)] = {"sha256": digest, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                         "large": after.st_size >= LARGE_BYTES}
    return digest


def bind_model(config, assets):
    model, adapter = Path(config["model_path"]), Path(config["adapter_path"])
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        record_asset(model / filename, assets)
    for directory in (model, adapter):
        for filename in ("generation_config.json", "special_tokens_map.json", "added_tokens.json",
                         "vocab.json", "merges.txt", "chat_template.jinja"):
            if (directory / filename).is_file():
                record_asset(directory / filename, assets)
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        record_asset(index_path, assets)
        shards = set(read_json(index_path)["weight_map"].values())
        for filename in sorted(shards):
            path = (model / filename).resolve()
            if not path.is_relative_to(model.resolve()):
                fail("Base weight index contains an escaping filename")
            record_asset(path, assets)
    else:
        record_asset(model / "model.safetensors", assets)
    record_asset(adapter / "adapter_config.json", assets)
    adapter_config = read_json(adapter / "adapter_config.json")
    if adapter_config.get("peft_type") != "LORA":
        fail("The selected checkpoint is not a LoRA adapter")
    record_asset(adapter / "adapter_model.safetensors", assets)
    metadata_path = adapter / "full_tied_embedding.json"
    record_asset(metadata_path, assets)
    metadata = read_json(metadata_path)
    model_config = read_json(model / "config.json")
    if (metadata.get("format_version") != 1 or metadata.get("weight_file") != "full_tied_embedding.safetensors"
            or metadata.get("tensor_name") != "shared_embedding_weight" or metadata.get("master_dtype") != "float32"
            or metadata.get("tied") is not True or model_config.get("tie_word_embeddings") is not True
            or metadata.get("shape") != [model_config["vocab_size"], model_config["hidden_size"]]):
        fail("Full tied-embedding metadata is incompatible with the selected Base")
    weight = adapter / metadata["weight_file"]
    record_asset(weight, assets, metadata["sha256"])
    if weight.stat().st_size != metadata["bytes"]:
        fail("Full tied-embedding byte count differs")
    return metadata


def bind_training(config, examples, assets):
    import pyarrow.parquet as pq

    prepared = Path(config["training_prepared_dir"])
    manifest_path = prepared / "manifest.json"
    record_asset(manifest_path, assets)
    manifest = read_json(manifest_path)
    if manifest.get("complete") is not True or manifest.get("counts", {}).get("new_episodes") != 100:
        fail("Training manifest must bind the complete AetherSearch + new100 mixture")
    for path, expected in manifest.get("source_hashes", {}).items():
        record_asset(path, assets, expected)
    overlap = {}
    for filename, metadata in manifest.get("files", {}).items():
        path = (prepared / filename).resolve()
        if not path.is_relative_to(prepared.resolve()):
            fail("Training manifest contains an escaping path")
        record_asset(path, assets, metadata["sha256"])
        if path.stat().st_size != metadata["bytes"]:
            fail(f"Training prepared byte count differs: {filename}")
    for label, filename, count_key in (("train", "train.parquet", "train"),
                                       ("validation", "val.parquet", "validation")):
        if filename not in manifest.get("files", {}):
            fail(f"Training prepared manifest does not bind {filename}")
        rows = pq.read_table(prepared / filename, columns=["sample_id", "question"]).to_pylist()
        if len(rows) != manifest["counts"][count_key] or len(rows) != manifest["files"][filename]["rows"]:
            fail(f"Training prepared row count differs: {filename}")
        index = QuestionIndex(rows)
        collisions = [{"evaluation_sample_id": row["sample_id"], "matches": matches}
                      for row in examples if (matches := index.matches(row))]
        overlap[label] = {"source_rows": len(rows), "overlapping_evaluation_questions": len(collisions),
                          "collisions": collisions}
        if collisions and config["require_no_training_overlap"]:
            fail(f"Evaluation overlaps SFT {label} at Jaccard >= 0.85: {collisions[0]['evaluation_sample_id']}")
    return overlap


def bind_monitor_subset(config, examples, assets):
    """Match training-monitor questions by content because historical IDs differ."""
    path = config.get("reward_monitor_file")
    if path is None:
        return {"selected": 0, "evaluation_sample_ids": [], "sources": {}}
    record_asset(path, assets)
    monitor = normalize_examples(read_jsonl(path))
    by_question = {}
    for row in examples:
        key = (row["data_source"], normalized_question(row["question"]))
        by_question.setdefault(key, []).append(row)
    selected, mappings = [], []
    for row in monitor:
        matches = by_question.get((row["data_source"], normalized_question(row["question"])), [])
        if len(matches) != 1:
            fail(f"Training reward-monitor question must match exactly one evaluation question: {row['sample_id']}")
        evaluation = matches[0]
        monitor_gold = {normalize_answer(answer, row["data_source"]) for answer in row["golden_answers"]}
        evaluation_gold = {normalize_answer(answer, row["data_source"]) for answer in evaluation["golden_answers"]}
        if monitor_gold != evaluation_gold:
            fail(f"Training reward-monitor gold aliases disagree: {row['sample_id']}")
        selected.append(evaluation["sample_id"])
        mappings.append({"monitor_sample_id": row["sample_id"], "evaluation_sample_id": evaluation["sample_id"]})
    if len(set(selected)) != len(selected) or len(monitor) != config.get("expected_reward_monitor_count", 32):
        fail("Training monitor subset contains duplicate matches or unexpected count")
    return {"selected": len(selected), "evaluation_sample_ids": selected, "mapping": mappings,
            "sources": dict(Counter(row["data_source"] for row in monitor)),
            "matching": "source + normalized exact question + equal normalized gold-alias set",
            "interpretation": "These questions were monitored during training; report their excluded subset separately"}


def verify_run(config_path, *, full_hash=False):
    """Verify before CUDA; return config, immutable manifest, normalized examples.

    Full preparation hashes each large asset once. Normal worker/report checks
    rehash small inputs and ensure large files retain the bound size and mtime.
    ``full_hash=True`` also rehashes every large weight for an end-of-run audit.
    """
    config = load_config(config_path) if not isinstance(config_path, dict) else config_path
    output = Path(config["output_dir"])
    manifest = read_json(output / "run_manifest.json")
    if manifest.get("format_version") != 1 or manifest.get("complete") is not True or manifest.get("subset") != SUBSET:
        fail("Evaluation run manifest is incomplete or incompatible")
    if manifest.get("configuration") != config or manifest.get("config_sha256") != digest_text(canonical(config)):
        fail("Evaluation configuration changed; use a fresh output directory")
    if manifest.get("num_workers") != config["num_workers"]:
        fail("Evaluation worker count changed")
    if not set(CODE_FILES).issubset(manifest.get("code_sha256", {})):
        fail("Evaluation manifest is missing required code bindings")
    for filename, expected in manifest["code_sha256"].items():
        if sha256(Path(__file__).with_name(filename)) != expected:
            fail(f"Evaluation code changed: {filename}")
    assets = manifest.get("asset_sha256")
    if not isinstance(assets, dict) or not assets:
        fail("Evaluation manifest is missing asset bindings")
    for filename, expected in assets.items():
        path = Path(filename)
        stat = path.stat()
        if stat.st_size != expected["bytes"]:
            fail(f"Evaluation asset byte count changed: {filename}")
        if expected["large"] and not full_hash:
            if stat.st_mtime_ns != expected["mtime_ns"]:
                fail(f"Large evaluation asset changed since CPU fingerprinting: {filename}")
        elif sha256(path) != expected["sha256"]:
            fail(f"Evaluation asset checksum changed: {filename}")
    examples = load_examples(config)
    if (manifest.get("selected") != len(examples) or manifest.get("selected_sha256") != selected_hash(examples)
            or manifest.get("sources") != dict(Counter(row["data_source"] for row in examples))):
        fail("Selected question records/order differ from prepared run")
    if sha256(output / "examples.jsonl") != manifest["examples_sha256"]:
        fail("The prepared question snapshot changed")
    if read_jsonl(output / "examples.jsonl") != examples:
        fail("Prepared question snapshot differs from the source")
    return config, manifest, examples


def prepare_run(config_path):
    config = load_config(config_path) if not isinstance(config_path, dict) else config_path
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare_eval.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("Another evaluation preparation is running")
        if (output / "run_manifest.json").exists():
            _, manifest, _ = verify_run(config, full_hash=True)
            print(canonical({"phase": "existing_preparation_verified", "selected": manifest["selected"],
                             "output_dir": str(output)}), flush=True)
            return manifest
        if any(path.name != ".prepare_eval.lock" for path in output.iterdir()):
            fail("Evaluation output is nonempty without a run manifest; use a fresh output directory")
        examples = load_examples(config)
        assets = {}
        record_asset(config["data_file"], assets, config["expected_data_sha256"])
        embedding = bind_model(config, assets)
        overlap = bind_training(config, examples, assets)
        monitor_subset = bind_monitor_subset(config, examples, assets)
        code = {filename: sha256(Path(__file__).with_name(filename)) for filename in CODE_FILES}
        code.update({filename: sha256(Path(__file__).with_name(filename)) for filename in OPTIONAL_CODE_FILES
                     if Path(__file__).with_name(filename).is_file()})
        example_text = "".join(canonical(row) + "\n" for row in examples)
        manifest = {"format_version": 1, "complete": True, "subset": SUBSET,
                    "configuration": config, "config_sha256": digest_text(canonical(config)),
                    "selected": len(examples), "sources": dict(Counter(row["data_source"] for row in examples)),
                    "num_workers": config["num_workers"], "data_sha256": config["expected_data_sha256"],
                    "selected_sha256": selected_hash(examples), "examples_sha256": digest_text(example_text),
                    "examples_file": "examples.jsonl", "asset_sha256": assets, "code_sha256": code,
                    "full_embedding": embedding, "overlap_checks": {"jaccard_threshold": 0.85,
                    "normalization": "NFKC, casefold, remove Unicode punctuation, collapse whitespace",
                    "features": "word 3-grams (whole sequence for shorter questions)", "results": overlap,
                    "limitation": "semantic paraphrases are not guaranteed excluded"},
                    "training_monitor_subset": monitor_subset,
                    "large_weight_verification": "SHA256 all weights once during CPU preparation; size+mtime in workers; optional final full rehash",
                    "gold_answers_sent_to_model": False, "inference_started": False,
                    "prepared_at": datetime.now(timezone.utc).isoformat()}
        # Publish the manifest last. A crash cannot present a partial run as ready.
        (output / "examples.jsonl").write_text(example_text, encoding="utf-8")
        atomic_json(output / "run_manifest.json", manifest)
        verify_run(config)
    print(canonical({"phase": "prepared", "selected": len(examples), "sources": manifest["sources"],
                     "output_dir": str(output), "inference_started": False}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    if args.verify:
        _, manifest, _ = verify_run(args.config, full_hash=args.full_hash)
        print(canonical({"phase": "verified", "selected": manifest["selected"]}), flush=True)
    else:
        prepare_run(args.config)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"评测准备失败：{error}", file=sys.stderr)
        sys.exit(1)
