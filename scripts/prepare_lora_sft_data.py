#!/usr/bin/env python3
"""Prepare CPU-only, complete-episode Base-model LoRA SFT data.

This command never starts generation, an API request, model weights, or training.
The generated run must contain all 100 accepted trajectories before preparation.
Original AetherSearch and teacher files are read and fingerprinted, never edited.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unicodedata

from prepare_data import PROMPT
from qa_metrics import score_answer
from sft_length import tokenizer_provenance


FORMAT_VERSION = 1
TARGETS = {"nq": 50, "hotpotqa": 50}
PATH_FIELDS = ("model_path", "aether_dir", "generated_run_dir", "protected_qa_val",
               "protected_grpo", "prepared_dir")
MASK_POLICY = {
    "prompt_masked": True,
    "information_masked": True,
    "padding_label": -100,
    "terminal_eos_supervised": True,
    "training_unit": "complete_episode",
    "chat_template": False,
    "boundary_straddling_tokens_masked": True,
    "terminal_eos_token_id": 151643,
    "tokenization": "one fast-tokenizer pass over Base prompt and complete episode",
}


def fail(message):
    raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_object)


def read_rows(path, tolerate_incomplete_tail=False):
    """Only the live progress display may ignore a last unfinished JSON line."""
    content = Path(path).read_text(encoding="utf-8")
    lines = content.splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=unique_object)
        except json.JSONDecodeError:
            if tolerate_incomplete_tail and index == len(lines) - 1 and not content.endswith("\n"):
                break
            raise
        if not isinstance(value, dict):
            fail(f"Expected an object at {path}:{index + 1}")
        result.append(value)
    return result


def positive_integer(value, label):
    if type(value) is not int or value < 1:
        fail(f"{label} must be a positive integer")
    return value


def load_configuration(path):
    raw = read_json(path)
    result = {}
    for field in PATH_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            fail(f"Configuration requires {field}")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = Path(path).resolve().parent / candidate
        result[field] = str(candidate.resolve())
    result["max_length"] = positive_integer(raw.get("max_length", 4096), "max_length")
    result["expected_new_episodes"] = positive_integer(raw.get("expected_new_episodes", 100), "expected_new_episodes")
    if result["expected_new_episodes"] != sum(TARGETS.values()):
        fail("This mixture requires exactly 100 new episodes: NQ 50 and HotpotQA 50")
    result["seed"] = raw.get("seed", 42)
    if type(result["seed"]) is not int:
        fail("seed must be an integer")
    output = Path(result["prepared_dir"])
    for field in PATH_FIELDS[:-1]:
        source = Path(result[field])
        if output == source or output in source.parents or source in output.parents:
            fail(f"prepared_dir must be separate from input paths: {field}")
    return result


def live_progress(root, report, running):
    attempts_path = root / "attempts.jsonl"
    attempts = read_rows(attempts_path, tolerate_incomplete_tail=True) if attempts_path.is_file() else []
    accepted = {row.get("sample_id"): row for row in attempts if row.get("status") == "accepted"}
    sources = Counter(row.get("data_source") for row in accepted.values())
    print(canonical({"phase": "waiting_for_generated_100", "checked_at": now(),
                     "generator_running": running, "attempts": len(attempts),
                     "accepted_unique": len(accepted), "accepted_by_source": dict(sources),
                     "targets": TARGETS, "report_complete": report.get("complete", False)}), flush=True)


@contextmanager
def completed_generation(root, wait, timeout, poll):
    """Hold a shared lock through preparation, so a new generator cannot write."""
    started = time.monotonic()
    handle = None
    try:
        while True:
            report_path = root / "generation_report.json"
            lock_path = root / ".generation.lock"
            running = False
            if lock_path.is_file():
                handle = lock_path.open("rb")
                try:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    handle = None
                    running = True
            report = read_json(report_path) if report_path.is_file() else {}
            if report.get("fatal_error"):
                fail("The generation report records a fatal error; resolve generation before training")
            if not running and report.get("complete") is True:
                if handle is None:
                    fail("Completed generation is missing .generation.lock; cannot ensure a stable snapshot")
                yield report
                return
            live_progress(root, report, running)
            if handle is not None:
                handle.close()
                handle = None
            if not wait:
                fail("New 100-episode generation is incomplete or running. Re-run with --wait-for-generated to wait without generating")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                fail("Timed out waiting for the completed 100-episode generation; no training data was published")
            time.sleep(min(poll, remaining, 60))
    finally:
        if handle is not None:
            handle.close()


def record_hash(path, sources, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        fail(f"Required source file is missing: {path}")
    actual = sha256(path)
    if expected is not None and actual != expected:
        fail(f"Source SHA256 mismatch: {path}")
    sources[str(path)] = actual
    return actual


def relative_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        fail(f"Manifest file escapes its dataset directory: {name}")
    return path


def validate_aether(config, sources):
    root = Path(config["aether_dir"])
    manifest_path = root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("complete") is not True or not isinstance(manifest.get("files"), dict):
        fail("AetherSearch dataset manifest is incomplete")
    record_hash(manifest_path, sources)
    for name, entry in manifest["files"].items():
        path = relative_file(root, name)
        record_hash(path, sources, entry["sha256"])
        if path.stat().st_size != entry["bytes"]:
            fail(f"AetherSearch manifest byte count differs: {name}")
    for required in ("train.jsonl", "validation.jsonl"):
        if required not in manifest["files"]:
            fail(f"AetherSearch manifest does not bind {required}")
    adapter = manifest.get("training_adapter")
    if adapter:
        record_hash(Path(adapter["path"]), sources, adapter["sha256"])
    settings = manifest.get("settings", {})
    project = root.parents[2]
    for name, expected in settings.get("code_sha256", {}).items():
        record_hash(relative_file(project / "scripts", name), sources, expected)
    for name, expected in settings.get("tokenizer_assets_sha256", {}).items():
        record_hash(relative_file(Path(config["model_path"]), name), sources, expected)
    verification = root / "verification_report.json"
    if verification.exists():
        audit = read_json(verification)
        record_hash(verification, sources)
        if audit.get("complete") is not True or audit.get("passed") is not True:
            fail("AetherSearch verification report is not complete and passed")
        if audit.get("dataset_manifest_sha256") != sources[str(manifest_path.resolve())]:
            fail("AetherSearch verification report binds a different dataset manifest")
        for name, expected in audit.get("audit_sources_sha256", {}).items():
            record_hash(relative_file(project / "scripts", name), sources, expected)
    partitions = {}
    for split, filename in (("train", "train.jsonl"), ("val", "validation.jsonl")):
        values = read_rows(root / filename)
        expected_count = manifest.get("counts", {}).get("train" if split == "train" else "validation")
        if len(values) != expected_count:
            fail(f"AetherSearch {split} row count differs from manifest")
        partitions[split] = values
    return partitions


def validate_generated(config, report, sources):
    root = Path(config["generated_run_dir"])
    for filename in ("run_manifest.json", "generation_report.json", "accepted.jsonl", "attempts.jsonl"):
        expected = report.get(filename[:-6] + "_sha256") if filename in ("accepted.jsonl", "attempts.jsonl") else None
        if filename in ("accepted.jsonl", "attempts.jsonl") and not expected:
            fail(f"Generation report does not bind {filename}")
        record_hash(root / filename, sources, expected)
    settings = read_json(root / "run_manifest.json")["settings"]
    if settings.get("targets") != TARGETS or report.get("targets") != TARGETS:
        fail("Generated data targets must be exactly NQ 50 and HotpotQA 50")
    if settings.get("student_prompt_sha256") != digest_text(PROMPT):
        fail("Generated student PROMPT differs; restore its recorded prompt before preparation")
    if settings.get("quality_review") is not True or not settings.get("student_tokenizer_provenance"):
        fail("New trajectories require enabled evidence review and student-tokenizer length checks")
    external = settings.get("external_validation", {})
    if external.get("sha256") != sources[str(Path(config["protected_qa_val"]).resolve())]:
        fail("Configured protected QA validation differs from the generation manifest")
    episodes = read_rows(root / "accepted.jsonl")
    attempts = read_rows(root / "attempts.jsonl")
    if len(attempts) != report.get("attempts"):
        fail("Generation report attempt count is stale")
    if len(episodes) != config["expected_new_episodes"]:
        fail("Exactly 100 complete new episodes are required; partial export is not allowed")
    ids = [row.get("sample_id") for row in episodes]
    if len(set(ids)) != len(ids) or any(not isinstance(sid, str) or not sid for sid in ids):
        fail("New episode sample IDs must be nonempty and unique")
    accepted_attempts = {}
    for row in attempts:
        if row.get("status") == "accepted":
            sid = row.get("sample_id")
            if sid in accepted_attempts:
                fail("Multiple accepted attempts exist for one generated sample ID")
            accepted_attempts[sid] = row
    if accepted_attempts != {row["sample_id"]: row for row in episodes}:
        fail("Accepted file differs from the successful attempts in the generation checkpoint")
    counts = dict(Counter(row.get("data_source") for row in episodes))
    if counts != TARGETS or report.get("accepted") != TARGETS:
        fail("New accepted source quotas must be exactly NQ 50 and HotpotQA 50")
    for row in episodes:
        sid = row["sample_id"]
        if (row.get("status") != "accepted" or type(row.get("answer_em")) not in (bool, int)
                or row["answer_em"] != 1):
            fail(f"New episode lacks strict accepted EM: {sid}")
        if row.get("quality_review", {}).get("passed") is not True or row.get("length_check", {}).get("passed") is not True:
            fail(f"New episode lacks passed evidence/length gates: {sid}")
        if not score_answer(row.get("final_answer", ""), row.get("golden_answers", []), row["data_source"])["em"]:
            fail(f"New episode does not pass official source-specific strict EM: {sid}")
    return episodes, settings


def normalized_question(question):
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join("".join(c for c in text if not unicodedata.category(c).startswith("P")).split())


def shingles(question):
    words = normalized_question(question).split()
    if not words:
        fail("A question normalizes to empty text")
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)} if len(words) >= 3 else {tuple(words)}


class QuestionIndex:
    """Exact word-shingle Jaccard verification, with a recall-complete index."""

    def __init__(self, rows):
        self.rows = rows
        self.features = [shingles(row["question"]) for row in rows]
        self.postings = defaultdict(set)
        for index, features in enumerate(self.features):
            for feature in features:
                self.postings[feature].add(index)

    def matches(self, row):
        features = shingles(row["question"])
        candidates = set().union(*(self.postings[item] for item in features))
        result = []
        for index in sorted(candidates):
            other = self.features[index]
            intersection = len(features & other)
            union = len(features | other)
            # Integer arithmetic keeps the >= 0.85 boundary exact.
            if 20 * intersection >= 17 * union:
                result.append({"sample_id": self.rows[index]["sample_id"],
                               "jaccard": intersection / union})
        return result


def protected_questions(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = read_rows(path)
    result = []
    prefix, suffix = PROMPT.split("{question}")
    for index, row in enumerate(rows):
        question = row.get("question")
        if question is None:
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0].get("role") != "user":
                fail(f"Cannot recover protected question: {path}:{index + 1}")
            content = prompt[0]["content"]
            if not content.startswith(prefix) or not content.endswith(suffix):
                fail(f"Protected QA prompt differs from PROMPT: {path}:{index + 1}")
            question = content[len(prefix):len(content) - len(suffix)] if suffix else content[len(prefix):]
        if not isinstance(question, str) or not question.strip():
            fail(f"Empty protected question: {path}:{index + 1}")
        result.append({"question": question, "sample_id": str(row.get("sample_id", f"{path.name}:{index + 1}"))})
    if not result:
        fail(f"Protected set is empty: {path}")
    return result


def check_question(row):
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
        fail("Episode sample_id must be a nonempty string")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        fail(f"Episode question is empty: {row['sample_id']}")


def aether_segments(row):
    """Use original character regions before removing the one terminal EOT."""
    text = row.get("full_trajectory_text")
    segments = row.get("loss_segments")
    if not isinstance(text, str) or not isinstance(segments, list) or len(segments) < 2:
        fail(f"Invalid AetherSearch text/segments: {row['sample_id']}")
    previous = 0
    extracted = []
    for index, segment in enumerate(segments):
        start, end = segment.get("char_start"), segment.get("char_end")
        if type(start) is not int or type(end) is not int or start != previous or not start < end <= len(text):
            fail(f"AetherSearch character regions are not a complete contiguous partition: {row['sample_id']}")
        previous = end
        kind, supervised = segment.get("kind"), segment.get("supervised")
        if index == 0:
            if kind != "prompt" or supervised is not False:
                fail(f"AetherSearch first region must be a masked prompt: {row['sample_id']}")
            continue
        expected_kind = "assistant_action_and_eot" if index == len(segments) - 1 else (
            "assistant_action" if index % 2 else "tool_observation")
        expected_supervised = index % 2 == 1
        if kind != expected_kind or supervised is not expected_supervised:
            fail(f"AetherSearch action/observation regions do not alternate: {row['sample_id']}")
        content = text[start:end]
        if index == len(segments) - 1:
            if not expected_supervised or not content.endswith("<|im_end|>"):
                fail(f"AetherSearch final action lacks its exact terminal EOT: {row['sample_id']}")
            content = content[:-len("<|im_end|>")]
        extracted.append({"content": content, "supervised": expected_supervised,
                          "source_char_start": start, "source_char_end": end, "source_kind": kind})
    if previous != len(text):
        fail(f"AetherSearch character regions do not cover the original text: {row['sample_id']}")
    if sum(not item["supervised"] for item in extracted) != row.get("search_count"):
        fail(f"AetherSearch search count differs from its observation regions: {row['sample_id']}")
    return extracted


def generated_segments(row):
    events = row.get("events")
    if not isinstance(events, list) or not events or len(events) % 2 != 1:
        fail(f"Generated episode has invalid event count: {row['sample_id']}")
    segments, spans, text = [], [], ""
    for index, event in enumerate(events):
        expected_role = "assistant" if index % 2 == 0 else "environment"
        if event.get("role") != expected_role or not isinstance(event.get("content"), str):
            fail(f"Generated events do not alternate assistant/environment: {row['sample_id']}")
        content = event["content"] + "\n"
        start = len(text)
        text += content
        supervised = expected_role == "assistant"
        if supervised:
            spans.append({"start": start, "end": len(text), "loss": 1})
        segments.append({"content": content, "supervised": supervised,
                         "source_char_start": start, "source_char_end": len(text),
                         "source_kind": expected_role})
    if row.get("trajectory") != text or row.get("assistant_loss_spans") != spans:
        fail(f"Generated trajectory/loss spans differ from its recorded events: {row['sample_id']}")
    if (len(events) - 1) // 2 != row.get("search_count"):
        fail(f"Generated search count differs from its recorded events: {row['sample_id']}")
    final_tag = "<answer>" + html.escape(row["final_answer"], quote=False) + "</answer>"
    if not events[-1]["content"].endswith(final_tag):
        fail(f"Generated final assistant action differs from its graded answer: {row['sample_id']}")
    return segments


def encode_episode(row, origin, segments, tokenizer, maximum, source_file, source_line):
    check_question(row)
    question = row["question"]
    prompt = PROMPT.format(question=question if question.endswith("?") else question + "?")
    text, regions = prompt, []
    for index, segment in enumerate(segments):
        content = segment["content"]
        if not content:
            fail(f"Empty action/observation region: {row['sample_id']}")
        if segment["supervised"]:
            if "<information>" in content or "</information>" in content:
                fail(f"Assistant region contains environment information tags: {row['sample_id']}")
            closing = "</answer>" if index == len(segments) - 1 else "</search>"
            if not content.rstrip().endswith(closing):
                fail(f"Assistant action does not end at the expected protocol tag: {row['sample_id']}")
        elif not content.lstrip().startswith("<information>") or not content.rstrip().endswith("</information>"):
            fail(f"Observation must be a complete information block: {row['sample_id']}")
        start = len(text)
        text += content
        regions.append({**segment, "char_start": start, "char_end": len(text)})
    if not regions[-1]["supervised"] or "<answer>" not in regions[-1]["content"]:
        fail(f"Complete episode does not end in an assistant answer: {row['sample_id']}")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
    if len(ids) != len(offsets):
        fail("Tokenizer returned inconsistent input IDs and character offsets")
    forbidden_ids = {151643, 151644, 151645}
    if any(token in forbidden_ids for token in ids):
        fail(f"Embedded Qwen control token survived Base conversion: {row['sample_id']}")
    labels, counts = [], [0] * len(regions)
    prompt_tokens = information_tokens = boundaries = 0
    region_index = 0
    for token, (start, end) in zip(ids, offsets):
        while region_index < len(regions) and start >= regions[region_index]["char_end"]:
            region_index += 1
        region = regions[region_index] if region_index < len(regions) else None
        inside = region is not None and start >= region["char_start"] and end <= region["char_end"] and end > start
        supervised = inside and region["supervised"]
        labels.append(token if supervised else -100)
        if supervised:
            counts[region_index] += 1
        elif end <= len(prompt):
            prompt_tokens += 1
        elif inside and not region["supervised"]:
            information_tokens += 1
        else:
            boundaries += 1
    if any(counts[index] == 0 for index, region in enumerate(regions) if region["supervised"]):
        fail(f"An assistant action has no wholly supervised token: {row['sample_id']}")
    eos = tokenizer.eos_token_id
    ids.append(eos)
    labels.append(eos)
    if len(ids) > maximum:
        return None, {"sample_id": row["sample_id"], "data_origin": origin,
                      "reason": "complete_episode_over_max_length", "length": len(ids), "max_length": maximum}
    result = {"sample_id": row["sample_id"], "question": question, "data_origin": origin,
              "input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
              "length": len(ids), "supervised_tokens": sum(value != -100 for value in labels)}
    audit = {"sample_id": row["sample_id"], "data_origin": origin,
             "source_file": str(source_file), "source_line": source_line,
             "source_record_sha256": digest_text(canonical(row)), "input_text_sha256": digest_text(text),
             "source_audit_path": row.get("audit_path"),
             "conversion": "replace only source prompt and terminal EOT; preserve action/observation text" if origin == "aethersearch" else "rebuild exact recorded events with original newline separators",
             "regions": [{key: value for key, value in region.items() if key != "content"}
                         for region in regions],
             "masked_prompt_tokens": prompt_tokens, "masked_information_tokens": information_tokens,
             "masked_boundary_tokens": boundaries, "assistant_region_supervised_tokens": [
                 counts[index] for index, region in enumerate(regions) if region["supervised"]],
             "length": result["length"], "supervised_tokens": result["supervised_tokens"],
             "terminal_eos_token_id": eos}
    return result, audit


def validate_existing(output, identity):
    manifest = read_json(output / "manifest.json")
    for field, expected in identity.items():
        if manifest.get(field) != expected:
            fail(f"Existing prepared dataset has conflicting {field}; choose a new prepared_dir")
    if manifest.get("complete") is not True:
        fail("Existing prepared dataset is incomplete")
    for filename, metadata in manifest.get("files", {}).items():
        path = relative_file(output, filename)
        if sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
            fail(f"Existing prepared file checksum differs: {filename}")
    for required in ("train.parquet", "val.parquet", "preparation_report.json", "episodes.audit.jsonl"):
        if required not in manifest.get("files", {}):
            fail(f"Existing prepared manifest does not bind {required}")
    print(canonical({"phase": "reuse_verified_prepared_dataset", "prepared_dir": str(output),
                     "counts": manifest["counts"]}), flush=True)
    return manifest


def ensure_unique_ids(partitions):
    ids = [row["sample_id"] for values in partitions.values() for row in values]
    if len(ids) != len(set(ids)):
        fail("Episode sample IDs are not globally unique across training and validation")


def prepare(config, generation_report):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    sources = {}
    for field in ("protected_qa_val", "protected_grpo"):
        record_hash(Path(config[field]), sources)
    aether = validate_aether(config, sources)
    generated, generation_settings = validate_generated(config, generation_report, sources)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True,
                                              local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.eos_token_id != 151643:
        fail("Preparation requires the Qwen3-4B-Base fast tokenizer with EOS token 151643")
    provenance = tokenizer_provenance(tokenizer)
    original_provenance = generation_settings["student_tokenizer_provenance"]
    for field in ("backend_sha256", "special_tokens_sha256", "eos_token_id"):
        if provenance.get(field) != original_provenance.get(field):
            fail(f"Student tokenizer differs from the generator's length-check tokenizer: {field}")
    for filename in ("prepare_data.py", "qa_metrics.py", "sft_length.py"):
        record_hash(Path(__file__).with_name(filename), sources)
    identity = {"format_version": FORMAT_VERSION, "configuration": config,
                "config_fingerprint": digest_text(canonical(config)), "source_hashes": sources,
                "implementation_sha256": sha256(Path(__file__)), "prompt_sha256": digest_text(PROMPT),
                "tokenizer": provenance, "max_length": config["max_length"],
                "expected_new_episodes": config["expected_new_episodes"], "mask_policy": MASK_POLICY}
    output = Path(config["prepared_dir"])
    if output.exists() and any(output.iterdir()):
        if not (output / "manifest.json").is_file():
            fail("prepared_dir is nonempty without a complete manifest; choose a new output directory")
        return validate_existing(output, identity)
    for values in [aether["train"], aether["val"], generated]:
        for row in values:
            check_question(row)
    new_index = QuestionIndex(generated)
    for row in generated:
        matches = [match for match in new_index.matches(row) if match["sample_id"] != row["sample_id"]]
        if matches:
            fail(f"New episodes contain duplicate/near-duplicate questions: {row['sample_id']}")
    val_index = QuestionIndex(aether["val"])
    for row in generated:
        if val_index.matches(row):
            fail(f"New training episode overlaps AetherSearch validation; protected validation is retained: {row['sample_id']}")
    removed_duplicates = []
    kept_aether_train = []
    for row in aether["train"]:
        matches = new_index.matches(row)
        if matches:
            removed_duplicates.append({"sample_id": row["sample_id"], "reason": "aether_train_overlaps_new100", "matches": matches})
        else:
            kept_aether_train.append(row)
    for row in kept_aether_train:
        if val_index.matches(row):
            fail(f"AetherSearch train overlaps its validation: {row['sample_id']}")
    protected_counts = {}
    all_selected = kept_aether_train + aether["val"] + generated
    for field in ("protected_qa_val", "protected_grpo"):
        protected = protected_questions(config[field])
        protected_counts[field] = len(protected)
        index = QuestionIndex(protected)
        for row in all_selected:
            if index.matches(row):
                fail(f"Selected SFT episode overlaps {field}: {row['sample_id']}")
    partitions, audits, overlength = {"train": [], "val": []}, [], []
    selected_ids = {row["sample_id"] for row in kept_aether_train}
    for split, values, filename in (("train", aether["train"], "train.jsonl"),
                                     ("val", aether["val"], "validation.jsonl")):
        for index, row in enumerate(values, 1):
            if split == "train" and row["sample_id"] not in selected_ids:
                continue
            item, audit = encode_episode(row, "aethersearch", aether_segments(row), tokenizer,
                                         config["max_length"], Path(config["aether_dir"]) / filename, index)
            if item is None:
                overlength.append({**audit, "split": split})
            else:
                partitions[split].append(item)
                audits.append({**audit, "split": split})
    for index, row in enumerate(generated, 1):
        item, audit = encode_episode(row, "deepseek_new100", generated_segments(row), tokenizer,
                                     config["max_length"], Path(config["generated_run_dir"]) / "accepted.jsonl", index)
        if item is None:
            fail(f"New100 complete episode exceeds max_length; cannot silently train 99: {row['sample_id']}")
        partitions["train"].append(item)
        audits.append({**audit, "split": "train"})
    ensure_unique_ids(partitions)
    if not partitions["train"] or not partitions["val"]:
        fail("Training and validation must both remain nonempty after complete-episode length checks")
    for split, values in partitions.items():
        values.sort(key=lambda row: digest_text(f"{config['seed']}|{split}|{row['sample_id']}"))
    counts = {"train": len(partitions["train"]), "validation": len(partitions["val"]),
              "new_episodes": len(generated), "new_by_source": dict(Counter(row["data_source"] for row in generated)),
              "aether_train": sum(row["data_origin"] == "aethersearch" for row in partitions["train"]),
              "aether_validation": len(partitions["val"])}
    report = {"prepared_at": now(), "counts": counts,
              "aether_input_counts": {key: len(value) for key, value in aether.items()},
              "aether_train_removed_for_new100_duplicates": removed_duplicates,
              "aether_complete_episodes_dropped_for_length": overlength,
              "protected_set_rows": protected_counts,
              "leakage_guard": {"normalization": "Unicode NFKC + casefold + remove Unicode punctuation + collapse whitespace",
                                "features": "word 3-grams; whole sequence for questions under 3 words",
                                "jaccard_threshold": 0.85, "verification": "exact set Jaccard with integer threshold",
                                "limitation": "semantic paraphrases are not guaranteed excluded"},
              "lengths": {split: {"min": min(row["length"] for row in values),
                                    "max": max(row["length"] for row in values),
                                    "total_tokens": sum(row["length"] for row in values),
                                    "supervised_tokens": sum(row["supervised_tokens"] for row in values)}
                          for split, values in partitions.items()},
              "mask_checks": {"all_prompt_and_information_masked": True,
                              "each_assistant_region_has_supervised_tokens": True,
                              "single_final_eos_supervised": True, "no_truncation": True,
                              "no_chatml_control_tokens": True},
              "training_started": False,
              "limitations": ["AetherSearch factual correctness is inherited from upstream, which provides no independent gold aliases",
                              "Automatic teacher evidence review does not replace human spot checks",
                              "AetherSearch licensing and retrieval-distribution limitations remain in its source attribution"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    try:
        schema = pa.schema([("sample_id", pa.string()), ("question", pa.string()), ("data_origin", pa.string()),
                            ("input_ids", pa.list_(pa.int64())), ("attention_mask", pa.list_(pa.int64())),
                            ("labels", pa.list_(pa.int64())), ("length", pa.int64()), ("supervised_tokens", pa.int64())])
        files = {}
        for split, values in partitions.items():
            filename = "train.parquet" if split == "train" else "val.parquet"
            pq.write_table(pa.Table.from_pylist(values, schema=schema), temporary / filename, compression="snappy")
            files[filename] = {"sha256": sha256(temporary / filename), "bytes": (temporary / filename).stat().st_size, "rows": len(values)}
        (temporary / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (temporary / "episodes.audit.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(audits, key=lambda value: (value["split"], value["sample_id"])):
                handle.write(canonical(row) + "\n")
        for filename in ("preparation_report.json", "episodes.audit.jsonl"):
            files[filename] = {"sha256": sha256(temporary / filename), "bytes": (temporary / filename).stat().st_size}
        # Recheck every bound input before publication, including source scripts.
        for path, expected in sources.items():
            if sha256(path) != expected:
                fail(f"An input changed during preparation; no dataset published: {path}")
        manifest = {**identity, "complete": True, "prepared_at": report["prepared_at"],
                    "counts": counts, "files": files, "training_started": False}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(canonical({"phase": "prepared", "prepared_dir": str(output), "counts": counts,
                     "aether_duplicates_removed": len(removed_duplicates),
                     "aether_overlength_removed": len(overlength), "training_started": False}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wait-for-generated", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=86400)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.wait_timeout_seconds < 0 or args.poll_seconds < 1:
        parser.error("wait-timeout-seconds must be nonnegative; poll-seconds must be positive")
    config = load_configuration(args.config)
    output = Path(config["prepared_dir"])
    if output.exists() and not output.is_dir():
        fail("prepared_dir exists and is not a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.prepare.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("Another data-preparation process is already using this prepared_dir")
        with completed_generation(Path(config["generated_run_dir"]), args.wait_for_generated,
                                  args.wait_timeout_seconds, args.poll_seconds) as report:
            prepare(config, report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; no generation or training was started", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
