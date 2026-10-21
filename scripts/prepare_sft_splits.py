#!/usr/bin/env python3
"""Exact hash + MinHash LSH/Jaccard cleanup and disjoint QA partitions.

LSH recalls candidates; exact Jaccard rejects false positives. An additional
exact prefix-index guard protects ALL exported partitions against missed LSH
pairs and against existing NQ test / HotpotQA dev question overlap.
"""

import argparse
from collections import Counter, defaultdict, deque
import importlib.metadata
import math
from pathlib import Path
import random
import sys

from prepare_data import arrow_schema, fingerprints, sample as rl_sample, save_json
from prepare_sft import allocate, digest_text, normalize_answer, now, question_key, rows, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def shingles(question, width=3):
    tokens = question_key(question).split()
    width = min(width, len(tokens))
    if not width:
        return frozenset()
    return frozenset(sys.intern(f"w{width}:" + " ".join(tokens[i:i+width]))
                     for i in range(len(tokens) - width + 1))


def jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class ExactPrefixIndex:
    """Exact Jaccard threshold search using a sufficient rare-feature prefix.

    If J(A,B)>=t, then |A intersect B|>=ceil(t*|A|). Consequently B must
    intersect any prefix of A with |A|-ceil(t*|A|)+1 features. Index all B
    features, query that prefix of A, then verify every candidate exactly.
    Sorting by global feature frequency makes the necessary prefix selective.
    """

    def __init__(self, threshold, frequencies):
        self.threshold = threshold
        self.frequencies = frequencies
        self.postings = defaultdict(list)
        self.items = {}

    def insert(self, key, features, metadata):
        self.items[key] = (features, metadata)
        for feature in features:
            self.postings[feature].append(key)

    def query(self, features):
        n = len(features)
        prefix_length = n - math.ceil(self.threshold * n - 1e-12) + 1
        prefix = sorted(features, key=lambda feature: (self.frequencies[feature], feature))[:prefix_length]
        candidates = set()
        for feature in prefix:
            candidates.update(self.postings.get(feature, ()))
        for key in sorted(candidates):
            other, metadata = self.items[key]
            if min(n, len(other)) / max(n, len(other)) + 1e-12 < self.threshold:
                continue
            score = jaccard(features, other)
            if score + 1e-12 >= self.threshold:
                return key, score, metadata
        return None


def load_inputs(data_dir):
    evaluation, training, provenance = [], [], {}
    for source, split in [("nq", "test"), ("hotpotqa", "dev"), ("nq", "train"), ("hotpotqa", "train")]:
        path = data_dir / "raw" / source / f"{split}.jsonl"
        provenance[f"{source}/{split}"] = {"path": str(path), "sha256": fingerprints(path)[0]}
        for index, row in rows(path):
            question = row.get("question")
            item = {"data_source": source, "source_split": split,
                    "source_id": str(row.get("id", index)), "source_index": index,
                    "question": question.strip() if isinstance(question, str) else "",
                    "golden_answers": row.get("golden_answers", [])}
            metadata = row.get("metadata", {})
            item["stratum"] = f"{metadata.get('type', 'unknown')}/{metadata.get('level', 'unknown')}" if source == "hotpotqa" else "all"
            key = question_key(item["question"])
            item["sample_id"] = digest_text(f"{source}:{item['source_id']}:{key}")[:24]
            (training if split == "train" else evaluation).append(item)
    return evaluation, training, provenance


def clean(evaluation, training, threshold=0.85, num_perm=128, seed=42, progress=True):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, weights=(0.1, 0.9))
    template = MinHash(num_perm=num_perm, seed=seed)
    features_by_id, metadata_by_id, hashes = {}, {}, {}
    retained_eval, eligible, audit = [], [], []
    counts = {source: Counter() for source in ("nq", "hotpotqa")}
    for index, item in enumerate(evaluation + training):
        is_train = item["source_split"] == "train"
        source = item["data_source"]
        if is_train:
            counts[source]["raw"] += 1
        key = question_key(item["question"])
        if (not key or (is_train and (not isinstance(item["golden_answers"], list) or
                not item["golden_answers"] or any(not isinstance(a, str) or not normalize_answer(a)
                for a in item["golden_answers"])))):
            if is_train:
                counts[source]["invalid"] += 1
            continue
        fingerprint = digest_text(key)
        if fingerprint in hashes:
            parent = hashes[fingerprint]
            if is_train:
                reason = "evaluation_exact_overlap" if parent["source_split"] != "train" else "exact_duplicate"
                counts[source][reason] += 1
                audit.append({"removed_id": item["sample_id"], "source": source, "reason": reason,
                              "matched_id": parent["sample_id"], "jaccard": 1.0})
            continue
        # Exact dedup precedes the approximate stage, including rows later removed.
        hashes[fingerprint] = item
        features = shingles(item["question"])
        signature = template.copy()
        signature.update_batch([feature.encode() for feature in sorted(features)])
        match = None
        if is_train:
            for other_id in sorted(lsh.query(signature)):
                score = jaccard(features, features_by_id[other_id])
                if score + 1e-12 >= threshold:
                    match = other_id, score
                    break
        if match:
            other_id, score = match
            parent = metadata_by_id[other_id]
            reason = "evaluation_near_overlap" if parent["source_split"] != "train" else "near_duplicate"
            counts[source][reason] += 1
            audit.append({"removed_id": item["sample_id"], "source": source, "reason": reason,
                          "matched_id": other_id, "jaccard": score})
            continue
        sid = item["sample_id"]
        lsh.insert(sid, signature)
        features_by_id[sid] = features
        metadata_by_id[sid] = item
        if is_train:
            eligible.append(item)
            counts[source]["eligible_after_lsh"] += 1
        else:
            retained_eval.append(item)
        if progress and index and index % 20000 == 0:
            print(f"Scanned {index:,}/{len(evaluation)+len(training):,}; eligible train {len(eligible):,}", flush=True)
    settings = {"method": "normalized_question_SHA256_then_MinhashLSH_then_exact_Jaccard",
                "shingles": "word_3gram; shorter questions use their entire token sequence",
                "jaccard_threshold": threshold, "num_perm": num_perm, "seed": seed,
                "lsh_weights": [0.1, 0.9], "lsh_bands": lsh.b, "lsh_rows_per_band": lsh.r,
                "library": f"datasketch=={importlib.metadata.version('datasketch')}",
                "scope": "question_text_only; no semantic paraphrase guarantees"}
    return retained_eval, eligible, counts, audit, features_by_id, settings


def partition(evaluation, eligible, features, threshold, seed, source_targets, audit):
    frequencies = Counter(feature for feature_set in features.values() for feature in feature_set)
    guard = ExactPrefixIndex(threshold, frequencies)
    for item in evaluation:
        guard.insert(item["sample_id"], features[item["sample_id"]], {**item, "partition": "official_evaluation"})
    pools = {source: defaultdict(list) for source in source_targets}
    for item in eligible:
        pools[item["data_source"]][item["stratum"]].append(item)
    for source, groups in pools.items():
        for stratum, items in groups.items():
            items.sort(key=lambda row: digest_text(f"{seed}:split:{row['sample_id']}"))
            groups[stratum] = deque(items)
    output = {name: [] for name in ["validation", "grpo", "sft_initial", "sft_reserve"]}
    exact_rejects = Counter()
    for name in output:
        for source in sorted(source_targets):
            groups = pools[source]
            target = source_targets[source][name]
            quotas = allocate({key: len(items) for key, items in groups.items()}, target)
            taken = Counter()
            for stratum in sorted(groups):
                while taken[stratum] < quotas[stratum]:
                    if not groups[stratum]:
                        raise ValueError(f"Not enough clean questions in {source}/{stratum} for {name}")
                    item = groups[stratum].popleft()
                    match = guard.query(features[item["sample_id"]])
                    if match:
                        other_id, score, parent = match
                        reason = "exact_guard_evaluation_near_overlap" if parent["partition"] == "official_evaluation" else "exact_guard_partition_near_duplicate"
                        exact_rejects[f"{source}/{reason}"] += 1
                        audit.append({"removed_id": item["sample_id"], "source": source, "reason": reason,
                                      "matched_id": other_id, "matched_partition": parent["partition"], "jaccard": score})
                        continue
                    item = {**item, "partition": name}
                    guard.insert(item["sample_id"], features[item["sample_id"]], item)
                    output[name].append(item)
                    taken[stratum] += 1
        random.Random(f"{seed}:{name}").shuffle(output[name])
    return output, exact_rejects


def write_rl_parquet(items, path, split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    converted = []
    for index, item in enumerate(items):
        row = rl_sample({"id": item["source_id"], "question": item["question"],
                         "golden_answers": item["golden_answers"]}, item["data_source"], split, "train", index)
        converted.append(row)
    pq.write_table(pa.Table.from_pylist(converted, schema=arrow_schema()), path, compression="snappy")


def run(args):
    if (args.output_dir / "selection_manifest.json").exists():
        raise ValueError("Output selection already exists; choose a new --output-dir")
    evaluation, training, provenance = load_inputs(args.data_dir)
    evaluation, eligible, counts, audit, features, settings = clean(
        evaluation, training, args.jaccard_threshold, args.num_perm, args.seed)
    source_targets = {source: {"validation": args.val_per_source, "grpo": args.grpo_per_source,
                      "sft_initial": args.sft_per_source, "sft_reserve": args.reserves_per_source}
                      for source in ["nq", "hotpotqa"]}
    output, exact_rejects = partition(evaluation, eligible, features, args.jaccard_threshold,
                                    args.seed, source_targets, audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in output["sft_initial"]:
        item["selection"] = "initial"
    for item in output["sft_reserve"]:
        item["selection"] = "reserve"
    candidates = output["sft_initial"] + output["sft_reserve"]
    for name, items in [("selected", output["sft_initial"]), ("reserves", output["sft_reserve"]),
                        ("candidates", candidates), ("grpo", output["grpo"]), ("val", output["validation"]),
                        ("dedup_audit", audit)]:
        write_jsonl(args.output_dir / f"{name}.jsonl", items)
    write_rl_parquet(output["grpo"], args.output_dir / "grpo.parquet", "train")
    write_rl_parquet(output["validation"], args.output_dir / "val.parquet", "validation")
    qa_manifest = args.data_dir / "qa_manifest.json"
    upstream = __import__("json").loads(qa_manifest.read_text()) if qa_manifest.exists() else {}
    split_counts = {name: {"total": len(items), "sources": dict(Counter(row["data_source"] for row in items)),
                          "strata": dict(Counter(f"{row['data_source']}/{row['stratum']}" for row in items))}
                    for name, items in output.items()}
    settings["export_guard"] = "exact_prefix_index_then_exact_Jaccard; no missed above-threshold exported pairs"
    manifest = {"version": 2, "created_at": now(), "seed": args.seed,
                "targets": {source: args.sft_per_source for source in source_targets},
                "stats": {source: dict(value) for source, value in counts.items()},
                "dedup_settings": settings, "exact_guard_rejections": dict(exact_rejects),
                "partitions": split_counts, "sources": provenance,
                "upstream_repository": upstream.get("repository"), "upstream_revision": upstream.get("revision"),
                "candidates_sha256": fingerprints(args.output_dir / "candidates.jsonl")[0],
                "external_validation": {"path": str(args.output_dir / "val.parquet"),
                                        "sha256": fingerprints(args.output_dir / "val.parquet")[0]},
                "grpo": {"path": str(args.output_dir / "grpo.parquet"),
                         "sha256": fingerprints(args.output_dir / "grpo.parquet")[0]},
                "status": "sampled_only_no_teacher_trajectories"}
    save_json(args.output_dir / "selection_manifest.json", manifest)
    print(__import__("json").dumps({"output_dir": str(args.output_dir), "cleaning": manifest["stats"],
                                  "exact_guard_rejections": dict(exact_rejects), "partitions": split_counts,
                                  "dedup_settings": settings}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft/selection_3000_clean")
    parser.add_argument("--sft-per-source", type=int, default=1500)
    parser.add_argument("--reserves-per-source", type=int, default=1500)
    parser.add_argument("--grpo-per-source", type=int, default=1000)
    parser.add_argument("--val-per-source", type=int, default=500)
    parser.add_argument("--jaccard-threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.jaccard_threshold <= 1 or args.num_perm < 16:
        parser.error("Threshold must be in (0,1]; num_perm must be at least 16")
    if args.sft_per_source < 1 or min(args.reserves_per_source, args.grpo_per_source, args.val_per_source) < 0:
        parser.error("SFT count must be positive; other split counts must be nonnegative")
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
