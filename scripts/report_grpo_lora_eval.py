#!/usr/bin/env python3
"""Periodically summarize four workers without loading a model or using a GPU."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

from prepare_grpo_lora_eval import (SUBSET, atomic_json, canonical, fail, load_config,
                                    load_grpo, read_json, read_jsonl, selected_hash, verify_run)
from qa_metrics import metric_spec, score_answer


STATUSES = {"ok", "invalid_action", "unfinished_generation", "search_budget_exceeded",
            "context_budget_exceeded", "generation_budget_exceeded", "retrieval_error",
            "model_error", "oom"}
ERROR_CATEGORIES = {"invalid_action": "format_failure", "unfinished_generation": "unfinished_generation",
                    "search_budget_exceeded": "search_budget_error",
                    "context_budget_exceeded": "context_budget_error",
                    "generation_budget_exceeded": "token_budget_error",
                    "retrieval_error": "retrieval_failure", "model_error": "model_failure", "oom": "oom"}


def nonnegative_integer(value, label):
    if type(value) is not int or value < 0:
        fail(f"Prediction {label} must be a nonnegative integer")
    return value


def load_predictions(config, manifest, examples):
    output = Path(config["output_dir"])
    count = manifest["num_workers"]
    indexed = {row["sample_id"]: (index, row) for index, row in enumerate(examples)}
    records, worker_progress, raw_count = {}, [], 0
    for worker in range(count):
        path = output / f"worker-{worker}.predictions.jsonl"
        values = read_jsonl(path, tolerate_partial_tail=True) if path.is_file() else []
        final = {}
        for row in values:
            sid = row.get("sample_id")
            if sid not in indexed:
                fail(f"Worker {worker} contains an unknown evaluation sample ID")
            index, original = indexed[sid]
            if index % count != worker or row.get("worker_index") != worker or row.get("source_index") != index:
                fail(f"Prediction has inconsistent source index/worker ownership: {sid}")
            for field in ("data_source", "question", "golden_answers"):
                if row.get(field) != original[field]:
                    fail(f"Prediction {field} differs from the unchanged evaluation source: {sid}")
            status = row.get("status")
            if status not in STATUSES:
                fail(f"Prediction is not a recognized terminal status: {sid}")
            if not isinstance(row.get("answer"), str) or not isinstance(row.get("events"), list):
                fail(f"Prediction answer/events have invalid types: {sid}")
            search_count = nonnegative_integer(row.get("search_count"), "search_count")
            usage = row.get("usage", {})
            if not isinstance(usage, dict):
                fail(f"Prediction usage must be an object: {sid}")
            tokens = nonnegative_integer(usage.get("generated_tokens", row.get("generated_tokens", 0)), "generated_tokens")
            if status == "ok":
                scores = score_answer(row["answer"], original["golden_answers"], original["data_source"])
            else:
                scores = {"em": 0, "f1": 0.0}
            # Recompute from the immutable gold; never trust a worker's generous score.
            if type(row.get("em")) not in (int, bool) or int(row["em"]) != scores["em"]:
                fail(f"Prediction EM differs from official local rescoring: {sid}")
            if (not isinstance(row.get("f1"), (int, float)) or not math.isfinite(row["f1"])
                    or not math.isclose(row["f1"], scores["f1"], abs_tol=1e-12)):
                fail(f"Prediction F1 differs from official local rescoring: {sid}")
            final[sid] = {**row, **scores, "search_count": search_count, "generated_tokens": tokens}
        records.update(final)
        raw_count += len(values)
        worker_progress.append({"worker_index": worker, "selected": sum(index % count == worker for index in range(len(examples))),
                                "terminal_unique": len(final), "raw_terminal_records": len(values),
                                "predictions_file": str(path)})
    extra = [path.name for path in output.glob("worker-*.predictions.jsonl")
             if path.name not in {f"worker-{index}.predictions.jsonl" for index in range(count)}]
    if extra:
        fail("Unexpected worker prediction files exist; worker-count/run configuration differs")
    return records, worker_progress, raw_count


def group_report(examples, predictions):
    attempted = [predictions[row["sample_id"]] for row in examples if row["sample_id"] in predictions]
    selected, n = len(examples), len(attempted)
    correct = sum(row["em"] for row in attempted)
    f1 = sum(row["f1"] for row in attempted)
    complete = n == selected
    statuses = Counter(row["status"] for row in attempted)
    categories = Counter(ERROR_CATEGORIES[row["status"]] for row in attempted if row["status"] != "ok")
    return {"selected": selected, "attempted": n, "pending": selected - n,
            "valid_answers": statuses.get("ok", 0), "failed": n - statuses.get("ok", 0),
            "correct": correct, "complete": complete,
            "em": correct / selected if selected else None,
            "f1": f1 / selected if selected else None,
            "metrics_are_final": complete,
            "accuracy_final": 100 * correct / selected if complete and selected else None,
            "accuracy_final_unit": "percent",
            "attempted_em": correct / n if n else None, "attempted_f1": f1 / n if n else None,
            "status_counts": dict(statuses), "error_categories": dict(categories),
            "mean_search_count": sum(row["search_count"] for row in attempted) / n if n else None,
            "mean_generated_tokens": sum(row["generated_tokens"] for row in attempted) / n if n else None,
            "total_generated_tokens": sum(row["generated_tokens"] for row in attempted)}


def report(config):
    output = Path(config["output_dir"])
    manifest = read_json(output / "run_manifest.json")
    verify_run(config, manifest)
    examples = load_grpo(config["data_file"])
    if manifest.get("selected_sha256") != selected_hash(examples):
        fail("Selected evaluation records/order changed since preparation")
    predictions, workers, raw_count = load_predictions(config, manifest, examples)
    overall = group_report(examples, predictions)
    by_source = {source: group_report([row for row in examples if row["data_source"] == source], predictions)
                 for source in sorted({row["data_source"] for row in examples})}
    complete = overall["complete"]
    result = {"updated_at": datetime.now(timezone.utc).isoformat(), "subset": SUBSET,
              "official_test_set": False, "complete": complete,
              "status": "complete_with_failures" if complete and overall["failed"] else "complete" if complete else "partial",
              "metric_denominator": "Fixed selected set: 2000 overall, 1000 per source; failures score zero. Before completion, em/f1 are lower bounds with pending not yet scored; accuracy_final remains null",
              "mean_metric_denominator": "All attempted terminal questions, including failures; pending questions excluded",
              "accuracy_final": overall["accuracy_final"], "accuracy_final_unit": "percent",
              "metric_specs": {source: metric_spec(source) for source in by_source},
              "sub_em_used_for_acceptance_or_accuracy": False,
              "overall": overall, "by_source": by_source, "workers": workers,
              "raw_terminal_records": raw_count,
              "data_sha256": manifest["data_sha256"], "selected_sha256": manifest["selected_sha256"],
              "adapter_path": config["adapter_path"], "base_model_path": config["model_path"]}
    atomic_json(output / "report.json", result)
    print(canonical({"评测集": SUBSET, "完成题数": overall["attempted"], "总题数": overall["selected"],
                     "待评测": overall["pending"], "失败题数": overall["failed"],
                     "最终EM准确率百分比": overall["accuracy_final"],
                     "各来源最终EM百分比": {source: values["accuracy_final"] for source, values in by_source.items()},
                     "错误分类": overall["error_categories"], "报告": str(output / "report.json")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report(load_config(args.config))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"评测汇总失败：{error}", file=sys.stderr)
        sys.exit(1)
