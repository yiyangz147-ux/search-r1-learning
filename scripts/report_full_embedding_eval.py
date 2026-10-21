#!/usr/bin/env python3
"""Independently rescore committed RAG trajectories without loading a model."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

from prepare_full_embedding_eval import (SUBSET, atomic_json, canonical, digest_text, fail,
                                         read_jsonl, verify_run)
from qa_metrics import metric_spec
from search_sft_protocol import build_prompt
from sft_reward_monitor import parse_action, score_trajectory, valid_step_format


STATUSES = {"ok", "turn_limit", "context_limit", "invalid_action_limit", "generated_token_limit"}
INFRASTRUCTURE_STATUSES = {"infrastructure_error", "retrieval_error", "model_error", "oom"}
SCORES = ("total_reward", "answer_reward", "format_reward", "qa_em", "qa_f1")


def nonnegative_integer(value, label):
    if type(value) is not int or value < 0:
        fail(f"Prediction {label} must be a nonnegative integer")
    return value


def rescore_record(row, example, config):
    """Recover the final answer from generated events, then compare immutable gold.

    Worker-reported reward, format flags, parsed actions and answer are never
    treated as ground truth. A correct answer with broken formatting keeps EM=1.
    """
    sid = example["sample_id"]
    for field in ("sample_id", "question", "data_source", "golden_answers"):
        if row.get(field) != example[field]:
            fail(f"Prediction {field} differs from the selected question: {sid}")
    if row.get("id", sid) != sid:
        fail(f"Prediction id differs from sample_id: {sid}")
    if row.get("status") not in STATUSES:
        fail(f"Unknown terminal prediction status: {sid}")
    if row.get("initial_prompt") != build_prompt(example["question"]):
        fail(f"Prediction prompt differs from the shared training protocol: {sid}")
    events = row.get("events")
    if not isinstance(events, list):
        fail(f"Prediction events must be a list: {sid}")
    all_valid, answer, generated, searches, calls = True, "", 0, 0, 0
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("content"), str):
            fail(f"Malformed trajectory event: {sid}")
        if answer:
            fail(f"Trajectory continues after its first complete answer: {sid}")
        if event.get("role") == "assistant":
            text = event["content"]
            action = parse_action(text)
            all_valid = all_valid and valid_step_format(text)
            generated += nonnegative_integer(event.get("generated_tokens"), "event.generated_tokens")
            nonnegative_integer(event.get("input_tokens"), "event.input_tokens")
            calls += 1
            if action is not None and action["action"] == "answer":
                answer = action["value"]
        elif event.get("role") == "environment":
            if event.get("kind") == "retrieval":
                searches += 1
            elif event.get("kind") == "search_limit_feedback":
                all_valid = False
            elif event.get("kind") != "invalid_action_feedback":
                fail(f"Unrecognized environment event: {sid}")
        else:
            fail(f"Unexpected event role: {sid}")
    complete = row["status"] == "ok"
    if complete != bool(answer) or row.get("answer") != answer:
        fail(f"Prediction answer/status disagrees with its generated action: {sid}")
    if (row.get("search_count") != searches or row.get("generated_tokens") != generated
            or row.get("model_calls") != calls):
        fail(f"Prediction usage disagrees with its generated trajectory: {sid}")
    if (searches > config["max_searches"] or generated > config["max_total_generated_tokens"]
            or calls > config["max_turns"]):
        fail(f"Prediction exceeds the immutable rollout budget: {sid}")
    scores = score_trajectory(answer, example["golden_answers"], example["data_source"],
                               completed=complete, all_steps_valid=all_valid)
    for name, expected in scores.items():
        actual = row.get(name)
        if (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not math.isclose(actual, expected, abs_tol=1e-12)):
            fail(f"Prediction {name} disagrees with independent rescoring: {sid}")
    return {**row, **scores}


def load_predictions(config, manifest, examples):
    output = Path(config["output_dir"]) / "workers"
    indexed = {row["sample_id"]: (index, row) for index, row in enumerate(examples)}
    binding = digest_text(canonical(manifest))
    records, progress, infra, raw_count = {}, [], [], 0
    expected_paths = {f"worker-{worker:02d}.jsonl" for worker in range(config["num_workers"])}
    for worker in range(config["num_workers"]):
        path = output / f"worker-{worker:02d}.jsonl"
        values = read_jsonl(path, tolerate_partial_tail=True) if path.is_file() else []
        worker_records = 0
        for row in values:
            sid = row.get("sample_id")
            if sid not in indexed:
                fail(f"Worker {worker} contains an unknown sample ID")
            index, example = indexed[sid]
            if (index % config["num_workers"] != worker or row.get("worker_index") != worker
                    or row.get("sample_index") != index or row.get("run_binding_sha256") != binding):
                fail(f"Worker ownership/run binding differs: {sid}")
            if row.get("status") in INFRASTRUCTURE_STATUSES:
                infra.append({"sample_id": sid, "worker_index": worker, "status": row["status"]})
                continue
            if sid in records:
                fail(f"Duplicate committed prediction ID: {sid}")
            records[sid] = rescore_record(row, example, config)
            worker_records += 1
        raw_count += len(values)
        progress.append({"worker_index": worker, "selected": sum(i % config["num_workers"] == worker for i in range(len(examples))),
                         "attempted": worker_records, "terminal_unique": worker_records,
                         "raw_records": len(values), "predictions_file": str(path)})
    extras = [path.name for path in output.glob("worker-*.jsonl") if path.name not in expected_paths]
    if extras:
        fail("Unexpected worker journals exist; worker-count/run configuration differs")
    return records, progress, infra, raw_count


def group_report(examples, predictions):
    attempted = [predictions[row["sample_id"]] for row in examples if row["sample_id"] in predictions]
    selected, count = len(examples), len(attempted)
    complete = count == selected
    correct = sum(row["total_reward"] for row in attempted)
    format_correct = sum(row["format_reward"] for row in attempted)
    statuses = Counter(row["status"] for row in attempted)
    return {"selected": selected, "count": count, "attempted": count, "pending": selected - count,
            "complete": complete, "metrics_are_final": complete, "correct": correct,
            "total_reward_sum": correct, "format_reward_sum": format_correct,
            "total_reward": correct / selected if complete and selected else None,
            "answer_em": correct / selected if complete and selected else None,
            "em": correct / selected if complete and selected else None,
            "accuracy_final": 100 * correct / selected if complete and selected else None,
            "accuracy_final_unit": "percent",
            "format_reward": format_correct / selected if complete and selected else None,
            "format_accuracy_percent": 100 * format_correct / selected if complete and selected else None,
            "qa_em": sum(row["qa_em"] for row in attempted) / selected if complete and selected else None,
            "qa_f1": sum(row["qa_f1"] for row in attempted) / selected if complete and selected else None,
            "attempted_mean_total_reward": correct / count if count else None,
            "attempted_mean_format_reward": format_correct / count if count else None,
            "valid_answers": statuses.get("ok", 0), "failed": count - statuses.get("ok", 0),
            "status_counts": dict(statuses),
            "mean_search_count": sum(row["search_count"] for row in attempted) / count if count else None,
            "mean_generated_tokens": sum(row["generated_tokens"] for row in attempted) / count if count else None,
            "total_generated_tokens": sum(row["generated_tokens"] for row in attempted)}


def report(config_path, *, full_hash=False):
    config, manifest, examples = verify_run(config_path, full_hash=full_hash)
    predictions, workers, infra, raw_count = load_predictions(config, manifest, examples)
    overall = group_report(examples, predictions)
    sources = sorted({row["data_source"] for row in examples})
    by_source = {source: group_report([row for row in examples if row["data_source"] == source], predictions)
                 for source in sources}
    monitor_ids = set(manifest.get("training_monitor_subset", {}).get("evaluation_sample_ids", []))
    excluding = [row for row in examples if row["sample_id"] not in monitor_ids]
    monitored = [row for row in examples if row["sample_id"] in monitor_ids]
    complete = overall["complete"]
    result = {"updated_at": datetime.now(timezone.utc).isoformat(), "subset": SUBSET,
              "status": "complete_with_failures" if complete and overall["failed"] else "complete" if complete else "partial",
              "complete": complete, "accuracy_final": overall["accuracy_final"], "accuracy_final_unit": "percent",
              "overall": overall, "by_source": by_source,
              "excluding_training_monitor": {"overall": group_report(excluding, predictions),
                    "by_source": {source: group_report([row for row in excluding if row["data_source"] == source], predictions)
                                  for source in sources}},
              "training_monitor": {"overall": group_report(monitored, predictions),
                    "by_source": {source: group_report([row for row in monitored if row["data_source"] == source], predictions)
                                  for source in sources}},
              "metric_policy": "total_reward=Search-R1 normalized answer EM; format_reward separate, no shaping bonus; source-specific EM/F1 auxiliary",
              "metric_denominator": "All selected questions, including model failures; final means remain null until every selected question has a terminal trajectory",
              "partial_metric_policy": "attempted_mean_* describe completed terminal trajectories only; they are not final selected-set accuracy",
              "infrastructure_error_policy": "Infrastructure failures remain pending for retry and never become incorrect answers",
              "monitoring_caveat": "The original Base development questions include training-monitor questions; the excluded-monitor result is reported separately",
              "metric_specs": {source: metric_spec(source) for source in ["search-r1", *sources]},
              "sub_em_used_for_acceptance_or_accuracy": False, "workers": workers,
              "infrastructure_records": infra, "raw_records": raw_count,
              "data_sha256": manifest["data_sha256"], "selected_sha256": manifest["selected_sha256"],
              "run_binding_sha256": digest_text(canonical(manifest)),
              "adapter_path": config["adapter_path"], "base_model_path": config["model_path"]}
    path = Path(config["output_dir"]) / "report.json"
    atomic_json(path, result)
    print(canonical({"完成题数": overall["attempted"], "总题数": overall["selected"], "待评测": overall["pending"],
                     "最终EM百分比": overall["accuracy_final"], "最终格式Reward": overall["format_reward"],
                     "各来源最终EM百分比": {source: row["accuracy_final"] for source, row in by_source.items()},
                     "排除训练监控题EM百分比": result["excluding_training_monitor"]["overall"]["accuracy_final"],
                     "报告": str(path)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    report(args.config, full_hash=args.full_hash)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"评测汇总失败：{error}", file=sys.stderr)
        sys.exit(1)
