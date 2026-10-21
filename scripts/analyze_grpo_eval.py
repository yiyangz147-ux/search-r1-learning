#!/usr/bin/env python3
"""CPU-only post-hoc diagnostics; never alter predictions or primary scores."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from qa_metrics import answer_diagnostics, score_answer


ACTION = re.compile(r"<(search|answer)>(.*?)</\1>", re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir
    report = json.loads((directory / "report.json").read_text())
    assert report["complete"] and report["overall"]["attempted"] == 2000
    records = {}
    for path in sorted(directory.glob("worker-*.predictions.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            records[row["sample_id"]] = row
    assert len(records) == 2000
    rows = sorted(records.values(), key=lambda row: row["source_index"])
    prefix = Counter()
    search_counts = Counter()
    by_status = defaultdict(Counter)
    first_turn_failures = Counter()
    lenient = Counter()
    sub_em_only = []
    missed_answers = []
    examples = defaultdict(list)
    for row in rows:
        assistants = [event for event in row["events"] if event["role"] == "assistant"]
        environments = [event for event in row["events"] if event["role"] == "environment"]
        first = assistants[0]["content"] if assistants else ""
        last = assistants[-1]["content"] if assistants else ""
        opening = re.match(r"\s*(<[^>\n]{1,50}>)", first)
        first_category = opening.group(1) if opening else "plain_or_other"
        prefix[first_category] += 1
        texts = "\n".join(event["content"] for event in assistants)
        status = row["status"]
        counts = by_status[status]
        counts["questions"] += 1
        counts["contains_translation_tag"] += int("<translation>" in texts)
        repeated = any(value >= 3 for line, value in Counter(
            line.strip() for line in texts.splitlines() if len(line.strip()) >= 20).items())
        counts["has_line_repeated_3plus_times"] += int(repeated)
        counts["starts_with_think"] += int(first.lstrip().startswith("<think>"))
        counts["with_retrieved_observation"] += int(bool(environments))
        counts["ends_after_first_model_call"] += int(len(assistants) == 1)
        if status != "ok" and len(assistants) == 1:
            first_turn_failures[status] += 1
        search_counts[row["search_count"]] += 1
        if status == "ok":
            diagnostic = answer_diagnostics(row["answer"], row["golden_answers"], row["data_source"])
            if diagnostic["search_r1_sub_em"] and not row["em"]:
                sub_em_only.append({"sample_id": row["sample_id"], "answer": row["answer"],
                                    "golden_answers": row["golden_answers"]})
        else:
            action = ACTION.search(last)
            category = action.group(1) if action else "no_complete_action"
            lenient[f"{status}/{category}"] += 1
            if action and action.group(1) == "answer":
                value = html.unescape(action.group(2)).strip()
                score = score_answer(value, row["golden_answers"], row["data_source"])
                missed_answers.append({"sample_id": row["sample_id"], "status": status,
                                       "answer": value, "em": score["em"], "f1": score["f1"]})
        if len(examples[status]) < 3:
            examples[status].append({"sample_id": row["sample_id"], "question": row["question"],
                                     "status": status, "search_count": row["search_count"],
                                     "model_calls": len(assistants), "first_output_head": first[:600],
                                     "last_output_tail": last[-600:], "answer": row["answer"],
                                     "golden_answers": row["golden_answers"], "em": row["em"]})
    strict_correct = report["overall"]["correct"]
    additional = sum(row["em"] for row in missed_answers)
    result = {
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-hoc inspection only; no new model generations or retrieval; original report unchanged",
        "questions": len(rows), "first_output_prefixes": dict(prefix.most_common()),
        "by_status": {key: dict(value) for key, value in by_status.items()},
        "failed_at_first_model_call": dict(first_turn_failures),
        "search_count_distribution": dict(sorted(search_counts.items())),
        "questions_with_any_search": sum(row["search_count"] > 0 for row in rows),
        "total_searches": sum(row["search_count"] for row in rows),
        "official_regex_compatible_first_action_in_failed_outputs": dict(lenient),
        "offline_answer_extraction_diagnostic": {
            "failed_questions_with_first_action_answer": len(missed_answers),
            "additional_exact_matches": additional,
            "strict_original_correct": strict_correct,
            "terminal_answer_only_correct": strict_correct + additional,
            "terminal_answer_only_em_percent": 100 * (strict_correct + additional) / len(rows),
            "limitation": "This only rescored complete answers already generated. Search continuation and invalid-action feedback require new rollouts; this is NOT an official-compatible rerun score.",
            "cases": missed_answers,
        },
        "valid_answer_conditioned_em_percent": 100 * strict_correct / report["overall"]["valid_answers"],
        "strict_mismatches_with_substring_match": len(sub_em_only),
        "substring_only_cases": sub_em_only,
        "substring_limitation": "Character containment is not semantic correctness and never replaces strict EM.",
        "representative_examples": dict(examples),
    }
    target = directory / "diagnostics.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    short = {key: value for key, value in result.items()
             if key not in ("representative_examples", "substring_only_cases")}
    short["offline_answer_extraction_diagnostic"] = {
        key: value for key, value in short["offline_answer_extraction_diagnostic"].items() if key != "cases"}
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
