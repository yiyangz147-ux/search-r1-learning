#!/usr/bin/env python3
"""Rescore saved QA answers offline without changing checkpoints or SFT status."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from qa_metrics import answer_diagnostics, score_answer
from sft_answer_aliases import check_sft_answer, load_reviewed_aliases


def audit_scores(input_file, output_dir, aliases_file=None):
    """Snapshot completed JSONL records once; output only answer-scoring fields."""
    input_file, output_dir = Path(input_file), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Audit output must be a new or empty directory")
    raw = input_file.read_bytes()
    lines = raw.splitlines(keepends=True)
    ignored_tail = bool(lines and not lines[-1].endswith(b"\n"))
    if ignored_tail:
        lines.pop()
    records = [json.loads(line) for line in lines if line.strip()]
    aliases = load_reviewed_aliases(aliases_file) if aliases_file else None
    output, counters, by_source = [], Counter(), defaultdict(Counter)
    for index, row in enumerate(records):
        source, gold = row["data_source"], row["golden_answers"]
        answer = row.get("final_answer", row.get("answer", "")) or ""
        metrics = score_answer(answer, gold, source)
        gate = check_sft_answer(row, answer, aliases) if aliases is not None else None
        diagnostic = answer_diagnostics(answer, gold, source)
        counts = {
            "attempt_records": 1,
            "records_with_answer": int(bool(answer.strip())),
            "previous_recorded_EM_matches": int(bool(row.get("answer_em", row.get("em", False)))),
            "rescored_original_gold_EM_matches": metrics["em"],
            "reviewed_alias_only_matches": int(gate is not None and gate["method"] == "reviewed_alias"),
            "search_r1_sub_em_matches_diagnostic_only": diagnostic["search_r1_sub_em"],
        }
        counters.update(counts)
        by_source[source].update(counts)
        output.append({
            "input_record_index": index, "sample_id": row.get("sample_id"), "attempt": row.get("attempt"),
            "data_source": source, "source_split": row.get("source_split"), "partition": row.get("partition"),
            "question": row["question"], "golden_answers": gold, "prediction": answer,
            "original_status": row.get("status"), "original_reject_reason": row.get("reject_reason"),
            "previous_recorded_em": row.get("answer_em", row.get("em")),
            "rescored_original_gold": metrics, "sft_answer_check_only": gate,
            "diagnostics": diagnostic,
        })
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "input": str(input_file.resolve()),
        "input_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "complete_lines_sha256": hashlib.sha256(b"".join(lines)).hexdigest(),
        "incomplete_trailing_line_ignored": ignored_tail,
        "reviewed_aliases_sha256": hashlib.sha256(Path(aliases_file).read_bytes()).hexdigest() if aliases_file else None,
        "counts": dict(counters), "by_source": {key: dict(value) for key, value in by_source.items()},
        "teacher_api_calls": 0, "review_api_calls": 0, "checkpoint_records_modified": 0,
        "trajectories_promoted_to_accepted": 0,
        "scope": "Answer-only diagnostics on saved attempt records, not an accuracy benchmark or a new SFT acceptance decision. Alias matches still require length and evidence/query review.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-aliases-file", type=Path)
    args = parser.parse_args()
    try:
        result = audit_scores(args.input, args.output_dir, args.reviewed_aliases_file)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
