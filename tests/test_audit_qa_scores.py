"""Offline rescoring must never modify or promote paid generation attempts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_qa_scores import audit_scores


class AuditTests(unittest.TestCase):
    def test_snapshot_ignores_inflight_tail_and_never_promotes_or_rewrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "attempts.jsonl"
            record = {"sample_id": "fixture", "data_source": "hotpotqa", "source_split": "train",
                      "question": "Capital?", "golden_answers": ["Paris"], "final_answer": "not Paris",
                      "status": "rejected", "reject_reason": "answer_em_mismatch", "answer_em": False}
            raw = (json.dumps(record) + "\n" + '{"unfinished":').encode()
            source.write_bytes(raw)
            output = root / "audit"
            report = audit_scores(source, output)
            self.assertEqual(source.read_bytes(), raw)
            self.assertTrue(report["incomplete_trailing_line_ignored"])
            self.assertEqual(report["counts"]["attempt_records"], 1)
            self.assertEqual(report["counts"]["rescored_original_gold_EM_matches"], 0)
            self.assertEqual(report["counts"]["search_r1_sub_em_matches_diagnostic_only"], 1)
            self.assertEqual(report["trajectories_promoted_to_accepted"], 0)
            with self.assertRaisesRegex(ValueError, "new or empty"):
                audit_scores(source, output)


if __name__ == "__main__":
    unittest.main()
