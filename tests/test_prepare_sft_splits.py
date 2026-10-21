from collections import Counter
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_sft_splits import ExactPrefixIndex, clean, jaccard, partition, shingles


def item(source, index, question, split="train"):
    return {"sample_id": f"{source}-{split}-{index}", "data_source": source,
            "source_id": str(index), "source_index": index, "source_split": split,
            "question": question, "golden_answers": ["Paris"], "stratum": "all"}


class CleaningTests(unittest.TestCase):
    def test_exact_prefix_guard_matches_brute_force(self):
        rng = random.Random(42)
        universe = [f"feature-{i}" for i in range(20)]
        sets = [frozenset(rng.sample(universe, rng.randint(1, 14))) for _ in range(160)]
        for threshold in [0.5, 0.85, 1.0]:
            index = ExactPrefixIndex(threshold, Counter(feature for group in sets for feature in group))
            inserted = []
            for n, features in enumerate(sets):
                brute = any(jaccard(features, other) + 1e-12 >= threshold for other in inserted)
                found = index.query(features)
                self.assertEqual(bool(found), brute, (threshold, features))
                if found:
                    self.assertGreaterEqual(found[1] + 1e-12, threshold)
                index.insert(str(n), features, {})
                inserted.append(features)

    def test_hash_near_dedup_and_evaluation_priority(self):
        base = " ".join(f"word{i}" for i in range(24))
        evaluation = [item("nq", 0, base, "test")]
        train = [item("nq", 0, base.upper() + "?"),
                 item("nq", 1, base + " extra"),
                 item("nq", 2, "Who is the president of this fictional island"),
                 item("hotpotqa", 0, "Who is the president of this fictional island?"),
                 item("hotpotqa", 1, "Which scientist won the award for a discovery")]
        ev, eligible, counts, audit, features, settings = clean(evaluation, train, progress=False)
        self.assertEqual(counts["nq"]["evaluation_exact_overlap"], 1)
        self.assertEqual(counts["nq"]["evaluation_near_overlap"], 1)
        self.assertEqual(counts["hotpotqa"]["exact_duplicate"], 1)
        self.assertEqual(len(eligible), 2)
        self.assertEqual(len(audit), 3)

    def test_disjoint_partitions_and_guard_for_missed_lsh_pairs(self):
        # Deliberately supply an eligible pair LSH might have failed to recall.
        base = " ".join(f"entity{i}" for i in range(24))
        evaluation = [item("nq", 0, base, "test")]
        eligible = [item("nq", 0, base + " extra")]
        for source in ["nq", "hotpotqa"]:
            eligible.extend(item(source, i+1, f"topic{source}{i} author{source}{i} title{source}{i} question{source}{i}") for i in range(10))
        features = {row["sample_id"]: shingles(row["question"]) for row in evaluation + eligible}
        targets = {source: {name: 1 for name in ["validation", "grpo", "sft_initial", "sft_reserve"]}
                   for source in ["nq", "hotpotqa"]}
        audit = []
        # Change seed until the planted near overlap is inspected for a partition.
        found = False
        for seed in range(10):
            output, rejected = partition(evaluation, eligible, features, 0.85, seed, targets, audit)
            found |= bool(rejected)
            selected = [row for group in output.values() for row in group]
            self.assertEqual(len(selected), 8)
            self.assertEqual(len({row["sample_id"] for row in selected}), 8)
            for name, group in output.items():
                self.assertEqual(Counter(row["data_source"] for row in group), {"nq": 1, "hotpotqa": 1})
            for i, row in enumerate(selected):
                for other in selected[i+1:] + evaluation:
                    self.assertLess(jaccard(features[row["sample_id"]], features[other["sample_id"]]), 0.85)
        self.assertTrue(found, "The exact guard must catch the planted held-out overlap")


if __name__ == "__main__":
    unittest.main()
