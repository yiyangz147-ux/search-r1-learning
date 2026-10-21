"""CPU-only checks for independent scoring, group integrity and dev selection."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_grpo_metrics import (
    audit_actor_parameters, audit_batch, cast_model_parameters_preserving_buffers,
    record_training, record_validation,
)
from search_grpo_reward import score_record


def trajectory(sample_id, *, answer="Paris", source="nq", with_retrieval=False):
    events = []
    if with_retrieval:
        events.extend([
            {"role": "assistant", "content": "<think>Find it.</think><search>France capital</search>",
             "generated_tokens": 4, "response_token_start": 0, "response_token_end": 4},
            {"role": "environment", "kind": "retrieval", "included_in_context": True,
             "content": "<information><answer>Paris</answer></information>",
             "response_token_start": 4, "response_token_end": 9},
        ])
    start = 9 if with_retrieval else 0
    events.append({"role": "assistant", "content": f"<think>Answer it.</think><answer>{answer}</answer>",
                   "generated_tokens": 10, "response_token_start": start, "response_token_end": start + 10})
    record = {"sample_id": sample_id, "question": f"Question {sample_id}?", "data_source": source,
              "events": events, "status": "ok", "answer": answer,
              "generated_tokens": 14 if with_retrieval else 10}
    record.update(score_record(record, ["Paris"], source))
    mask = ([1] * 4 + [0] * 5 if with_retrieval else []) + [1] * 10 + [0] * 3
    return record, mask


class TokenScores(list):
    def sum(self, axis):
        assert axis == -1
        return [sum(row) for row in self]


def batch_for(specs):
    records, masks = zip(*(trajectory(**spec) for spec in specs))
    metadata = {
        "search_grpo_trajectory": list(records),
        "data_source": [row["data_source"] for row in records],
        "uid": ["uid:" + row["sample_id"] for row in records],
        "extra_info": [{"sample_id": row["sample_id"], "question": row["question"]} for row in records],
        "reward_model": [{"ground_truth": {"target": ["Paris"]}} for _ in records],
        "format_reward": [row["format_reward"] for row in records],
    }
    return SimpleNamespace(non_tensor_batch=metadata, batch={
        "response_mask": list(masks),
        "token_level_scores": TokenScores([[row["score"]] for row in records]),
    })


def fake_trainer(output, ids=("one", "two"), step=0, batch_size=2):
    config = {"trainer": {"default_local_dir": str(output)},
              "data": {"train_batch_size": batch_size},
              "actor_rollout_ref": {"model": {"path": str(output / "merged-base")},
                                    "rollout": {"n": 5, "val_kwargs": {"n": 1}}}}
    (output / "merged-base").mkdir(exist_ok=True)
    dataset = SimpleNamespace(rows=[{"extra_info": {"sample_id": sid}} for sid in ids])
    class Dataset:
        rows = dataset.rows
        def __len__(self):
            return len(self.rows)
    return SimpleNamespace(config=config, val_dataset=Dataset(), global_steps=step)


class FakeTensor:
    """Represent real-shaped parameter metadata without allocating 4B weights."""
    def __init__(self, shape, *, dtype="torch.float32", nonzero=0):
        self.shape, self.dtype, self.nonzero = tuple(shape), dtype, nonzero

    def to(self, *, dtype):
        return FakeTensor(self.shape, dtype=dtype, nonzero=self.nonzero)


class FakeParameter:
    def __init__(self, shape, *, dtype="torch.float32", nonzero=0, trainable=False):
        self.data = FakeTensor(shape, dtype=dtype, nonzero=nonzero)
        self.requires_grad = trainable

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def numel(self):
        import math
        return math.prod(self.shape)

    def is_floating_point(self):
        return True

    def detach(self):
        return self

    def count_nonzero(self):
        return SimpleNamespace(item=lambda: self.data.nonzero)


def fake_actor(directory):
    root = Path(directory)
    model_path = root / "merged"
    model_path.mkdir()
    expected_hash = "a" * 64
    manifest = {"complete": True, "shared_embedding": {"tensor_sha256": expected_hash},
                "non_parameter_buffer_dtypes": {"model.rotary_emb.inv_freq": "torch.float32"}}
    (model_path / "sft_export_manifest.json").write_text(json.dumps(manifest))
    class Model:
        def __init__(self):
            self.embedding = SimpleNamespace(weight=FakeParameter((151936, 2560), dtype="torch.bfloat16"))
            self.head = self.embedding
            self.buffer = FakeTensor((64,))
            self.peft_config = {"default": {"r": 32, "lora_alpha": 16, "lora_dropout": 0.,
                                            "target_modules": {"q_proj", "v_proj"}, "bias": "none"}}
            self.params = {"base_model.model.model.embed_tokens.weight": self.embedding.weight}
            for layer in range(36):
                for projection, output in (("q_proj", 4096), ("v_proj", 1024)):
                    prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
                    self.params[prefix + ".lora_A.default.weight"] = FakeParameter((32, 2560), trainable=True, nonzero=1)
                    self.params[prefix + ".lora_B.default.weight"] = FakeParameter((output, 32), trainable=True)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.head

        def named_parameters(self):
            return iter(self.params.items())

        def parameters(self):
            return iter(self.params.values())

        def named_buffers(self):
            return iter([("base_model.model.model.rotary_emb.inv_freq", self.buffer)])
    return Model(), {"model": {"path": str(model_path), "lora_rank": 32, "lora_alpha": 16,
                               "lora_adapter_path": None}}, expected_hash


class MetricTests(unittest.TestCase):
    def test_parameter_audit_measures_frozen_shared_embedding_and_full_qv_set(self):
        with tempfile.TemporaryDirectory() as directory:
            model, config, expected = fake_actor(directory)
            output = Path(directory) / "output"
            record = audit_actor_parameters(model, config, output_dir=output, tensor_hasher=lambda _: expected)
            self.assertTrue(record["embedding_frozen"])
            self.assertTrue(record["shared_embedding_tied"])
            self.assertTrue(record["only_qv_lora_trainable"])
            self.assertEqual(record["trainable_count"], 11796480)
            self.assertEqual(record["initial_lora_b_nonzero"], 0)
            self.assertEqual(record["rotary_buffer_dtypes"], {"model.rotary_emb.inv_freq": "torch.float32"})
            self.assertEqual(len(record["trainable_parameters"]), 144)
            self.assertTrue((output / "runtime_parameter_audit.json").is_file())
            self.assertTrue((output / "runtime_parameter_audit_rank0.json").is_file())
            other = audit_actor_parameters(model, config, rank=1, output_dir=output,
                                           tensor_hasher=lambda _: self.fail("Only rank0 hashes the full tensor"))
            self.assertFalse(other["embedding_hash_verified_on_this_rank"])
            self.assertTrue((output / "runtime_parameter_audit_rank1.json").is_file())

    def test_parameter_audit_rejects_wrong_weight_loading_trainable_base_nonzero_b_and_rope(self):
        for mutation, message in (
            (lambda model: setattr(model.embedding.weight, "requires_grad", True), "tied and frozen"),
            (lambda model: setattr(model, "head", SimpleNamespace(weight=FakeParameter((151936, 2560)))), "tied and frozen"),
            (lambda model: model.params.update({"base_model.model.norm.weight": FakeParameter((2560,), trainable=True)}), "Unexpected trainable"),
            (lambda model: setattr(next(p.data for n, p in model.params.items() if ".lora_B." in n), "nonzero", 1), "must actually be zero"),
            (lambda model: setattr(model.buffer, "dtype", "torch.bfloat16"), "buffer dtypes differ"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                model, config, expected = fake_actor(directory)
                mutation(model)
                with self.assertRaisesRegex(ValueError, message):
                    audit_actor_parameters(model, config, output_dir=Path(directory) / "output", tensor_hasher=lambda _: expected)
        with tempfile.TemporaryDirectory() as directory:
            model, config, _ = fake_actor(directory)
            with self.assertRaisesRegex(ValueError, "tensor SHA256 differs"):
                audit_actor_parameters(model, config, output_dir=Path(directory) / "output", tensor_hasher=lambda _: "b" * 64)

    def test_parameter_only_cast_keeps_original_fp32_rotary_object(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, _ = fake_actor(directory)
            original = model.buffer
            cast_model_parameters_preserving_buffers(model, "torch.bfloat16")
            self.assertIs(model.buffer, original)
            self.assertEqual(model.buffer.dtype, "torch.float32")
            self.assertTrue(all(param.dtype == "torch.bfloat16" for param in model.parameters()))

    def test_role_separated_recompute_does_not_reward_observation_answer(self):
        batch = batch_for([{"sample_id": "one", "answer": "wrong", "with_retrieval": True}])
        rows = audit_batch(batch, [0], step=0, split="validation")
        self.assertEqual(rows[0]["score"], 0)
        self.assertEqual(rows[0]["search_count"], 1)
        self.assertEqual(rows[0]["format_reward"], 1)

    def test_mismatched_verl_or_stored_score_raises(self):
        batch = batch_for([{"sample_id": "one"}])
        with self.assertRaisesRegex(ValueError, "outcome score"):
            audit_batch(batch, [0], step=0, split="validation")
        batch.non_tensor_batch["search_grpo_trajectory"][0]["score"] = 0
        with self.assertRaisesRegex(ValueError, "trajectory.score"):
            audit_batch(batch, [1], step=0, split="validation")

    def test_observation_supervision_or_padding_supervision_raises(self):
        for location in (4, -1):
            batch = batch_for([{"sample_id": "one", "with_retrieval": True}])
            batch.batch["response_mask"][0][location] = 1
            with self.assertRaisesRegex(ValueError, "response_mask supervises"):
                audit_batch(batch, [1], step=0, split="validation")

    def test_wrong_question_or_sample_binding_raises(self):
        batch = batch_for([{"sample_id": "one"}])
        batch.non_tensor_batch["extra_info"][0]["sample_id"] = "other"
        with self.assertRaisesRegex(ValueError, "sample_id"):
            audit_batch(batch, [1], step=0, split="validation")

    def test_full_validation_selection_includes_step_zero_and_tie_earlier(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = fake_trainer(output)
            batch = batch_for([{"sample_id": "one"}, {"sample_id": "two", "answer": "wrong", "source": "hotpotqa"}])
            rows = audit_batch(batch, [1, 0], step=0, split="validation")
            metrics = record_validation(trainer, rows)
            self.assertEqual(metrics["val/answer_em"], .5)
            self.assertEqual(metrics["val/format_reward"], 1)
            self.assertEqual(metrics["val/nq_answer_em"], 1)
            best = json.loads((output / "best_checkpoint.json").read_text())
            self.assertEqual(best["global_step"], 0)
            self.assertEqual(best["checkpoint_kind"], "merged_sft_base_zero_grpo_increment")
            trainer.global_steps = 50
            rows = [dict(row, global_step=50) for row in rows]
            record_validation(trainer, rows)
            self.assertEqual(json.loads((output / "best_checkpoint.json").read_text())["global_step"], 0)
            batch = batch_for([{"sample_id": "one"}, {"sample_id": "two", "source": "hotpotqa"}])
            record_validation(trainer, audit_batch(batch, [1, 1], step=50, split="validation"))
            best = json.loads((output / "best_checkpoint.json").read_text())
            self.assertEqual(best["global_step"], 50)
            self.assertEqual(best["val_answer_em"], 1)
            self.assertFalse(best["checkpoint_ready_at_validation"])
            self.assertTrue(best["actor_checkpoint"].endswith("global_step_50/actor"))
            self.assertEqual(len((output / "validation_history.jsonl").read_text().splitlines()), 3)
            self.assertEqual(len(list((output / "audit").glob("validation-step000050*.jsonl"))), 2)

    def test_validation_rejects_missing_duplicate_and_foreign_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fake_trainer(Path(directory))
            batch = batch_for([{"sample_id": "one"}, {"sample_id": "two"}])
            rows = audit_batch(batch, [1, 1], step=0, split="validation")
            for bad in (rows[:1], [rows[0], rows[0]], [rows[0], dict(rows[1], sample_id="foreign")]):
                with self.assertRaisesRegex(ValueError, "every selected dev question"):
                    record_validation(trainer, bad)

    def test_complete_group_metrics_and_raw_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = fake_trainer(output, step=50, batch_size=3)
            (output / "global_step_50" / "actor").mkdir(parents=True)
            specs = ([{"sample_id": "zero", "answer": "wrong"}] * 5
                     + [{"sample_id": "all"}] * 5
                     + [{"sample_id": "mixed", "answer": "Paris" if i < 2 else "wrong", "source": "hotpotqa"}
                        for i in range(5)])
            batch = batch_for(specs)
            # Balanced DP changes order, which must not change grouping.
            for values in batch.non_tensor_batch.values():
                values.reverse()
            batch.batch["response_mask"].reverse()
            batch.batch["token_level_scores"].reverse()
            metrics = record_training(trainer, batch, {"actor/kl_loss": .1})
            self.assertAlmostEqual(metrics["train/total_reward"], 7/15)
            self.assertAlmostEqual(metrics["train/group_all_zero_fraction"], 1/3)
            self.assertAlmostEqual(metrics["train/group_all_one_fraction"], 1/3)
            self.assertAlmostEqual(metrics["train/group_mixed_fraction"], 1/3)
            self.assertAlmostEqual(metrics["train/pass_at_5"], 2/3)
            self.assertEqual(metrics["train/format_reward"], 1)
            lines = (output / "audit" / "train-step000050.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 15)
            self.assertIn("events", json.loads(lines[0]))
            self.assertTrue(json.loads((output / "last_checkpoint.json").read_text())["checkpoint_ready"])

    def test_bad_group_member_or_uid_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = fake_trainer(Path(directory), step=1)
            batch = batch_for([{"sample_id": "one"}] * 5 + [{"sample_id": "two"}] * 5)
            batch.non_tensor_batch["uid"][0] = "new-group"
            with self.assertRaisesRegex(ValueError, "GRPO group members"):
                record_training(trainer, batch, {})
            batch.non_tensor_batch["uid"] = ["one-group"] * 10
            with self.assertRaisesRegex(ValueError, "GRPO group members"):
                record_training(trainer, batch, {})

    def test_main_run_has_100_training_trajectories_and_all_200_dev_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ids = tuple(f"dev-{index}" for index in range(200))
            trainer = fake_trainer(output, ids=ids, step=1, batch_size=20)
            specs = [{"sample_id": f"train-{question}", "source": "nq" if question < 10 else "hotpotqa",
                      "answer": "Paris" if candidate == 0 else "wrong"}
                     for question in range(20) for candidate in range(5)]
            metrics = record_training(trainer, batch_for(specs), {})
            self.assertEqual(metrics["train/trajectory_count"], 100)
            self.assertEqual(metrics["train/question_count"], 20)
            self.assertEqual(metrics["train/answer_reward"], .2)
            self.assertEqual(metrics["train/group_mixed_fraction"], 1)
            self.assertEqual(metrics["train/pass_at_5"], 1)
            val_batch = batch_for([{"sample_id": sid, "source": "nq" if i < 100 else "hotpotqa"}
                                   for i, sid in enumerate(ids)])
            val = record_validation(trainer, audit_batch(val_batch, [1] * 200, step=1, split="validation"))
            self.assertEqual(val["val/trajectory_count"], 200)
            self.assertEqual(val["val/nq_count"], 100)
            self.assertEqual(val["val/hotpotqa_count"], 100)


if __name__ == "__main__":
    unittest.main()
