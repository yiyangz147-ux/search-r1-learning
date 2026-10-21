"""Actual native dataclass conversion and validation of the complete run config.

CPU only: this loads the real exported tokenizer/HF config, never model weights.
Run in .venv-grpo with CUDA_VISIBLE_DEVICES='' and project scripts on PYTHONPATH.
"""

from dataclasses import fields
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any
import unittest


RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None
                        for name in ("verl", "hydra", "omegaconf", "torch"))


@unittest.skipUnless(RUNTIME_AVAILABLE, "Full .venv-grpo runtime required")
class ActualDataclassConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        root = Path(os.environ.get("SEARCH_GRPO_ROOT", Path(__file__).resolve().parents[1]))
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
            cls.config = compose(config_name="search_grpo_v1")
        OmegaConf.resolve(cls.config)

    def test_full_hf_model_dataclass_keeps_actual_qv_list_and_loads_exported_tokenizer(self):
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig
        typed = omega_conf_to_dataclass(self.config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
        self.assertIsInstance(typed, HFModelConfig)
        self.assertEqual(list(typed.target_modules), ["q_proj", "v_proj"])
        self.assertIs(next(field.type for field in fields(HFModelConfig) if field.name == "target_modules"), Any)
        self.assertEqual(typed.lora_rank, 32)
        self.assertEqual(typed.lora_alpha, 16)
        self.assertIsNone(typed.lora_adapter_path)
        self.assertIsNotNone(typed.tokenizer)
        self.assertEqual(typed.hf_config.model_type, "qwen3")
        self.assertTrue(typed.share_embeddings_and_output_weights)
        self.assertEqual(typed.override_config["attn_implementation"], "sdpa")

    def test_full_actor_and_rollout_dataclass_conversion_and_native_validation(self):
        from verl.utils.config import omega_conf_to_dataclass, validate_config
        from verl.workers.config import FSDPActorConfig, RolloutConfig, FSDPEngineConfig
        # These roles use their YAML _target_ through native Hydra instantiate;
        # unlike the HF model, native workers do not use explicit structured merge.
        actor = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        rollout = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout)
        self.assertIsInstance(actor, FSDPActorConfig)
        self.assertIsInstance(rollout, RolloutConfig)
        self.assertEqual(actor.ppo_mini_batch_size, 20)
        self.assertEqual(actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertEqual(rollout.n, 5)
        self.assertEqual(rollout.val_kwargs.n, 1)
        self.assertFalse(rollout.val_kwargs.do_sample)
        self.assertEqual(rollout.max_model_len, 8192)
        self.assertEqual(rollout.response_length, 8192)
        # verl v0.6.1 has no RefConfig dataclass: its actual ref role uses this
        # DictConfig plus the same FSDPEngineConfig conversion as fsdp_workers.
        reference = self.config.actor_rollout_ref.ref
        engine = omega_conf_to_dataclass(reference.fsdp_config)
        self.assertIsInstance(engine, FSDPEngineConfig)
        self.assertEqual(reference.strategy, "fsdp")
        self.assertEqual(reference.log_prob_micro_batch_size_per_gpu, 1)
        self.assertEqual(reference.ulysses_sequence_parallel_size, 1)
        self.assertTrue(engine.param_offload)
        actor.validate(4, self.config.data.train_batch_size, self.config.actor_rollout_ref.model)
        validate_config(self.config, use_reference_policy=True, use_critic=False)


if __name__ == "__main__":
    unittest.main()
