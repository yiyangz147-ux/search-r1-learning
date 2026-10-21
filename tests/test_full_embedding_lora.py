"""CPU contract tests for the full shared-embedding checkpoint extension.

Run in the training environment with CUDA_VISIBLE_DEVICES='' python this_file.
No downloaded model, original checkpoint modification, or GPU is needed.
"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback, TrainingArguments

from full_embedding_lora import (
    FullEmbeddingTrainer, METADATA_NAME, WEIGHTS_NAME,
    assert_full_embedding_tied, configure_full_embedding_training,
    load_full_embedding_adapter, load_full_embedding_weights, save_full_embedding_weights,
)


torch.set_num_threads(2)


def fresh_base():
    torch.manual_seed(412)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
                        pad_token_id=0, eos_token_id=2, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
    model.config.use_cache = False
    with torch.no_grad():
        model.get_input_embeddings().weight[51].copy_(model.get_input_embeddings().weight[50])
    return model


def fresh_model():
    model = get_peft_model(fresh_base(), LoraConfig(task_type=TaskType.CAUSAL_LM,
                           r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    configure_full_embedding_training(model)
    return model


def sample():
    ids = torch.tensor([[3, 4, 50, 5, 51, 6, 2]])
    labels = ids.clone()
    labels[:, :2] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


class TinyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {key: value[0].clone() for key, value in sample().items()}


def training_args(output_dir):
    return TrainingArguments(output_dir=str(output_dir), use_cpu=True, bf16=True,
                             max_steps=2, per_device_train_batch_size=1,
                             learning_rate=1e-3, lr_scheduler_type="constant", weight_decay=0.01,
                             save_strategy="steps", save_steps=1, logging_strategy="no",
                             eval_strategy="no", report_to=[], disable_tqdm=True,
                             seed=123, data_seed=123, dataloader_num_workers=0,
                             label_names=["labels"], save_safetensors=True)


class StopAfterOne(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
        return control


class FullEmbeddingTests(unittest.TestCase):
    def test_fp32_weight_bf16_activation_and_real_update(self):
        model = fresh_model()
        weight = assert_full_embedding_tied(model, require_trainable=True)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertEqual(model.get_input_embeddings()(torch.tensor([[1]])).dtype, torch.bfloat16)
        self.assertTrue(torch.equal(weight[50], weight[51]))
        original = weight.detach().clone()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss = model(**sample()).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(weight.grad.dtype, torch.float32)
        self.assertGreater(weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for name, p in model.named_parameters() if "lora_" in name))
        optimizer.step()
        self.assertIs(weight, model.get_output_embeddings().weight)
        self.assertFalse(torch.equal(original, weight))
        self.assertFalse(torch.equal(weight[50], weight[51]))
        self.assertEqual(optimizer.state[weight]["exp_avg"].dtype, torch.float32)
        self.assertEqual(optimizer.state[weight]["exp_avg_sq"].dtype, torch.float32)

    def test_save_reload_exact_and_inference_no_autocast(self):
        model = fresh_model()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        model(**sample()).loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=sample()["input_ids"]).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, save_embedding_layers=False)
            save_full_embedding_weights(model, directory)
            reloaded = load_full_embedding_adapter(fresh_base(), directory).eval()
            self.assertFalse(any(p.requires_grad for p in reloaded.parameters()))
            self.assertTrue(torch.equal(model.get_input_embeddings().weight,
                                        reloaded.get_input_embeddings().weight))
            with torch.no_grad():
                actual = reloaded(input_ids=sample()["input_ids"]).logits
                generated = reloaded.generate(input_ids=torch.tensor([[3, 4]]),
                                             max_new_tokens=2, do_sample=False)
            self.assertTrue(torch.equal(expected, actual))
            self.assertGreater(generated.shape[1], 2)
            self.assertIs(reloaded.get_input_embeddings().weight, reloaded.get_output_embeddings().weight)

    def test_in_place_restore_keeps_optimizer_reference_and_rejects_corruption(self):
        model = fresh_model()
        weight = model.get_input_embeddings().weight
        with tempfile.TemporaryDirectory() as directory:
            save_full_embedding_weights(model, directory)
            expected = weight.detach().clone()
            with torch.no_grad():
                weight.add_(1)
            load_full_embedding_weights(model, directory)
            self.assertIs(weight, model.get_input_embeddings().weight)
            self.assertTrue(torch.equal(weight, expected))
            path = Path(directory) / WEIGHTS_NAME
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                load_full_embedding_weights(model, directory)

    def test_supervised_position_logits_loss_matches_full(self):
        model = fresh_model().eval()
        data = sample()
        with torch.no_grad():
            full_loss = model(**data).loss
            positions = (data["labels"][:, 1:] != -100).any(dim=0).nonzero().flatten()
            logits = model(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                           logits_to_keep=positions).logits
            labels = data["labels"][:, positions + 1]
            sparse_loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                            labels.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(full_loss, sparse_loss, rtol=1e-6, atol=1e-6)

    def test_trainer_resume_and_best_checkpoint_restore_full_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = FullEmbeddingTrainer(model=fresh_model(), args=training_args(root / "continuous"),
                                              train_dataset=TinyDataset())
            continuous.train()
            reference = {name: p.detach().clone() for name, p in continuous.model.named_parameters()
                         if p.requires_grad}
            partial = FullEmbeddingTrainer(model=fresh_model(), args=training_args(root / "resume"),
                                           train_dataset=TinyDataset(), callbacks=[StopAfterOne()])
            partial.train()
            checkpoint = root / "resume" / "checkpoint-1"
            self.assertTrue((checkpoint / WEIGHTS_NAME).is_file())
            self.assertTrue((checkpoint / METADATA_NAME).is_file())
            from safetensors import safe_open

            with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt") as saved:
                self.assertTrue(all("lora_" in name for name in saved.keys()))
            with safe_open(str(checkpoint / WEIGHTS_NAME), framework="pt") as saved:
                self.assertEqual(list(saved.keys()), ["shared_embedding_weight"])
            resumed = FullEmbeddingTrainer(model=fresh_model(), args=training_args(root / "resume"),
                                           train_dataset=TinyDataset())
            resumed.train(resume_from_checkpoint=str(checkpoint))
            for name, p in resumed.model.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p, reference[name], rtol=0, atol=0)
            # Exercise the same override used by load_best_model_at_end, after
            # deliberately altering the live shared parameter.
            saved_weight = partial.model.get_input_embeddings().weight.detach().clone()
            weight = resumed.model.get_input_embeddings().weight
            with torch.no_grad():
                weight.add_(1)
            resumed.state.best_model_checkpoint = str(checkpoint)
            resumed.state.best_metric = 1.0
            resumed._load_best_model()
            self.assertIs(weight, resumed.model.get_output_embeddings().weight)
            self.assertTrue(torch.equal(weight, saved_weight))


if __name__ == "__main__":
    unittest.main(verbosity=2)
