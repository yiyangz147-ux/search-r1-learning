"""CPU numerical regression of the actual native and patched verl forwards.

Needs torch+transformers, but no GPU/verl import.  Set SEARCH_GRPO_TEST_VERL_ROOT
to a pinned checkout: only its AST function is loaded, avoiding Ray/CUDA setup.
"""

import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from patch_search_grpo_runtime import BASELINE_SHA256, PATCH_VERSION, patched_source
from search_grpo_metrics import trim_micro_batch_padding

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = Qwen3Config = Qwen3ForCausalLM = None


def _native_source():
    root = Path(os.environ.get("SEARCH_GRPO_TEST_VERL_ROOT", "/tmp/search-grpo-implementation/verl-0.6.1"))
    relative = "verl/workers/actor/dp_actor.py"
    content = (root / relative).read_bytes()
    if hashlib.sha256(content).hexdigest() != BASELINE_SHA256[relative]:
        content = (root / ".search-grpo-patches" / PATCH_VERSION / "original" / relative).read_bytes()
    assert hashlib.sha256(content).hexdigest() == BASELINE_SHA256[relative]
    return content.decode()


def _function(source):
    module = ast.parse(source)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch")
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "torch": torch,
        "verl_F": SimpleNamespace(entropy_from_logits=lambda logits:
                                  torch.logsumexp(logits, -1) - (logits.softmax(-1) * logits).sum(-1)),
        "logprobs_from_logits": lambda logits, labels, **kwargs:
            logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    }
    exec(compile(isolated, "<actual-pinned-verl-forward>", "exec"), namespace)
    return namespace["_forward_micro_batch"]


def _inputs():
    prompt = [0] * 4 + [5, 6, 7, 8, 9]
    response = list(range(31, 42)) + [0] * 21
    attention = torch.tensor([[0] * 4 + [1] * 16 + [0] * 21])
    return {"input_ids": torch.tensor([prompt + response]), "attention_mask": attention,
            "position_ids": (attention.cumsum(-1) - 1).clamp_min(0) * attention,
            "responses": torch.tensor([response])}


def _actor(function, model):
    class Recorder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model, self.widths = model, []

        def forward(self, **kwargs):
            self.widths.append(kwargs["input_ids"].shape[1])
            return self.model(**kwargs)
    actor = SimpleNamespace(actor_module=Recorder(model), use_remove_padding=False,
                            use_fused_kernels=False, ulysses_sequence_parallel_size=1,
                            device_name="cpu", param_dtype=torch.float64,
                            config=SimpleNamespace(entropy_checkpointing=False))
    actor._forward_micro_batch = MethodType(function, actor)
    return actor


@unittest.skipIf(torch is None, "CPU torch+transformers required for native Qwen3 equivalence")
class PaddingTrimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        torch.set_num_threads(1)
        source = _native_source()
        cls.native = staticmethod(_function(source))
        cls.patched = staticmethod(_function(patched_source("verl/workers/actor/dp_actor.py", source)))

    def model(self):
        config = Qwen3Config(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True,
                            use_sliding_window=False)
        config._attn_implementation = "sdpa"
        return Qwen3ForCausalLM(config).double().train()

    def forward(self, actor, inputs, *, entropy=True):
        with warnings.catch_warnings():
            # CPU float64 deliberately disables autocast for a strict reference.
            warnings.filterwarnings("ignore", message="In CPU autocast.*")
            return actor._forward_micro_batch(inputs, temperature=1., calculate_entropy=entropy)

    def test_actual_native_and_patched_qwen3_match_masked_loss_and_all_gradients(self):
        original = self.model()
        native = _actor(self.native, original)
        trimmed = _actor(self.patched, copy.deepcopy(original))
        inputs = _inputs()
        old_entropy, old_log_probs = self.forward(native, inputs)
        new_entropy, new_log_probs = self.forward(trimmed, inputs)
        self.assertEqual(native.actor_module.widths, [41])
        self.assertEqual(trimmed.actor_module.widths, [16])
        self.assertEqual(tuple(new_log_probs.shape), (1, 32))
        self.assertEqual(tuple(new_entropy.shape), (1, 32))
        self.assertLess(float((old_log_probs[:, :11] - new_log_probs[:, :11]).abs().max()), 1e-10)
        self.assertLess(float((old_entropy[:, :11] - new_entropy[:, :11]).abs().max()), 1e-10)
        self.assertTrue(torch.equal(new_log_probs[:, 11:], torch.zeros_like(new_log_probs[:, 11:])))
        mask = torch.tensor([[1.] * 4 + [0.] * 4 + [1.] * 3 + [0.] * 21], dtype=torch.float64)
        advantages = torch.tensor([[.8, -.4, .8, -.4] + [0.] * 4 + [.8, -.4, .8] + [0.] * 21])
        old = old_log_probs.detach() + torch.linspace(-.13, .13, 32)[None]
        reference = old_log_probs.detach() - .01
        def loss(log_probs, entropy):
            ratio = (log_probs - old).clamp(-20, 20).exp()
            pg = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(.8, 1.2))
            logratio = (reference - log_probs).clamp(-20, 20)
            kl = (logratio.exp() - logratio - 1).clamp(-10, 10)
            return ((pg - .001 * entropy + .001 * kl) * mask).sum() / mask.sum()
        old_loss, new_loss = loss(old_log_probs, old_entropy), loss(new_log_probs, new_entropy)
        self.assertLess(float((old_loss - new_loss).abs().detach()), 1e-10)
        old_loss.backward()
        new_loss.backward()
        for old_parameter, new_parameter in zip(native.actor_module.parameters(), trimmed.actor_module.parameters()):
            self.assertLess(float((old_parameter.grad - new_parameter.grad).abs().max()), 1e-10)

    def test_no_entropy_stays_none_and_batch_two_uses_full_native_path(self):
        actor = _actor(self.patched, self.model())
        entropy, log_probs = self.forward(actor, _inputs(), entropy=False)
        self.assertIsNone(entropy)
        self.assertEqual(tuple(log_probs.shape), (1, 32))
        self.assertEqual(actor.actor_module.widths, [16])
        inputs = {key: value.repeat(2, 1) for key, value in _inputs().items()}
        entropy, log_probs = self.forward(actor, inputs)
        self.assertEqual(actor.actor_module.widths, [16, 41])
        self.assertEqual(tuple(log_probs.shape), (2, 32))

    def test_uses_attention_mask_keeps_environment_and_original_positions(self):
        original = _inputs()
        trimmed, original_width = trim_micro_batch_padding(original)
        self.assertEqual(original_width, 32)
        self.assertTrue(torch.equal(trimmed["input_ids"], original["input_ids"][:, 4:20]))
        self.assertTrue(torch.equal(trimmed["position_ids"], original["position_ids"][:, 4:20]))
        self.assertTrue(torch.equal(trimmed["responses"], original["responses"][:, :11]))
        self.assertIsNone(trim_micro_batch_padding(trimmed), "No second recursion when padding is absent")
        original["attention_mask"][0, 12] = 0
        self.assertIsNone(trim_micro_batch_padding(original), "Non-edge holes stay on the native path")
        original = _inputs()
        original["multi_modal_inputs"] = {}
        self.assertIsNone(trim_micro_batch_padding(original))


if __name__ == "__main__":
    unittest.main()
