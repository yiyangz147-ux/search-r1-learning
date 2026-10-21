"""CPU regression at the real padded verl forward seam, plus gradient parity."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import patch_search_grpo_runtime as runtime_patch


def pristine_actor():
    root = Path(os.environ.get('SEARCH_GRPO_TEST_VERL_ROOT', '/tmp/search-grpo-implementation/verl-0.6.1'))
    relative = 'verl/workers/actor/dp_actor.py'
    candidates = [root / relative] + [root / '.search-grpo-patches' / version / 'original' / relative
        for version in ('search-grpo-v061-v6', 'search-grpo-v061-v5', 'search-grpo-v061-v4', 'search-grpo-v061-v3')]
    for path in candidates:
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == runtime_patch.BASELINE_SHA256[relative]:
            return path.read_text()
    raise AssertionError('Pinned pristine actor source not found')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch runtime')
class ChunkedLogitsTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(173)

    @staticmethod
    def native_kernels():
        from verl.utils import torch_functional as native
        return native

    def test_native_values_masked_loss_and_gradients_match(self):
        import torch
        from search_grpo_metrics import chunked_logprobs_entropy
        native = self.native_kernels()
        for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-6), (torch.bfloat16, .02)):
            for entropy_enabled in (False, True):
                original = torch.randn(2, 271, 47, dtype=dtype, requires_grad=True)
                candidate = original.detach().clone().requires_grad_(True)
                labels = torch.randint(0, 47, (2, 271))
                mask = (torch.arange(271) % 3 == 0).expand(2, -1)
                before_lp = native.logprobs_from_logits(original, labels, inplace_backward=False)
                before_entropy = native.entropy_from_logits(original) if entropy_enabled else None
                entropy, lp = chunked_logprobs_entropy(candidate, labels, calculate_entropy=entropy_enabled,
                            chunk_size=128, logprobs_fn=native.logprobs_from_logits, entropy_fn=native.entropy_from_logits)
                torch.testing.assert_close(lp, before_lp, atol=tolerance, rtol=tolerance)
                if entropy_enabled:
                    torch.testing.assert_close(entropy, before_entropy, atol=tolerance, rtol=tolerance)
                else:
                    self.assertIsNone(entropy)
                before_loss = ((before_lp - (.001 * before_entropy if entropy_enabled else 0)) * mask).sum() / mask.sum()
                after_loss = ((lp - (.001 * entropy if entropy_enabled else 0)) * mask).sum() / mask.sum()
                before_loss.backward(); after_loss.backward()
                torch.testing.assert_close(candidate.grad, original.grad, atol=tolerance, rtol=tolerance)

    def test_no_grad_sliced_multibatch_uses_views_and_bounded_native_chunks(self):
        import torch
        from search_grpo_metrics import chunked_logprobs_entropy
        native = self.native_kernels()
        full = torch.randn(2, 301, 53)
        logits = full[:, 17:-11, :]
        labels = torch.randint(0, 53, logits.shape[:-1])
        lengths = []
        def checked(value, target, **kwargs):
            lengths.append(value.shape[-2])
            self.assertLessEqual(value.shape[-2], 128)
            self.assertEqual(value.untyped_storage().data_ptr(), full.untyped_storage().data_ptr())
            return native.logprobs_from_logits(value, target, **kwargs)
        with torch.no_grad(), patch('torch.utils.checkpoint.checkpoint', side_effect=AssertionError('no_grad must not checkpoint')):
            entropy, lp = chunked_logprobs_entropy(logits, labels, logprobs_fn=checked)
        self.assertIsNone(entropy)
        self.assertEqual(lengths, [128, 128, 17, 128, 128, 17])
        torch.testing.assert_close(lp, native.logprobs_from_logits(logits, labels, inplace_backward=False))

    def test_checkpoint_is_nonreentrant_without_rng_and_two_dimensional_supported(self):
        import torch
        import torch.utils.checkpoint
        from search_grpo_metrics import chunked_logprobs_entropy
        original = torch.utils.checkpoint.checkpoint
        logits = torch.randn(259, 43, requires_grad=True)
        labels = torch.randint(0, 43, (259,))
        with patch('torch.utils.checkpoint.checkpoint', wraps=original) as mocked:
            entropy, lp = chunked_logprobs_entropy(logits, labels, calculate_entropy=True)
            (lp - .001 * entropy).mean().backward()
        self.assertEqual(mocked.call_count, 3)
        for call in mocked.call_args_list:
            self.assertIs(call.kwargs['use_reentrant'], False)
            self.assertIs(call.kwargs['preserve_rng_state'], False)
        self.assertEqual(lp.shape, labels.shape)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def actual_forward(self, source, native):
        import torch
        module = ast.parse(source)
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'DataParallelPPOActor')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_forward_micro_batch')
        namespace = {'torch': torch, 'verl_F': native, 'logprobs_from_logits': native.logprobs_from_logits}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<real-padded-actor-regression>', 'exec'), namespace)
        return namespace['_forward_micro_batch']

    def test_actual_padded_actor_forward_and_parameter_gradients_match_native(self):
        import torch
        import torch.utils.checkpoint
        from torch import nn
        native = self.native_kernels()
        pristine = pristine_actor()
        patched = runtime_patch.patched_source('verl/workers/actor/dp_actor.py', pristine)
        methods = [self.actual_forward(source, native) for source in (pristine, patched)]
        class Policy(nn.Module):
            def __init__(self):
                super().__init__(); self.embedding = nn.Embedding(47, 13); self.head = nn.Linear(13, 47, bias=False)
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
        policies = [Policy().double()]; policies.append(copy.deepcopy(policies[0]))
        ids = torch.randint(0, 47, (1, 281)); responses = ids[:, 17:]
        micro = {'input_ids': ids, 'responses': responses, 'attention_mask': torch.ones_like(ids),
                 'position_ids': torch.arange(ids.shape[-1])[None]}
        results = []
        for method, model in zip(methods, policies):
            actor = SimpleNamespace(actor_module=model, use_remove_padding=False, use_fused_kernels=False,
                        ulysses_sequence_parallel_size=1, device_name='cpu', param_dtype=torch.bfloat16,
                        config=SimpleNamespace(entropy_checkpointing=True))
            actor._forward_micro_batch = MethodType(method, actor)
            entropy, lp = actor._forward_micro_batch(micro, temperature=.9, calculate_entropy=True)
            mask = (torch.arange(responses.shape[-1]) % 4 == 0)[None]
            loss = ((lp - .001 * entropy) * mask).sum() / mask.sum()
            loss.backward(); results.append((entropy.detach(), lp.detach(), loss.detach()))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        for after, before in zip(policies[1].parameters(), policies[0].parameters()):
            torch.testing.assert_close(after.grad, before.grad, atol=1e-10, rtol=1e-10)
        # This assertion goes red on the old real padded branch: its LP call
        # sees all 264 response tokens rather than a bounded token chunk.
        seen = []
        def guard(value, target, **kwargs):
            seen.append(value.shape[-2]); self.assertLessEqual(value.shape[-2], 128)
            return native.logprobs_from_logits(value, target, **kwargs)
        bounded_native = SimpleNamespace(entropy_from_logits=native.entropy_from_logits, logprobs_from_logits=guard)
        method = self.actual_forward(patched, bounded_native)
        actor._forward_micro_batch = MethodType(method, actor)
        actor._forward_micro_batch(micro, temperature=.9, calculate_entropy=True)
        self.assertEqual(seen, [128, 128, 8])


if __name__ == '__main__':
    unittest.main()
