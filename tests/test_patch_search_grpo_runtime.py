"""Hash-bound patch behavior and executable per-action budget regression tests."""

import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_search_grpo_runtime as runtime_patch


class PatchTests(unittest.TestCase):
    def test_actual_dump_generations_preserves_numpy_values_and_rejects_unknown_objects(self):
        import numpy as np
        from types import SimpleNamespace
        from search_grpo_metrics import _plain
        root = Path(os.environ.get("SEARCH_GRPO_TEST_VERL_ROOT", "/tmp/search-grpo-implementation/verl-0.6.1"))
        relative = "verl/trainer/ppo/ray_trainer.py"
        source = (root / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != runtime_patch.BASELINE_SHA256[relative]:
            for version in (runtime_patch.PATCH_VERSION, "search-grpo-v061-v4"):
                backup = root / ".search-grpo-patches" / version / "original" / relative
                if backup.exists():
                    source = backup.read_bytes()
                    break
        self.assertEqual(hashlib.sha256(source).hexdigest(), runtime_patch.BASELINE_SHA256[relative])
        module = ast.parse(runtime_patch.patched_source(relative, source.decode()))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_dump_generations")
        namespace = {"os": os, "json": json, "search_grpo_plain": _plain}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual-verl-dump-generations>", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            namespace["_dump_generations"](
                SimpleNamespace(global_steps=np.int64(9)),
                inputs=["问题一", "second"], outputs=["<think>é</think><answer>Paris</answer>", "raw"],
                gts=[["Paris", "City of Paris"], ["answer"]], scores=np.array([1., 0.], dtype=np.float32),
                reward_extra_infos_dict={"search_count": np.array([1, 2], dtype=np.int64),
                                         "nested": [{"token_ids": np.array([151667, 7, 151668]), "ok": np.bool_(True)},
                                                    {"scores": np.array([.25, .5], dtype=np.float64)}],
                                         "wrong_length": [100]}, dump_path=directory)
            rows = [json.loads(line) for line in (Path(directory) / "9.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [
                {"input": "问题一", "output": "<think>é</think><answer>Paris</answer>",
                 "gts": ["Paris", "City of Paris"], "score": 1., "step": 9, "search_count": 1,
                 "nested": {"token_ids": [151667, 7, 151668], "ok": True}},
                {"input": "second", "output": "raw", "gts": ["answer"], "score": 0., "step": 9,
                 "search_count": 2, "nested": {"scores": [.25, .5]}},
            ])
            with self.assertRaises(TypeError):
                namespace["_dump_generations"](
                    SimpleNamespace(global_steps=10), inputs=["q"], outputs=["a"], gts=[["a"]], scores=[1.],
                    reward_extra_infos_dict={"unknown": [object()]}, dump_path=directory)

    def test_generate_patch_executes_capped_budget_without_mutating_caller(self):
        before = (
            "async def generate(self, prompt_ids, sampling_params):\n"
            "        max_tokens = self.config.max_model_len - len(prompt_ids)\n"
            "        sampling_params[\"logprobs\"] = 0 if sampling_params.pop(\"logprobs\", False) else None\n"
            "        return SamplingParams(max_tokens=max_tokens, **sampling_params)\n"
        )
        source = "class Wrapper:\n    def __init__(self):\n        self.config.max_model_len = self.config.prompt_length + self.config.response_length\n\n" + before
        changed = runtime_patch.patched_source("x/vllm_async_server.py", source)
        namespace = {"SamplingParams": lambda **kwargs: kwargs}
        exec(compile(changed, "<budget-patch-test>", "exec"), namespace)
        from types import SimpleNamespace
        server = SimpleNamespace(config=SimpleNamespace(max_model_len=8192))
        params = {"max_tokens": 512, "logprobs": False, "top_p": .95}
        for prompt_length, expected in ((100, 512), (8000, 192)):
            result = asyncio.run(namespace["generate"](server, [1] * prompt_length, params))
            self.assertEqual(result["max_tokens"], expected)
            self.assertEqual(params, {"max_tokens": 512, "logprobs": False, "top_p": .95})
        with self.assertRaisesRegex(ValueError, "No positive generation budget"):
            asyncio.run(namespace["generate"](server, [1] * 8192, params))

    def test_apply_is_idempotent_backs_up_and_refuses_changes(self):
        relative = "x/vllm_async_server.py"
        original = (
            "class Engine:\n    def init(self):\n"
            "        self.config.max_model_len = self.config.prompt_length + self.config.response_length\n"
            "    def generate(self, prompt_ids, sampling_params):\n"
            "        max_tokens = self.config.max_model_len - len(prompt_ids)\n"
            "        sampling_params[\"logprobs\"] = 0 if sampling_params.pop(\"logprobs\", False) else None\n"
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(original)
            expected = {relative: hashlib.sha256(original).hexdigest()}
            with patch.object(runtime_patch, "BASELINE_SHA256", expected):
                first = runtime_patch.apply_patches(root)
                second = runtime_patch.apply_patches(root)
                self.assertEqual(first, second)
                self.assertEqual(Path(first["files"][0]["backup"]).read_bytes(), original)
                self.assertNotEqual(first["files"][0]["after_sha256"], first["files"][0]["before_sha256"])
                self.assertIn("--- a/x/vllm_async_server.py", Path(first["diff"]).read_text())
                self.assertIn("max_tokens = min", target.read_text())
                self.assertTrue(Path(first["diff"]).is_file())
                target.write_text(target.read_text() + "# changed by someone else\n")
                with self.assertRaisesRegex(ValueError, "changed after previous patch"):
                    runtime_patch.apply_patches(root)

    def test_unknown_baseline_and_anchor_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in runtime_patch.BASELINE_SHA256:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# modified upstream\n")
            with self.assertRaisesRegex(ValueError, "Unrecognized verl"):
                runtime_patch.apply_patches(root)
        with self.assertRaisesRegex(ValueError, "expected 1 occurrence"):
            runtime_patch.patched_source("x/vllm_async_server.py", "# no expected anchors\n")


if __name__ == "__main__":
    unittest.main()
