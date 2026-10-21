"""CPU mocks for UUID/numeric/empty CVD and exact-source patch safety."""

import ast
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_search_grpo_vllm as adapter


SOURCE = '''class Platform:
    device_control_env_var = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        if cls.device_control_env_var in os.environ and os.environ[cls.device_control_env_var] != "":
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]
            return int(physical_device_id)
        else:
            return device_id
'''


class VllmPatchTests(unittest.TestCase):
    def setUp(self):
        namespace = {"os": os}
        exec(compile(adapter.patched_source(SOURCE), "<uuid-patch-mock>", "exec"), namespace)
        self.platform = namespace["Platform"]

    def test_uuid_maps_to_physical_index_and_always_shuts_down_nvml(self):
        calls = []
        nvml = SimpleNamespace(nvmlInit=lambda: calls.append("init"),
                               nvmlDeviceGetHandleByUUID=lambda value: calls.append(value) or "handle",
                               nvmlDeviceGetIndex=lambda handle: 5,
                               nvmlShutdown=lambda: calls.append("shutdown"))
        with patch.dict(sys.modules, {"vllm.utils": SimpleNamespace(import_pynvml=lambda: nvml)}):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-last"}):
                self.assertEqual(self.platform.device_id_to_physical_device_id(1), 5)
        self.assertEqual(calls, ["init", b"GPU-last", "shutdown"])
        nvml.nvmlDeviceGetHandleByUUID = lambda value: (_ for _ in ()).throw(ValueError("unknown UUID"))
        calls.clear()
        with patch.dict(sys.modules, {"vllm.utils": SimpleNamespace(import_pynvml=lambda: nvml)}):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-missing"}):
                with self.assertRaisesRegex(ValueError, "unknown UUID"):
                    self.platform.device_id_to_physical_device_id(0)
        self.assertEqual(calls, ["init", "shutdown"])

    def test_numeric_empty_and_unset_keep_original_path_without_nvml(self):
        with patch.dict(sys.modules, {"vllm.utils": None}):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,4,5"}):
                self.assertEqual(self.platform.device_id_to_physical_device_id(0), 2)
                self.assertEqual(self.platform.device_id_to_physical_device_id(3), 5)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                self.assertEqual(self.platform.device_id_to_physical_device_id(2), 2)
            with patch.dict(os.environ, clear=True):
                self.assertEqual(self.platform.device_id_to_physical_device_id(3), 3)

    def test_backup_hash_manifest_and_idempotence_only_in_grpo_venv(self):
        original = SOURCE.encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / ".venv-grpo"
            target = venv / "lib/python3.11/site-packages" / adapter.RELATIVE_FILE
            target.parent.mkdir(parents=True)
            target.write_bytes(original)
            metadata = root / "grpo/vllm-patches/vllm011-uuid-v1"
            with patch.object(adapter, "BASELINE_SHA256", hashlib.sha256(original).hexdigest()):
                first = adapter.apply_patch(venv, metadata)
                self.assertEqual(adapter.apply_patch(venv, metadata), first)
                self.assertEqual(Path(first["backup"]).read_bytes(), original)
                self.assertEqual(first["after_sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
                self.assertTrue((metadata / "manifest.json").is_file())
                self.assertIn("nvmlDeviceGetIndex", Path(first["diff"]).read_text())
                target.write_text(target.read_text() + "# external modification\n")
                with self.assertRaisesRegex(ValueError, "changed after"):
                    adapter.apply_patch(venv, metadata)
            with self.assertRaisesRegex(ValueError, "isolated .venv-grpo"):
                adapter.apply_patch(root / ".venv", metadata)

    def test_unknown_source_and_nonunique_anchor_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adapter.patched_source("# no anchor\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adapter.patched_source(SOURCE + SOURCE)


if __name__ == "__main__":
    unittest.main()
