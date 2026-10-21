#!/usr/bin/env python3
"""Hash-bound vLLM 0.11 device-number adapter for this experiment's UUID CVD.

Only .venv-grpo's vllm/platforms/interface.py is changed.  UUID resolution uses
NVML, without initializing a CUDA context; numeric/unset/empty mappings retain
the exact upstream path.  No training or inference is launched by this script.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil


PATCH_VERSION = "vllm011-uuid-v1"
BASELINE_SHA256 = "ff142312173c173dae20a8303cd10f9c29fa255862396655b6888d2fdb73fae5"
RELATIVE_FILE = "vllm/platforms/interface.py"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def patched_source(source):
    before = "            return int(physical_device_id)\n"
    after = (
        "            # Search-R1: retain the approved UUID CUDA_VISIBLE_DEVICES mapping.\n"
        "            if physical_device_id.startswith(\"GPU-\"):\n"
        "                from vllm.utils import import_pynvml\n"
        "                pynvml = import_pynvml()\n"
        "                pynvml.nvmlInit()\n"
        "                try:\n"
        "                    handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id.encode(\"ascii\"))\n"
        "                    return int(pynvml.nvmlDeviceGetIndex(handle))\n"
        "                finally:\n"
        "                    pynvml.nvmlShutdown()\n"
        "            return int(physical_device_id)\n"
    )
    if source.count(before) != 1:
        raise ValueError("Pinned vLLM source must have exactly one physical-device integer conversion")
    result = source.replace(before, after, 1)
    ast.parse(result)
    return result


def apply_patch(venv_root, metadata_dir):
    venv_root, metadata_dir = Path(venv_root).resolve(), Path(metadata_dir).resolve()
    if venv_root.name != ".venv-grpo":
        raise ValueError("This compatibility patch may only modify the isolated .venv-grpo")
    targets = list((venv_root / "lib").glob("python*/site-packages/" + RELATIVE_FILE))
    if len(targets) != 1:
        raise ValueError("Expected exactly one installed vLLM interface.py inside .venv-grpo")
    target = targets[0]
    target.resolve().relative_to(venv_root)
    backup = metadata_dir / "original" / RELATIVE_FILE
    current = target.read_bytes()
    if sha256(current) == BASELINE_SHA256:
        original = current
    elif backup.is_file() and sha256(backup.read_bytes()) == BASELINE_SHA256:
        original = backup.read_bytes()
    else:
        raise ValueError("Unrecognized vLLM 0.11 interface.py; refusing to modify it")
    changed = patched_source(original.decode()).encode()
    if current not in (original, changed):
        raise ValueError("vLLM interface.py changed after the recorded compatibility patch")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = metadata_dir / "manifest.json"
    diff_file = metadata_dir / "changes.patch"
    manifest = {"patch_version": PATCH_VERSION, "upstream": "vllm 0.11.0",
                "target": str(target), "before_sha256": sha256(original),
                "after_sha256": sha256(changed), "backup": str(backup), "diff": str(diff_file),
                "scope": "UUID device-number compatibility only; no algorithm or GPU whitelist changes"}
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if any(previous.get(key) != value for key, value in manifest.items()):
            raise ValueError("Existing vLLM patch manifest disagrees with this fixed patch")
    else:
        previous = None
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError("Original vLLM backup was modified")
    if not backup.exists():
        backup.write_bytes(original)
        shutil.copystat(target, backup)
    if current != changed:
        temporary = target.with_name(target.name + ".search-grpo-tmp")
        temporary.write_bytes(changed)
        shutil.copystat(target, temporary)
        temporary.replace(target)
    diff_file.write_text("".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), changed.decode().splitlines(keepends=True),
        fromfile="a/" + RELATIVE_FILE, tofile="b/" + RELATIVE_FILE)))
    if previous is not None:
        return previous
    manifest["applied_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--venv-root", type=Path, default=root / ".venv-grpo")
    parser.add_argument("--metadata-dir", type=Path,
                        default=root / "grpo/vllm-patches" / PATCH_VERSION)
    args = parser.parse_args()
    print(json.dumps(apply_patch(args.venv_root, args.metadata_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
