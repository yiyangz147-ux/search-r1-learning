#!/usr/bin/env python3
"""Verify the ready-to-run package with standard-library SHA256 checks."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "package_manifest.json").read_text())
    files = manifest["files"]
    if not files:
        parser.exit(1, "Empty package manifest\n")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            parser.exit(1, f"Invalid package path: {name}\n")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            parser.exit(1, f"Missing package file: {name}\n")
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                count += len(block)
        if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            parser.exit(1, f"Package checksum mismatch: {name}\n")
    print(f"Verified {len(files)} package files; build={manifest.get('build', 'unknown')}")


if __name__ == "__main__":
    main()
