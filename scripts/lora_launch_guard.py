#!/usr/bin/env python3
"""Read-only launch checks; never import Torch, initialize CUDA, or contact W&B."""

import argparse
from collections import Counter
import csv
import json
import netrc
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit


def read_json(path):
    return json.loads(Path(path).read_text())


def wandb_credentials_present():
    if os.environ.get("WANDB_API_KEY", "").strip():
        return True
    host = urlsplit(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        path = Path(os.environ.get("NETRC", str(Path.home() / ".netrc")))
        auth = netrc.netrc(str(path)).authenticators(host)
        return bool(auth and auth[2])
    except (OSError, netrc.NetrcParseError, TypeError):
        return False


def configured_gpus(root):
    config = read_json(root / "configs/lora_sft_train.json")
    hardware = read_json(root / "configs/lora_sft_gpus.json")
    selected = config["selected_gpu_uuids"]
    if (not selected or len(set(selected)) != len(selected)
            or any(item not in hardware["allowed_gpu_uuids"] for item in selected)):
        raise ValueError("GPU selection must stay within the approved last four RTX 3090 UUIDs")
    confirmed = hardware.get("gpu3_confirmed") is True or os.environ.get("CONFIRM_GPU3_USE") == "1"
    return config, hardware, selected, confirmed


def precheck(root):
    _, hardware, selected, confirmed = configured_gpus(root)
    if hardware["gpu3_uuid"] in selected and not confirmed:
        raise ValueError("GPU3 is selected, but /etc/motd reports instability. Confirm its availability before enabling this recipe")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("This recipe requires WANDB_MODE=online")
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        raise ValueError("Unset WANDB_DISABLED so training metrics are recorded")
    if not wandb_credentials_present():
        raise ValueError("W&B is not configured. In the server terminal run: bash scripts/login_wandb_sft.sh")


def launch_values(root):
    precheck(root)
    _, hardware, selected, _ = configured_gpus(root)
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"
    ], check=True, capture_output=True, text=True)
    inventory = {row[0].strip(): (row[1].strip(), int(row[2].strip()))
                 for row in csv.reader(result.stdout.splitlines())}
    for item in selected:
        if item not in inventory or "3090" not in inventory[item][0]:
            raise ValueError("An approved RTX 3090 is missing; do not replace it with another GPU")
        if inventory[item][1] < hardware["min_free_memory_mib"]:
            index = hardware["physical_index_by_uuid"][item]
            raise ValueError(f"GPU {index} has insufficient free memory; wait for it to become available")
    # The wrapper uses mapfile; no shell text or eval is produced.
    print(len(selected))
    print(",".join(selected))


def status(root):
    config, hardware, selected, confirmed = configured_gpus(root)
    data = read_json(root / "configs/lora_sft_data.json")
    generation = Path(data["generated_run_dir"])
    report_path = generation / "generation_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    accepted, attempts = {}, 0
    path = generation / "attempts.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            attempts += 1
            if row.get("status") == "accepted":
                accepted[row["sample_id"]] = row
    prepared = Path(config["prepared_dir"]) / "manifest.json"
    result = {
        "training_project": str(root), "generated_episodes": len(accepted),
        "generated_by_source": dict(Counter(row["data_source"] for row in accepted.values())),
        "generation_attempts": attempts, "generation_complete": report.get("complete") is True,
        "original_100_file": str(generation / "accepted.jsonl"),
        "prepared_data_exists": prepared.is_file(),
        "prepared_counts": read_json(prepared).get("counts") if prepared.is_file() else None,
        "selected_physical_gpus": [hardware["physical_index_by_uuid"][item] for item in selected],
        "gpu3_confirmed": confirmed, "wandb_credentials_configured": wandb_credentials_present(),
        "wandb_project": config["wandb_project"], "training_output": config["output_dir"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "precheck", "launch-values"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        {"status": status, "precheck": precheck, "launch-values": launch_values}[args.mode](root)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Launch blocked: {error}\n")


if __name__ == "__main__":
    main()
