#!/usr/bin/env python3
"""Run four independent inference workers on the approved RTX 3090 cards."""

import argparse
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    hardware = json.loads((ROOT / "configs/lora_sft_gpus.json").read_text())
    selected = config["selected_gpu_uuids"]
    if (len(selected) != 4 or len(set(selected)) != 4
            or any(item not in hardware["allowed_gpu_uuids"] for item in selected)):
        raise ValueError("This evaluation must use exactly the approved four RTX 3090 UUIDs")
    if hardware["gpu3_uuid"] in selected and hardware.get("gpu3_confirmed") is not True:
        raise ValueError("GPU3 selection lacks the recorded user confirmation")
    base_env = os.environ.copy()
    base_env.update({
        "CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"), "TMPDIR": str(ROOT / ".tmp"),
        "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1",
        "WANDB_MODE": "disabled",
    })
    for key in ("PYTHONPATH", "PYTHONHOME", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        base_env.pop(key, None)
    python = str(ROOT / ".venv/bin/python")

    def command(script):
        return [python, str(ROOT / "scripts" / script), "--config", str(config_path)]

    # Keep the orchestration lock outside an initially empty data directory.
    with (output.parent / f".{output.name}.launcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(command("prepare_grpo_lora_eval.py"), env=base_env, cwd=ROOT, check=True)
        # A single bounded NVML query; no CUDA model is loaded before this check.
        gpu_result = subprocess.run([
            "nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"
        ], text=True, capture_output=True, timeout=45, check=True)
        inventory = {row[0].strip(): (row[1].strip(), int(row[2].strip()))
                     for row in csv.reader(gpu_result.stdout.splitlines())}
        for item in selected:
            if item not in inventory or "3090" not in inventory[item][0]:
                raise ValueError("A selected RTX 3090 is missing")
            if inventory[item][1] < 18000:
                raise ValueError("A selected GPU has less than 18000 MiB free; no inference worker started")
        metadata = {
            "started_at_utc": now(), "status": "running", "launcher_pid": os.getpid(),
            "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selected_gpu_uuids": selected,
            "selected_physical_gpus": [hardware["physical_index_by_uuid"][item] for item in selected],
            "inference_parallelism": "four independent single-GPU replicas; no distributed collectives",
            "workers": [],
        }
        processes, handles = [], []
        try:
            for index, gpu in enumerate(selected):
                env = {**base_env, "CUDA_VISIBLE_DEVICES": gpu}
                log_path = output / f"worker-{index}.log"
                handle = log_path.open("a", buffering=1)
                handles.append(handle)
                argv = command("evaluate_search_lora.py") + [
                    "--worker-index", str(index), "--num-workers", str(len(selected))]
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=handle, stderr=subprocess.STDOUT)
                processes.append(process)
                metadata["workers"].append({"worker_index": index, "gpu_uuid": gpu,
                                            "pid": process.pid, "log": str(log_path)})
            write_json(output / "execution.json", metadata)
            print(json.dumps({"started": now(), "workers": metadata["workers"]}, ensure_ascii=False), flush=True)
            while True:
                result = subprocess.run(command("report_grpo_lora_eval.py"), cwd=ROOT, env=base_env)
                if result.returncode:
                    raise RuntimeError("Result aggregation failed; see the launcher log")
                if all(process.poll() is not None for process in processes):
                    break
                time.sleep(30)
            subprocess.run(command("report_grpo_lora_eval.py"), cwd=ROOT, env=base_env, check=True)
            for worker, process in zip(metadata["workers"], processes):
                worker["exit_code"] = process.returncode
            report = json.loads((output / "report.json").read_text())
            metadata.update({"finished_at_utc": now(),
                             "status": report["status"] if all(p.returncode == 0 for p in processes) else "worker_failed",
                             "accuracy_final": report["accuracy_final"]})
            write_json(output / "execution.json", metadata)
            if any(process.returncode for process in processes) or not report["complete"]:
                raise RuntimeError("At least one evaluation worker failed; completed predictions are retained for resume")
        except BaseException:
            # Only processes launched and owned by this exact runner are signaled.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            metadata.update({"stopped_at_utc": now(), "status": "launcher_failed_or_interrupted"})
            write_json(output / "execution.json", metadata)
            raise
        finally:
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Evaluation launch failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
