#!/usr/bin/env python3
"""Bound preflight/smoke/train/finish orchestration for the four approved GPUs.

Does not install packages, kill unrelated processes or attach to an existing Ray
cluster.  The default all phase runs sequentially and stops at the first failed
check. Each child is recorded and its log is kept under the experiment root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SERVER_ROOT = Path('/srv/search-r1')
OFFICIAL_SEARCH_R1 = '598e61bd1d36895726d28a8d06b3a15bed19f5d3'
GPU_UUIDS = {
    2: 'GPU-00000000-0000-0000-0000-000000000001',
    3: 'GPU-00000000-0000-0000-0000-000000000004',
    4: 'GPU-00000000-0000-0000-0000-000000000003',
    5: 'GPU-00000000-0000-0000-0000-000000000002',
}
CODE_FILES = [
    'scripts/run_search_grpo.py', 'scripts/run_search_grpo.sh',
    'scripts/search_grpo_dataset.py', 'scripts/search_grpo_agent_loop.py',
    'scripts/search_grpo_reward.py', 'scripts/search_grpo_metrics.py',
    'scripts/search_sft_protocol.py', 'scripts/sft_reward_monitor.py',
    'scripts/qa_metrics.py', 'scripts/prepare_data.py',
    'scripts/prepare_search_grpo.py', 'scripts/patch_search_grpo_runtime.py',
    'scripts/finish_search_grpo.py',
    'scripts/patch_search_grpo_vllm.py',
    'configs/search_grpo_v1.yaml', 'configs/search_grpo_agent_loop_v1.yaml',
]
RUNTIME_PACKAGES = ('torch', 'ray', 'vllm', 'transformers', 'peft', 'hydra-core',
                    'omegaconf', 'tensordict', 'datasets', 'pyarrow', 'accelerate',
                    'numpy', 'requests', 'wandb', 'verl')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path, expected=None):
    path = Path(path).resolve()
    stat = path.stat()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f'File SHA mismatch: {path}')
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': actual}


def runtime_versions():
    result = {'python': sys.version.split()[0]}
    for package in RUNTIME_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def runtime_distributions():
    return dict(sorted((distribution.metadata['Name'].lower().replace('_', '-'), distribution.version)
                       for distribution in importlib.metadata.distributions()))


def verify_manifest_files(directory, manifest):
    records = []
    if manifest.get('complete') is not True or not isinstance(manifest.get('files'), dict):
        raise ValueError(f'Incomplete source manifest in {directory}')
    directory = Path(directory).resolve()
    for relative, claimed in sorted(manifest['files'].items()):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Manifest file escapes its source directory')
        record = fingerprint(path, claimed['sha256'])
        if record['bytes'] != claimed['bytes']:
            raise ValueError(f'File size mismatch: {path}')
        records.append(record)
    return records


def build_environment(root, run_id, phase):
    root = Path(root)
    env = os.environ.copy()
    cache = root / '.cache' / 'grpo'
    paths = {
        'HF_HOME': cache / 'huggingface', 'TORCH_HOME': cache / 'torch',
        'TORCH_EXTENSIONS_DIR': cache / 'torch_extensions', 'PIP_CACHE_DIR': cache / 'pip',
        'XDG_CACHE_HOME': cache / 'xdg', 'VLLM_CACHE_ROOT': cache / 'vllm',
        'WANDB_DIR': cache / 'wandb', 'WANDB_DATA_DIR': cache / 'wandb-data',
        'WANDB_CACHE_DIR': cache / 'wandb-cache', 'TMPDIR': cache / 'tmp' / run_id,
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    # Unix-domain sockets have a 107-byte pathname limit. Package/model/temp
    # files remain on /media/6; only vLLM's tiny IPC sockets use this short path.
    rpc_path = Path(vllm_rpc_path(run_id, phase))
    rpc_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if rpc_path.stat().st_uid != os.getuid():
        raise ValueError('vLLM IPC directory belongs to another user')
    env['VLLM_RPC_BASE_PATH'] = str(rpc_path)
    spill_path = Path(ray_spill_path(run_id, phase, root=root))
    spill_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    env.update({
        'SEARCH_GRPO_ROOT': str(root),
        'CUDA_VISIBLE_DEVICES': ','.join(GPU_UUIDS.values()), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'PYTHONPATH': os.pathsep.join([str(root / 'scripts'), str(root / 'vendor/verl-0.6.1')]),
        'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'WANDB_PROJECT': 'search-r1-qwen3-4b-lora-grpo',
        'WANDB_ENTITY': 'your-wandb-entity',
        'WANDB_LOG_MODEL': 'false', 'WANDB_WATCH': 'false', 'WANDB_DISABLE_CODE': 'true',
        'HTTP_PROXY': 'http://127.0.0.1:30189', 'HTTPS_PROXY': 'http://127.0.0.1:30189',
        'http_proxy': 'http://127.0.0.1:30189', 'https_proxy': 'http://127.0.0.1:30189',
        'NO_PROXY': 'localhost,127.0.0.1,::1,192.0.2.10',
        'no_proxy': 'localhost,127.0.0.1,::1,192.0.2.10',
    })
    # Never reuse a shared/foreign cluster selected by inherited environment.
    env.pop('RAY_ADDRESS', None)
    env.pop('RAY_NAMESPACE', None)
    if phase == 'smoke':
        env['WANDB_MODE'] = 'disabled'
        env.pop('WANDB_RUN_ID', None)
        env.pop('WANDB_RESUME', None)
    else:
        env['WANDB_RUN_ID'] = hashlib.sha256(run_id.encode()).hexdigest()[:8]
        env['WANDB_RESUME'] = 'allow'
        env['WANDB_MODE'] = 'online'
    return env


def vllm_rpc_path(run_id, phase):
    suffix = hashlib.sha256((run_id + ':' + phase).encode()).hexdigest()[:10]
    return '/tmp/vgrpo-' + suffix


def ray_spill_path(run_id, phase, root=SERVER_ROOT):
    suffix = hashlib.sha256((run_id + ':' + phase).encode()).hexdigest()[:10]
    return str(Path(root) / '.cache/grpo/ray-spill' / suffix)


def ray_overrides(run_id, phase):
    suffix = hashlib.sha256((run_id + ':' + phase).encode()).hexdigest()[:10]
    return [
        '++ray_kwargs.ray_init.address=local',
        f'++ray_kwargs.ray_init.namespace=sgrpo-{phase}-{suffix}',
        f'++ray_kwargs.ray_init._temp_dir=/tmp/sgrpo-{suffix}',
        '++ray_kwargs.ray_init.include_dashboard=false',
        '++ray_kwargs.ray_init.object_store_memory=4294967296',
        f'++ray_kwargs.ray_init.object_spilling_directory={ray_spill_path(run_id, phase)}',
        'ray_kwargs.ray_init.num_cpus=16',
    ]


def phase_overrides(config, phase, run_id):
    result = ray_overrides(run_id, phase)
    output = Path(config['trainer']['default_local_dir'])
    if phase == 'smoke':
        output = Path(config['trainer']['default_local_dir'] + '_smoke') / run_id
        result += [
            'data.train_max_samples=4', 'data.val_max_samples=4',
            'data.train_batch_size=4', 'data.val_batch_size=4',
            'actor_rollout_ref.actor.ppo_mini_batch_size=4',
            'actor_rollout_ref.rollout.n=5', 'trainer.total_epochs=1',
            'trainer.total_training_steps=1', 'actor_rollout_ref.actor.optim.total_training_steps=1',
            'actor_rollout_ref.actor.optim.lr_warmup_steps=0',
            'trainer.test_freq=1', 'trainer.save_freq=1', 'trainer.val_before_train=false',
            'trainer.logger=[console]', 'trainer.resume_mode=disable',
            f'trainer.experiment_name={config["trainer"]["experiment_name"]}_smoke',
            f'trainer.default_local_dir={output}',
        ]
    # Custom env vars are not assumed to pass implicitly through Ray workers.
    # No credentials are serialized into Hydra/config/logs; authentication uses
    # the existing server credential file, accessible to this user's workers.
    worker_env = {
        'SEARCH_GRPO_OUTPUT_DIR': str(output),
        'WANDB_MODE': 'disabled' if phase == 'smoke' else 'online',
        'VLLM_RPC_BASE_PATH': vllm_rpc_path(run_id, phase),
    }
    if phase != 'smoke':
        worker_env.update(WANDB_RUN_ID=hashlib.sha256(run_id.encode()).hexdigest()[:8],
                          WANDB_RESUME='allow')
    result += [f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(value)}'
               for key, value in worker_env.items()]
    return result


def validate_formal_config(config):
    search, data, actor = config['search_r1'], config['data'], config['actor_rollout_ref']
    trainer = config['trainer']
    if search['physical_gpus'] != [2, 3, 4, 5] or search['official_search_r1_commit'] != OFFICIAL_SEARCH_R1:
        raise ValueError('Unapproved GPU selection or unexpected official source pin')
    if str(search['verl_version']) != '0.6.1':
        raise ValueError('This launcher requires pinned verl v0.6.1')
    if [search['expected_train_count'], search['expected_val_count'], search['expected_test_count']] != [2000, 200, 1000]:
        raise ValueError('Expected dataset cardinalities must remain 2000/200/1000')
    if actor['model']['target_modules'] != ['q_proj', 'v_proj'] or actor['model']['lora_rank'] != 32:
        raise ValueError('This run is authorized for new rank-32 q/v LoRA only')
    if actor['rollout']['n'] != 5 or actor['actor']['ppo_micro_batch_size_per_gpu'] != 1:
        raise ValueError('Expected n=5 trajectories and per-GPU microbatch=1')
    if config['algorithm']['adv_estimator'] != 'grpo' or search['format_reward_weight'] != 0:
        raise ValueError('Expected GRPO with pure answer EM and no format bonus')
    epochs, batch = trainer['total_epochs'], data['train_batch_size']
    if not isinstance(epochs, int) or not 1 <= epochs <= 3 or 2000 % batch or (batch * 5) % 4:
        raise ValueError('Epochs must be 1..3 and batch must divide the selected data and four GPUs')
    expected_steps = 2000 // batch * epochs
    if trainer['total_training_steps'] != expected_steps or actor['actor']['optim']['total_training_steps'] != expected_steps:
        raise ValueError('Training step budget disagrees with full data coverage')
    if actor['actor']['ppo_mini_batch_size'] != batch:
        raise ValueError('Expected one optimizer update per collected prompt batch')
    if trainer['test_freq'] != 50 or trainer['save_freq'] != 50 or not trainer['val_before_train']:
        raise ValueError('Independent dev must run at step zero and every 50 iterations')
    if not search['final_test_only_after_selection']:
        raise ValueError('The 1000-question test set is reserved for final evaluation')


def compose_config(root, overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(Path(root) / 'configs')):
        config = compose(config_name='search_grpo_v1', overrides=overrides)
        OmegaConf.resolve(config)
        return OmegaConf.to_container(config, resolve=True), OmegaConf.to_yaml(config, resolve=True)


def command(root, overrides):
    return [str(Path(root) / '.venv-grpo/bin/python'), '-m', 'verl.trainer.main_ppo',
            '--config-path=' + str(Path(root) / 'configs'), '--config-name=search_grpo_v1', *overrides]


def target_gpu_snapshot():
    selected = ','.join(GPU_UUIDS.values())
    output = subprocess.check_output([
        'nvidia-smi', '-i', selected, '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in output.splitlines():
        index, uuid, free = [part.strip() for part in line.split(',')]
        if GPU_UUIDS.get(int(index)) != uuid:
            raise RuntimeError('Approved physical GPU index/UUID mapping changed')
        rows.append({'physical_index': int(index), 'uuid': uuid, 'free_mib': int(free)})
    if {row['physical_index'] for row in rows} != set(GPU_UUIDS):
        raise RuntimeError('Did not find all four approved GPUs')
    processes = subprocess.check_output([
        'nvidia-smi', '-i', selected, '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    occupied = []
    for line in processes.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 3 and parts[0] in GPU_UUIDS.values():
            occupied.append({'uuid': parts[0], 'pid': parts[1], 'used_mib': parts[2]})
    return {'gpus': rows, 'compute_processes': occupied,
            'ready': not occupied and all(row['free_mib'] >= 20 * 1024 for row in rows)}


def wait_for_gpus(wait_seconds, status_path):
    deadline = time.monotonic() + wait_seconds
    while True:
        result = target_gpu_snapshot()
        atomic_json(status_path, {'checked_at': now(), **result})
        if result['ready']:
            return result
        print('Approved GPUs are busy; no process will be stopped. ' + canonical(result), flush=True)
        if time.monotonic() >= deadline:
            raise RuntimeError('Approved GPUs are not idle with >=20 GiB free; rerun later or increase --wait-gpu-seconds')
        time.sleep(min(10, max(0.01, deadline - time.monotonic())))


def preflight(root, config, log_dir, run_id, overrides):
    validate_formal_config(config)
    root, log_dir = Path(root), Path(log_dir)
    python = root / '.venv-grpo/bin/python'
    if not python.exists():
        raise FileNotFoundError('Isolated .venv-grpo runtime is not ready')
    versions = runtime_versions()
    expected_versions = {'torch': '2.8.0', 'vllm': '0.11.0', 'verl': '0.6.1',
                         'transformers': '4.56.2', 'peft': '0.17.1', 'ray': '2.48.0',
                         'numpy': '1.26.4', 'wandb': '0.26.1', 'tensordict': '0.10.0'}
    if any(versions.get(key) != value for key, value in expected_versions.items()):
        raise ValueError('Installed GRPO runtime differs from the pinned compatible versions')
    if any(versions.get(key) is None for key in RUNTIME_PACKAGES):
        raise ValueError('Required GRPO runtime dependency is missing')
    data_dir = root / 'prepared/grpo2000_sft324_v1'
    model_dir = Path(config['actor_rollout_ref']['model']['path'])
    data_file, model_file = data_dir / 'data_manifest.json', model_dir / 'sft_export_manifest.json'
    data, model = json.loads(data_file.read_text()), json.loads(model_file.read_text())
    if [data['counts'][key] for key in ('train', 'val', 'test')] != [2000, 200, 1000]:
        raise ValueError('Prepared manifest does not contain the approved split sizes')
    records = verify_manifest_files(data_dir, data) + verify_manifest_files(model_dir, model)
    shared = model['shared_embedding']
    if shared.get('tied') is not True or shared.get('dtype') != 'bfloat16':
        raise ValueError('Merged SFT export must contain tied BF16 learned embeddings')
    audit_file = data_dir / 'tokenization_audit.json'
    audit = json.loads(audit_file.read_text())
    if not audit.get('all_checks_passed') or audit.get('total_checked_rows') != 3200:
        raise ValueError('Complete CPU tokenization audit is required')
    if any(audit['cross_split_id_overlap'].values()):
        raise ValueError('CPU audit found split ID overlap')
    if audit['tokenizer_json_sha256'] != model['files']['tokenizer.json']['sha256']:
        raise ValueError('Audited tokenizer differs from merged SFT tokenizer')
    for split, expected in [('train', 2000), ('val', 200), ('test', 1000)]:
        part = audit['splits'][split]
        if (part['count'] != expected or part['selected_id_coverage'] != expected
                or part['prompt_tokens_max'] > config['data']['max_prompt_length']
                or part['parquet_sha256'] != data['files'][split + '.parquet']['sha256']):
            raise ValueError(f'CPU tokenization audit mismatch: {split}')
    vendor = root / 'vendor/verl-0.6.1'
    git = lambda *args: subprocess.check_output(['git', '-C', str(vendor), *args], text=True).strip()
    head, tag = git('rev-parse', 'HEAD'), git('rev-parse', 'v0.6.1^{commit}')
    if head != tag:
        raise ValueError('Vendor HEAD is not the pinned v0.6.1 tag')
    tracked = git('ls-files', '--', '*.py', '*.yaml', '*.toml').splitlines()
    records += [fingerprint(vendor / relative) for relative in tracked]
    patch_manifests = sorted((vendor / '.search-grpo-patches').glob('**/manifest.json'))
    if not patch_manifests:
        raise ValueError('Version-controlled Search-R1 runtime patch manifest is required')
    records += [fingerprint(path) for path in patch_manifests]
    records += [fingerprint(root / relative) for relative in CODE_FILES]
    records += [fingerprint(data_file), fingerprint(model_file), fingerprint(audit_file)]
    records.append(fingerprint(root / 'grpo/runtime.freeze.txt'))
    records.append(fingerprint(importlib.metadata.distribution('vllm').locate_file('vllm/platforms/interface.py')))
    records.append(fingerprint(root / 'grpo/vllm-patches/vllm011-uuid-v1/manifest.json'))
    probe_file = root / 'grpo/runtime_gpu_probe.json'
    probe = json.loads(probe_file.read_text())
    if (probe.get('complete') is not True or probe.get('physical_gpu') != 2
            or probe.get('vllm_rank32_adapter_generation') is not True
            or probe.get('BF16_matmul_backward_finite') is not True):
        raise ValueError('Real CUDA/vLLM compatibility probe has not passed')
    records.append(fingerprint(probe_file))
    # Source datasets are small enough to verify again; exported model weights
    # were checked fully above rather than merely accepting recorded metadata.
    for path, expected in data['request']['source_hashes'].items():
        records.append(fingerprint(path, expected))
    binding = {'schema_version': 1, 'run_id': run_id, 'root': str(root),
               'formal_config': config, 'formal_overrides': overrides,
               'physical_gpu_uuids': GPU_UUIDS, 'official_search_r1_commit': OFFICIAL_SEARCH_R1,
               'verl_commit': head, 'runtime_versions': versions,
               'runtime_distributions': runtime_distributions(), 'files': records}
    binding['binding_sha256'] = hashlib.sha256(canonical(binding).encode()).hexdigest()
    path = log_dir / 'run_manifest.json'
    if path.exists() and canonical(json.loads(path.read_text())) != canonical(binding):
        raise ValueError('Run binding changed; use a new --run-id after modifying config/code/runtime')
    atomic_json(path, binding)
    atomic_json(log_dir / 'preflight.json', {'complete': True, 'finished_at': now(),
                                          'binding_sha256': binding['binding_sha256'],
                                          'verified_files': len(records)})
    return binding


def verify_binding(path, config, overrides):
    binding = json.loads(Path(path).read_text())
    claimed = binding.pop('binding_sha256')
    if hashlib.sha256(canonical(binding).encode()).hexdigest() != claimed:
        raise ValueError('Run manifest digest is invalid')
    binding['binding_sha256'] = claimed
    if binding['formal_config'] != config or binding['formal_overrides'] != overrides:
        raise ValueError('Formal configuration differs from frozen preflight')
    if 'runtime_versions' in binding and binding['runtime_versions'] != runtime_versions():
        raise ValueError('Installed package versions changed after frozen preflight')
    if 'runtime_distributions' in binding and binding['runtime_distributions'] != runtime_distributions():
        raise ValueError('Installed transitive dependencies changed after frozen preflight')
    for entry in binding['files']:
        path = Path(entry['path']); stat = path.stat()
        if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError(f'Bound input changed since preflight: {path}')
        if stat.st_size <= 64 * 1024 * 1024 and sha256(path) != entry['sha256']:
            raise ValueError(f'Bound code/input SHA changed: {path}')
    return binding


def run_child(argv, env, cwd, phase, directory):
    directory = Path(directory)
    status_file = directory / (phase + '.status.json')
    log = directory / (phase + '.log')
    status = {'phase': phase, 'started_at': now(), 'state': 'running', 'command': argv,
              'log': str(log), 'parent_pid': os.getpid()}
    previous = signal.getsignal(signal.SIGTERM)
    with log.open('a', buffering=1) as stream:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        status['pid'] = child.pid; atomic_json(status_file, status)
        print(f'{phase}: PID {child.pid}; log {log}', flush=True)

        def terminate_owned(signum, frame):
            # This group was created immediately above, never a foreign job.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate_owned)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            terminate_owned(None, None)
            code = child.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
    status.update(exit_code=code, state='complete' if code == 0 else 'failed', finished_at=now())
    atomic_json(status_file, status)
    if code:
        raise RuntimeError(f'{phase} exited {code}; inspect {log}')
    return status


def accept_smoke(output, destination, binding_sha256):
    output = Path(output)
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if len(metrics) != 1 or metrics[0]['global_step'] != 1:
        raise ValueError('Smoke must complete exactly one real optimization iteration')
    numeric = metrics[0]['metrics']
    for key in ('actor/pg_loss', 'actor/grad_norm', 'train/total_reward', 'train/format_reward'):
        if key not in numeric or not isinstance(numeric[key], (float, int)) or not math.isfinite(numeric[key]):
            raise ValueError(f'Smoke is missing finite post-update metric {key}')
    audit = metrics[0]['audit']
    if sha256(audit['path']) != audit['sha256']:
        raise ValueError('Smoke trajectory audit changed')
    rows = [json.loads(line) for line in Path(audit['path']).read_text().splitlines() if line.strip()]
    groups = Counter(row['group_uid'] for row in rows)
    if len(rows) != 20 or len(groups) != 4 or set(groups.values()) != {5}:
        raise ValueError('Smoke requires four questions with five intact trajectories each')
    if len({row['sample_id'] for row in rows}) != 4:
        raise ValueError('Smoke repeats/misses selected question IDs')
    checkpoint = output / 'global_step_1/actor'
    if not checkpoint.is_dir() or not any(checkpoint.rglob('*')):
        raise ValueError('Smoke did not write a real actor checkpoint')
    parameter_file = output / 'runtime_parameter_audit.json'
    parameters = json.loads(parameter_file.read_text())
    for key in ('embedding_frozen', 'only_qv_lora_trainable', 'shared_embedding_tied'):
        if parameters.get(key) is not True:
            raise ValueError(f'Actual actor runtime parameter audit failed: {key}')
    export_file = output / 'exported_adapter/grpo_adapter_export.json'
    exported = json.loads(export_file.read_text())
    if (exported.get('complete') is not True
            or exported.get('source_binding', {}).get('global_step') != 1
            or exported.get('source_binding', {}).get('source_actor') != str(checkpoint)
            or exported.get('frozen_backbone_weights_checked', 0) <= 0
            or exported.get('lora_B_nonzero_count', 0) <= 0):
        raise ValueError('Smoke must export the real updated adapter and verify every frozen backbone weight')
    for name, checksum in exported['file_sha256'].items():
        if sha256(output / 'exported_adapter' / name) != checksum:
            raise ValueError('Smoke exported adapter fingerprint changed')
    result = {'complete': True, 'accepted_at': now(), 'binding_sha256': binding_sha256,
              'smoke_output': str(output), 'real_optimizer_iterations': 1,
              'question_count': 4, 'trajectory_count': 20, 'group_size': 5,
              'response_masks_independently_audited': True, 'parameter_audit': parameters,
              'parameter_audit_sha256': sha256(parameter_file), 'training_audit': audit,
              'checkpoint': str(checkpoint), 'metrics': numeric,
              'checkpoint_export_proof': str(export_file),
              'checkpoint_export_proof_sha256': sha256(export_file)}
    atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight', 'smoke', 'train', 'finish', 'all'], default='all')
    parser.add_argument('--run-id', default='grpo-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--wait-gpu-seconds', type=int, default=0)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,60}', args.run_id):
        parser.error('--run-id must be a short filesystem-safe identifier')
    if args.wait_gpu_seconds < 0:
        parser.error('--wait-gpu-seconds must be nonnegative')
    root = SERVER_ROOT
    log_dir = root / 'logs/search_grpo' / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_file = log_dir / 'launcher.lock'
    with lock_file.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['SEARCH_GRPO_ROOT'] = str(root)
        config, text = compose_config(root, args.override)
        validate_formal_config(config)
        resolved = log_dir / 'resolved_formal_config.yaml'
        if resolved.exists() and resolved.read_text() != text:
            raise ValueError('Resolved configuration changed under existing run ID')
        resolved.write_text(text)
        stages = ['preflight', 'smoke', 'train', 'finish'] if args.phase == 'all' else [args.phase]
        for phase in stages:
            print(f'{now()} phase={phase} run_id={args.run_id}', flush=True)
            if phase == 'preflight':
                stage_status = {'phase': phase, 'started_at': now(), 'pid': os.getpid(), 'state': 'running'}
                atomic_json(log_dir / 'preflight.status.json', stage_status)
                try:
                    preflight(root, config, log_dir, args.run_id, args.override)
                except Exception as error:
                    stage_status.update(state='failed', finished_at=now(), exit_code=1,
                                        error_type=type(error).__name__, error=str(error))
                    atomic_json(log_dir / 'preflight.status.json', stage_status)
                    raise
                stage_status.update(state='complete', finished_at=now(), exit_code=0)
                atomic_json(log_dir / 'preflight.status.json', stage_status)
                continue
            binding = verify_binding(log_dir / 'run_manifest.json', config, args.override)
            if phase == 'finish':
                script = root / 'scripts/finish_search_grpo.py'
                if not script.is_file():
                    raise FileNotFoundError('Training may be complete, but finish_search_grpo.py is not ready; no final result claimed')
                train_status = json.loads((log_dir / 'train.status.json').read_text())
                if train_status.get('state') != 'complete' or train_status.get('exit_code') != 0:
                    raise ValueError('Final selection/evaluation requires a completed formal training process')
                wait_for_gpus(args.wait_gpu_seconds, log_dir / 'finish.gpu_readiness.json')
                run_child([str(root / '.venv-grpo/bin/python'), str(script), '--config', str(resolved)],
                          build_environment(root, args.run_id, phase), root, phase, log_dir)
                continue
            stage_extra = phase_overrides(config, phase, args.run_id)
            if phase == 'train':
                accepted = json.loads((log_dir / 'smoke_acceptance.json').read_text())
                if not accepted.get('complete') or accepted.get('binding_sha256') != binding['binding_sha256']:
                    raise ValueError('Formal training requires accepted smoke for this exact frozen binding')
            phase_config, phase_text = compose_config(root, args.override + stage_extra)
            (log_dir / f'resolved_{phase}_config.yaml').write_text(phase_text)
            output = Path(phase_config['trainer']['default_local_dir'])
            output.mkdir(parents=True, exist_ok=True)
            if phase == 'train':
                existing = output / 'launcher_run_manifest.json'
                if existing.exists() and json.loads(existing.read_text())['binding_sha256'] != binding['binding_sha256']:
                    raise ValueError('Existing formal output belongs to another frozen run')
                atomic_json(existing, binding)
                (output / 'resolved_config.yaml').write_text(phase_text)
            wait_for_gpus(args.wait_gpu_seconds, log_dir / f'{phase}.gpu_readiness.json')
            env = build_environment(root, args.run_id, phase)
            env['SEARCH_GRPO_OUTPUT_DIR'] = str(output)
            if phase == 'train':
                atomic_json(output / 'runtime_meta.json', {
                    'created_at': now(), 'launcher_run_id': args.run_id,
                    'wandb_run_id': env['WANDB_RUN_ID'],
                    'wandb_entity': env['WANDB_ENTITY'], 'wandb_project': env['WANDB_PROJECT'],
                    'entity': env['WANDB_ENTITY'], 'project': env['WANDB_PROJECT'],
                    'binding_sha256': binding['binding_sha256'],
                })
            run_child(command(root, args.override + stage_extra), env, root, phase, log_dir)
            if phase == 'smoke':
                # Reconstruct the actual four-rank checkpoint on CPU before
                # committing the GPUs to a formal run. This checks both the
                # native checkpoint format and the frozen base tensor values.
                export_env = build_environment(root, args.run_id, phase)
                export_env['CUDA_VISIBLE_DEVICES'] = ''
                run_child([str(root / '.venv-grpo/bin/python'),
                           str(root / 'scripts/finish_search_grpo.py'),
                           '--config', str(resolved),
                           '--export-checkpoint-only', str(output / 'global_step_1/actor'),
                           '--export-step', '1', '--export-dir', str(output / 'exported_adapter')],
                          export_env, root, 'smoke_export', log_dir)
                accept_smoke(output, log_dir / 'smoke_acceptance.json', binding['binding_sha256'])
    print('Requested phases completed; logs: ' + str(log_dir), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
