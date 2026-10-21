#!/usr/bin/env python3
"""Freeze dev-only GRPO selection, export its adapter, then run paired final tests.

Run with the GRPO environment (matching native torch DTensor checkpoints).
The existing inference launcher deliberately uses the original SFT environment.
--prepare-only writes evaluation plans; it never reads test answers or starts
inference, and does not require training to have completed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from prepare_search_grpo import read_rows, sha256, write_json

GPU_UUIDS = ['GPU-00000000-0000-0000-0000-000000000001', 'GPU-00000000-0000-0000-0000-000000000004',
             'GPU-00000000-0000-0000-0000-000000000003', 'GPU-00000000-0000-0000-0000-000000000002']
LORA_KEY = re.compile(r'^base_model\.model\.(model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj))\.lora_([AB])\.(?:default\.)?weight$')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    effective = Path(value['trainer']['default_local_dir']) / 'resolved_config.yaml'
    if effective.is_file():
        value = OmegaConf.to_container(OmegaConf.load(effective), resolve=True)
    return value


def paths(config):
    root = Path(config['search_r1']['root']).resolve()
    output = Path(config['trainer']['default_local_dir']).resolve()
    data = Path(config['data']['train_files']).resolve().parent
    return {'root': root, 'output': output, 'data': data, 'final': output / 'finalization',
            'model': Path(config['actor_rollout_ref']['model']['path']).resolve(),
            'sft_prepared': root / 'prepared/aether_plus_deepseek100_base_4k_v2',
            'sft_embedding_source': root / 'outputs/qwen3_4b_base_aether_new100_lora_r32_qv_full_embedding_v2/final_adapter'}


def make_eval_configs(config):
    p = paths(config)
    data_manifest = json.loads((p['data'] / 'data_manifest.json').read_text())
    if data_manifest.get('complete') is not True:
        raise ValueError('GRPO data preparation is incomplete')
    settings = config['search_r1']
    common = {'model_path': str(p['model']), 'data_file': str(p['data'] / 'test.selected.jsonl'),
              'expected_data_sha256': data_manifest['files']['test.selected.jsonl']['sha256'],
              'expected_count': 1000, 'expected_source_counts': {'nq': 500, 'hotpotqa': 500},
              'training_prepared_dir': str(p['sft_prepared']),
              'reward_monitor_file': str(p['sft_prepared'] / 'reward_monitor.jsonl'),
              'expected_reward_monitor_count': 32, 'selected_gpu_uuids': GPU_UUIDS, 'num_workers': 4,
              'seed': settings.get('seed', 1), 'batch_size': 32, 'generation_microbatch_size': 8,
              'max_new_tokens': settings['max_new_tokens_per_action'],
              'max_total_generated_tokens': settings['max_generated_tokens_per_question'],
              'max_context_tokens': settings['max_context_tokens'], 'max_searches': settings['max_searches'],
              'max_turns': settings['max_turns'], 'max_invalid_actions': settings['max_invalid_actions'],
              'topk': settings['top_k'], 'passage_chars': settings['max_chars_per_passage'],
              'retriever_url': settings['retrieval_url'], 'retrieval_timeout_seconds': settings['retrieval_timeout_seconds'],
              'retrieval_retries': settings['retrieval_retries'], 'require_no_training_overlap': True}
    results = {}
    for name, adapter in (('merged_sft_initial', p['output'] / 'initial_zero_adapter'),
                           ('best_grpo', p['output'] / 'final_adapter')):
        value = {**common, 'evaluation_name': f'Final held-out same1000 paired evaluation: {name}',
                 'adapter_path': str(adapter), 'output_dir': str(p['output'] / 'final_evaluation' / name)}
        destination = p['final'] / f'{name}.eval.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and json.loads(destination.read_text()) != value:
            raise ValueError(f'Existing final evaluation configuration differs: {destination}')
        if not destination.exists():
            atomic_json(destination, value)
        results[name] = destination
    return results


def choose_best(entries):
    if not entries or not any(entry['global_step'] == 0 for entry in entries):
        raise ValueError('Dev history must include the zero-GRPO initial model')
    for entry in entries:
        value = entry['val_answer_em']
        if type(entry['global_step']) is not int or entry['global_step'] < 0 or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid independent dev selection metric')
    # Repeated identical resume measurements are harmless; contradictory ones
    # must be inspected instead of selecting a lucky repeat of the same weights.
    scores = {}
    for entry in entries:
        if entry['global_step'] in scores and scores[entry['global_step']] != entry['val_answer_em']:
            raise ValueError('Contradictory validation measurements at the same checkpoint')
        scores[entry['global_step']] = entry['val_answer_em']
    return min(entries, key=lambda entry: (-entry['val_answer_em'], entry['global_step']))


def freeze_selection(config):
    import pyarrow.parquet as pq
    from search_grpo_reward import gold_aliases, score_record
    p = paths(config)
    data_manifest = json.loads((p['data'] / 'data_manifest.json').read_text())
    for name, entry in data_manifest['files'].items():
        if sha256(p['data'] / name) != entry['sha256']:
            raise ValueError('Prepared GRPO split artifact changed after preparation')
    expected_steps = int(config['trainer']['total_training_steps'])
    train_history = read_rows(p['output'] / 'metrics.jsonl')
    if not train_history or max(row['global_step'] for row in train_history) != expected_steps:
        raise ValueError('Training has not completed all configured updates; final test is forbidden')
    val_rows = pq.read_table(Path(config['data']['val_files'])).to_pylist()
    trusted = {row['extra_info']['sample_id']: row for row in val_rows}
    if len(trusted) != config['search_r1'].get('expected_val_count', 200):
        raise ValueError('Independent dev IDs/count differ from the approved split')
    history_path = p['output'] / 'validation_history.jsonl'
    entries = []
    for record in read_rows(history_path):
        audit = record['audit']
        audit_path = Path(audit['path']).resolve()
        if not audit_path.is_relative_to(p['output'] / 'audit') or sha256(audit_path) != audit['sha256']:
            raise ValueError('Validation audit fingerprint or source path changed')
        rows = read_rows(audit_path)
        if len(rows) != len(trusted) or len({row['sample_id'] for row in rows}) != len(trusted) or set(row['sample_id'] for row in rows) != set(trusted):
            raise ValueError('Independent dev audit has missing, duplicate or foreign questions')
        total = 0
        for row in rows:
            source = trusted[row['sample_id']]
            if row['question'] != source['question'] or row['data_source'] != source['data_source'] or row['global_step'] != record['global_step']:
                raise ValueError('Validation question/source/step binding changed')
            score = score_record(row, gold_aliases(source['reward_model']['ground_truth']), source['data_source'])
            if score['score'] != row['score']:
                raise ValueError('Validation EM does not independently recompute')
            total += score['score']
        em = total / len(rows)
        if abs(em - record['metrics']['val/answer_em']) > 1e-9:
            raise ValueError('Validation aggregate EM disagrees with raw assistant actions')
        step = record['global_step']
        actor = p['model'] if step == 0 else p['output'] / f'global_step_{step}' / 'actor'
        entries.append({'global_step': step, 'val_answer_em': em, 'val_correct': int(total), 'val_count': len(rows),
                        'actor_checkpoint': str(actor), 'checkpoint_kind': 'merged_sft_base_zero_grpo_increment' if step == 0 else 'verl_fsdp_actor',
                        'validation_audit': audit})
    required = set(range(0, expected_steps + 1, int(config['trainer']['test_freq']))) | {expected_steps}
    if not required.issubset({entry['global_step'] for entry in entries}):
        raise ValueError('Missing configured periodic/final independent validation')
    best = choose_best(entries)
    declared = json.loads((p['output'] / 'best_checkpoint.json').read_text())
    if declared['global_step'] != best['global_step'] or abs(declared['val_answer_em'] - best['val_answer_em']) > 1e-9:
        raise ValueError('Stored best checkpoint disagrees with dev-only recomputation')
    last = next(entry for entry in reversed(entries) if entry['global_step'] == expected_steps)
    for entry in (best, last):
        if entry['global_step'] > 0:
            validate_native_checkpoint(Path(entry['actor_checkpoint']), expected_world_size=4)
    selection = {'complete': True, 'frozen_at_utc': now(), 'selection_metric': 'independent_dev_answer_EM',
                 'tie_break': 'earlier_global_step', 'test_used_for_selection': False,
                 'validation_history_sha256': sha256(history_path), 'best': best, 'last': last,
                 'all_validation_steps': entries, 'expected_total_updates': expected_steps}
    destination = p['final'] / 'checkpoint_selection.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        for key in ('best', 'last', 'validation_history_sha256'):
            if previous[key] != selection[key]:
                raise ValueError('Checkpoint selection was already frozen and subsequently changed')
        return previous
    atomic_json(destination, selection)
    return selection


def validate_native_checkpoint(actor, expected_world_size=4):
    metadata = json.loads((actor / 'fsdp_config.json').read_text())
    if metadata['world_size'] != expected_world_size:
        raise ValueError('Actor checkpoint world size differs from the four approved GPUs')
    files = [actor / f'model_world_size_{expected_world_size}_rank_{rank}.pt' for rank in range(expected_world_size)]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError('Native actor checkpoint is missing model shards')
    return files


def hf_weight_index(model):
    index = model / 'model.safetensors.index.json'
    if index.exists():
        return json.loads(index.read_text())['weight_map']
    from safetensors import safe_open
    with safe_open(str(model / 'model.safetensors'), framework='pt', device='cpu') as stream:
        return {key: 'model.safetensors' for key in stream.keys()}


def expected_adapter_shapes(model, rank):
    from safetensors import safe_open
    result = {}
    for name, shard in hf_weight_index(model).items():
        if name.endswith(('.q_proj.weight', '.v_proj.weight')):
            with safe_open(str(model / shard), framework='pt', device='cpu') as stream:
                shape = stream.get_slice(name).get_shape()
            prefix = 'base_model.model.' + name.removesuffix('.weight')
            result[prefix + '.lora_A.weight'] = (rank, shape[1])
            result[prefix + '.lora_B.weight'] = (shape[0], rank)
    if not result:
        raise ValueError('Merged SFT base has no q/v projections')
    return result


def canonical_lora_key(name):
    match = LORA_KEY.fullmatch(name)
    if match is None:
        raise ValueError(f'Unexpected or foreign LoRA key: {name}')
    return f'base_model.model.{match.group(1)}.lora_{match.group(2)}.weight'


def merge_values(values, expected_shape):
    """Merge native pure-FSDP DTensors without the vendor's BF16 coercion."""
    import torch
    first = values[0]
    if hasattr(first, '_local_tensor') and hasattr(first, 'placements'):
        if len(first.placements) != 1:
            raise ValueError('Only the configured one-dimensional pure FSDP mesh is supported')
        placement = first.placements[0]
        if any(tuple(value.shape) != tuple(expected_shape) or value.placements != first.placements for value in values):
            raise ValueError('DTensor global shape/placement differs across ranks')
        local = [value._local_tensor.detach().cpu() for value in values]
        if placement.is_replicate():
            if any(not torch.equal(local[0], value) for value in local[1:]):
                raise ValueError('Replicated DTensor weights disagree')
            merged = local[0]
        elif placement.is_shard():
            merged = torch.cat(local, dim=placement.dim)
        else:
            raise ValueError('Partial/unknown DTensor placement cannot be exported')
    elif all(type(value) is torch.Tensor for value in values):
        # World-size one or replicated ordinary tensors are unambiguous.
        if all(tuple(value.shape) == tuple(expected_shape) and torch.equal(value, first) for value in values):
            merged = first.detach().cpu()
        else:
            raise ValueError('Unlabelled tensor shards are ambiguous; refusing a guessed concatenation')
    else:
        raise ValueError('Unsupported native checkpoint tensor type (legacy ShardedTensor requires a dedicated restore)')
    if tuple(merged.shape) != tuple(expected_shape) or not torch.isfinite(merged).all():
        raise ValueError('Merged checkpoint weight shape/finite-value check failed')
    return merged.contiguous()


def adapter_state_from_shards(actor, model, rank):
    import torch
    from safetensors import safe_open
    files = validate_native_checkpoint(actor)
    # These are locally generated native checkpoints from our completed run,
    # not externally downloaded pickle files. DTensor objects require full load.
    states = [torch.load(path, map_location='cpu', weights_only=False) for path in files]
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError('Native model shards have inconsistent parameter names')
    expected = expected_adapter_shapes(model, rank)
    base_index = hf_weight_index(model)
    result, frozen_checked = {}, set()
    for key in sorted(keys):
        values = [state.pop(key) for state in states]
        if 'lora_' in key:
            normalized = canonical_lora_key(key)
            if normalized not in expected or normalized in result:
                raise ValueError(f'Unknown or duplicate GRPO adapter parameter: {normalized}')
            result[normalized] = merge_values(values, expected[normalized])
        else:
            normalized = key.removeprefix('base_model.model.').replace('.base_layer.weight', '.weight')
            mapped = 'model.embed_tokens.weight' if normalized == 'lm_head.weight' and normalized not in base_index else normalized
            if mapped not in base_index:
                raise ValueError(f'Unknown frozen backbone state: {key}')
            with safe_open(str(model / base_index[mapped]), framework='pt', device='cpu') as stream:
                expected_weight = stream.get_tensor(mapped)
            actual = merge_values(values, expected_weight.shape)
            if not torch.equal(actual, expected_weight.to(actual.dtype)):
                raise ValueError(f'GRPO changed frozen SFT backbone weight: {normalized}')
            frozen_checked.add(mapped)
    if set(result) != set(expected) or not set(base_index).issubset(frozen_checked):
        raise ValueError('Checkpoint did not cover every q/v adapter and frozen backbone parameter')
    return result, {'frozen_backbone_weights_checked': len(frozen_checked), 'source_shards': {str(path): sha256(path) for path in files}}


def adapter_statistics(state):
    """CPU value-level proof that a saved new adapter contains a real update."""
    import torch
    total = sum(value.numel() for value in state.values())
    nonfinite = sum(int((~torch.isfinite(value)).sum().item()) for value in state.values())
    b_items = [value for key, value in state.items() if key.endswith('.lora_B.weight')]
    return {'adapter_numel': total,
            'lora_B_tensor_count': len(b_items),
            'lora_B_nonzero_count': sum(int(torch.count_nonzero(value).item()) for value in b_items),
            'adapter_nonfinite_value_count': nonfinite,
            'adapter_finite_value_count': total - nonfinite,
            'all_adapter_tensors_finite': nonfinite == 0}


def export_adapter(config, step, destination, source_actor=None):
    import torch
    from safetensors.torch import load_file, save_file
    p = paths(config)
    model_settings = config['actor_rollout_ref']['model']
    rank, alpha = model_settings['lora_rank'], model_settings['lora_alpha']
    if rank != 32 or alpha != 16 or set(model_settings['target_modules']) != {'q_proj', 'v_proj'}:
        raise ValueError('Adapter export differs from the approved rank32/alpha16 q/v recipe')
    proof_path = destination / 'grpo_adapter_export.json'
    merged_manifest = json.loads((p['model'] / 'sft_export_manifest.json').read_text())
    embedding_file = p['sft_embedding_source'] / 'full_tied_embedding.safetensors'
    embedding_sha = sha256(embedding_file)
    if merged_manifest['request']['source_hashes'].get(str(embedding_file.resolve())) != embedding_sha:
        raise ValueError('Embedding companion is not the source master of the fixed merged SFT base')
    source_binding = {'global_step': step, 'source_embedding_master_sha256': embedding_sha, 'merged_SFT_base': str(p['model']),
                      'merged_export_manifest_sha256': sha256(p['model'] / 'sft_export_manifest.json'),
                      'source_actor': str(source_actor) if source_actor else None, 'r': rank, 'lora_alpha': alpha}
    if destination.exists():
        proof = json.loads(proof_path.read_text())
        if proof.get('complete') is not True or proof['source_binding'] != source_binding:
            raise ValueError('Existing adapter belongs to different model/checkpoint')
        for name, checksum in proof['file_sha256'].items():
            if sha256(destination / name) != checksum:
                raise ValueError('Exported GRPO adapter was modified')
        for name, checksum in proof.get('source_shards', {}).items():
            if sha256(name) != checksum:
                raise ValueError('Selected native actor checkpoint changed after export')
        return proof
    if step == 0:
        state = {key: torch.zeros(shape, dtype=torch.bfloat16) for key, shape in expected_adapter_shapes(p['model'], rank).items()}
        extra = {'purpose': 'zero_increment_evaluation_only; never used to initialize training'}
    else:
        if source_actor is None:
            raise ValueError('A trained GRPO step requires its selected native actor checkpoint')
        state, extra = adapter_state_from_shards(source_actor, p['model'], rank)
    statistics = adapter_statistics(state)
    if not statistics['all_adapter_tensors_finite']:
        raise ValueError(f'Cannot export non-finite GRPO adapter parameters: {statistics}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.', dir=destination.parent))
    try:
        # Write the shared PEFT schema explicitly. The inference environment has
        # PEFT0.15, so avoid newer-version-only config fields from the RL runtime.
        write_json(temporary / 'adapter_config.json', {
            'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'base_model_name_or_path': str(p['model']),
            'r': rank, 'lora_alpha': alpha, 'target_modules': ['q_proj', 'v_proj'],
            'lora_dropout': 0.0, 'bias': 'none', 'inference_mode': True,
            'fan_in_fan_out': False, 'modules_to_save': None, 'use_rslora': False, 'use_dora': False})
        save_file(state, str(temporary / 'adapter_model.safetensors'), metadata={'format': 'pt'})
        restored = load_file(str(temporary / 'adapter_model.safetensors'))
        if set(restored) != set(state) or any(not torch.equal(state[key], restored[key]) for key in state):
            raise ValueError('GRPO adapter tensor serialization changed values')
        # Only the *embedding* companion is copied from SFT. Never copy or load
        # its old LoRA adapter: the merged base already contains that increment.
        for filename in ('full_tied_embedding.safetensors', 'full_tied_embedding.json'):
            shutil.copy2(p['sft_embedding_source'] / filename, temporary / filename)
        proof = {'complete': True, 'created_at_utc': now(), 'source_binding': source_binding,
                 'adapter_tensor_count': len(state), 'adapter_dtypes': sorted({str(value.dtype) for value in state.values()}),
                 **statistics,
                 'embedding_frozen': True, 'old_SFT_LoRA_copied': False, **extra,
                 'file_sha256': {file.name: sha256(file) for file in temporary.iterdir() if file.is_file()}}
        atomic_json(temporary / 'grpo_adapter_export.json', proof)
        os.rename(temporary, destination)
        return proof
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def make_comparison(initial, trained, selection):
    if not initial.get('complete') or not trained.get('complete') or initial['overall']['selected'] != 1000 or trained['overall']['selected'] != 1000:
        raise ValueError('Both held-out paired evaluations must be complete 1000-question reports')
    if initial['data_sha256'] != trained['data_sha256'] or initial['selected_sha256'] != trained['selected_sha256'] or initial['base_model_path'] != trained['base_model_path']:
        raise ValueError('Paired reports do not use identical questions/order and merged base')
    def metrics(report):
        overall = report['overall']
        excluded = report['excluding_training_monitor']['overall']
        return {'answer_em': overall['total_reward'], 'format_reward': overall['format_reward'],
                'correct': overall['correct'], 'count': overall['count'],
                'excluding_SFT_monitor_em': excluded['total_reward'],
                'excluding_SFT_monitor_count': excluded['count'],
                'by_source': {source: {'answer_em': value['total_reward'], 'format_reward': value['format_reward'],
                                       'correct': value['correct'], 'count': value['count']}
                              for source, value in report['by_source'].items()}}
    before, after = metrics(initial), metrics(trained)
    return {'complete': True, 'created_at_utc': now(), 'selected_checkpoint': selection['best'],
            'last_checkpoint': selection['last'], 'test_used_for_selection': False,
            'merged_SFT_initial': before, 'best_GRPO': after,
            'answer_EM_delta': after['answer_em'] - before['answer_em'],
            'answer_EM_delta_percentage_points': 100 * (after['answer_em'] - before['answer_em']),
            'format_reward_delta': after['format_reward'] - before['format_reward'],
            'comparability_note': 'The paired merged-SFT initial policy above is the baseline for attributing GRPO changes. Compare models using the same questions, inference path and tool budgets.'}


def wandb_run_id(config, output):
    if os.environ.get('WANDB_RUN_ID'):
        return os.environ['WANDB_RUN_ID']
    value = config['search_r1'].get('wandb_run_id')
    if value:
        return value
    meta = output / 'runtime_meta.json'
    if meta.exists():
        value = json.loads(meta.read_text())
        for candidate in (value.get('wandb_run_id'), value.get('WANDB_RUN_ID'), value.get('environment', {}).get('WANDB_RUN_ID'), value.get('env', {}).get('WANDB_RUN_ID')):
            if candidate:
                return candidate
    found = set()
    for file in (output / 'wandb').glob('run-*/run-*.wandb'):
        match = re.fullmatch(r'run-([A-Za-z0-9]+)\.wandb', file.name)
        if match:
            found.add(match.group(1))
    if len(found) != 1:
        raise ValueError('Cannot uniquely determine the original training W&B run ID')
    return found.pop()


def log_wandb(config, result):
    import wandb
    p = paths(config)
    run_id = wandb_run_id(config, p['output'])
    os.environ.update(WANDB_LOG_MODEL='false', WANDB_WATCH='false', WANDB_DISABLE_CODE='true')
    run = wandb.init(project=config['trainer']['project_name'], entity=config['search_r1']['wandb_entity'],
                     id=run_id, resume='must', settings=wandb.Settings(init_timeout=30), save_code=False)
    values = {'test/merged_sft_initial_answer_em': result['merged_SFT_initial']['answer_em'],
              'test/merged_sft_initial_format_reward': result['merged_SFT_initial']['format_reward'],
              'test/best_grpo_answer_em': result['best_GRPO']['answer_em'],
              'test/best_grpo_total_reward': result['best_GRPO']['answer_em'],
              'test/best_grpo_format_reward': result['best_GRPO']['format_reward'],
              'test/answer_em_delta': result['answer_EM_delta'],
              'test/selected_global_step': result['selected_checkpoint']['global_step'],
              'test/selected_dev_answer_em': result['selected_checkpoint']['val_answer_em'],
              'test/excluding_sft_monitor_answer_em': result['best_GRPO']['excluding_SFT_monitor_em']}
    for source, metrics in result['best_GRPO']['by_source'].items():
        values[f'test/{source}_answer_em'] = metrics['answer_em']
    run.log(values)
    run.summary.update(values)
    record = {'run_id': run_id, 'url': run.url, 'metrics': values, 'model_or_code_uploaded': False}
    run.finish()
    return record


def finish(config, prepare_only=False):
    p = paths(config)
    configurations = make_eval_configs(config)
    if prepare_only:
        return {'prepared_only': True, 'test_inference_started': False,
                'configs': {key: str(value) for key, value in configurations.items()}}
    status_path = p['final'] / 'status.json'
    def status(phase, **extra):
        atomic_json(status_path, {'phase': phase, 'updated_at_utc': now(), **extra})
    with (p['final'] / '.finish.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status('freezing_independent_dev_selection')
            selection = freeze_selection(config)
            status('exporting_selected_GRPO_adapter', selected_global_step=selection['best']['global_step'])
            export_adapter(config, 0, p['output'] / 'initial_zero_adapter')
            export_adapter(config, selection['best']['global_step'], p['output'] / 'final_adapter',
                           Path(selection['best']['actor_checkpoint']) if selection['best']['global_step'] else None)
            reports = {}
            for name, path in configurations.items():
                status('final_test_' + name, selection_frozen=True)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = ''
                for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'PYTHONPATH', 'PYTHONHOME'):
                    env.pop(key, None)
                log = p['final'] / f'{name}.evaluation.log'
                report_path = Path(json.loads(path.read_text())['output_dir']) / 'report.json'
                already_complete = report_path.is_file() and json.loads(report_path.read_text()).get('complete') is True
                script = 'report_full_embedding_eval.py' if already_complete else 'run_full_embedding_eval.py'
                command = [str(p['root'] / '.venv/bin/python'), str(p['root'] / 'scripts' / script), '--config', str(path)]
                if already_complete:
                    command.append('--full-hash')
                with log.open('ab') as stream:
                    subprocess.run(command, env=env, cwd=p['root'], stdout=stream, stderr=subprocess.STDOUT, check=True)
                reports[name] = json.loads(report_path.read_text())
            result = make_comparison(reports['merged_sft_initial'], reports['best_grpo'], selection)
            destination = p['output'] / 'final_evaluation/comparison.json'
            atomic_json(destination, result)
            status('logging_final_metrics_to_original_WandB_run')
            result['wandb'] = log_wandb(config, result)
            atomic_json(destination, result)
            text = (f"# GRPO final held-out comparison\n\nSelected checkpoint: step {selection['best']['global_step']}; "
                    f"independent dev EM {selection['best']['val_answer_em']:.2%}.\n\n"
                    f"| Same 1,000 questions | Answer EM | Format reward |\n|---|---:|---:|\n"
                    f"| Merged SFT initial | {result['merged_SFT_initial']['answer_em']:.2%} | {result['merged_SFT_initial']['format_reward']:.2%} |\n"
                    f"| Best GRPO | {result['best_GRPO']['answer_em']:.2%} | {result['best_GRPO']['format_reward']:.2%} |\n\n"
                    f"EM change: {result['answer_EM_delta_percentage_points']:+.2f} percentage points. "
                    f"Excluding the 32 old SFT-monitor questions: {result['best_GRPO']['excluding_SFT_monitor_em']:.2%}.\n\n"
                    f"The merged-SFT initial policy is the paired baseline under the same evaluation settings. "
                    f"The 1,000 test questions were first evaluated after dev-only checkpoint selection was frozen.\n")
            (destination.parent / 'RESULTS.md').write_text(text)
            status('complete', selected_global_step=selection['best']['global_step'], result_file=str(destination))
            return result
        except BaseException as error:
            status('failed', error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--export-checkpoint-only', type=Path, help='CPU export smoke-test native actor; no selection or test inference')
    parser.add_argument('--export-step', type=int)
    parser.add_argument('--export-dir', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.export_checkpoint_only is not None:
        if args.prepare_only or args.export_step is None or args.export_step < 1 or args.export_dir is None:
            raise ValueError('Checkpoint-only export requires positive --export-step and separate --export-dir')
        result = export_adapter(config, args.export_step, args.export_dir.resolve(), args.export_checkpoint_only.resolve())
    else:
        result = finish(config, args.prepare_only)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
