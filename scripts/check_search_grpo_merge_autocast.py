#!/usr/bin/env python3
"""Compare dynamic SFT and its merged candidate under the real BF16 eval context.

Validation questions only. This does not modify the existing exported model or
its manifest, and never reads or evaluates the held-out 1,000 test questions.
"""
from argparse import ArgumentParser
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

from prepare_search_grpo import (canonical_rows, capture_probe, compare_probes,
                                 read_rows, sha256, write_json)


def compare(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from full_embedding_lora import load_full_embedding_adapter
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('This scientific probe requires exactly the selected single visible GPU')
    source = Path(args.validation_file).resolve()
    questions = canonical_rows(read_rows(source), 'independent_dev')[:args.probe_count]
    if len(questions) != args.probe_count:
        raise ValueError('Not enough independent dev questions')
    output = Path(args.output_file).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite an earlier scientific comparison')
    device = torch.device('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False)
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'phase': 'load_dynamic_original_SFT'}), flush=True)
    original = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
               device_map='cpu', attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
    original = load_full_embedding_adapter(original, args.adapter_path, is_trainable=False).requires_grad_(False).to(device).eval()
    original_buffers = {name: str(value.dtype) for name, value in original.named_buffers()}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        before = capture_probe(original, tokenizer, questions, device, args.probe_tokens)
    del original
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'phase': 'load_standard_merged_SFT'}), flush=True)
    merged = AutoModelForCausalLM.from_pretrained(args.merged_model, torch_dtype=torch.bfloat16,
               device_map='cpu', attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
    merged = merged.requires_grad_(False).to(device).eval()
    merged_buffers = {name: str(value.dtype) for name, value in merged.named_buffers()}
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        after = capture_probe(merged, tokenizer, questions, device, args.probe_tokens)
    rows = compare_probes(before, after)
    record = {'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'compute_context': "torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
              'attention_implementation': 'sdpa (same as existing full_embedding_rag evaluator)',
              'validation_source': str(source), 'validation_sha256': sha256(source),
              'sample_ids': [row['sample_id'] for row in questions],
              'probe_max_new_tokens': args.probe_tokens, 'stop_strings': ['</search>', '</answer>'],
              'base_model': str(Path(args.base_model).resolve()),
              'original_SFT_adapter': str(Path(args.adapter_path).resolve()),
              'merged_SFT_model': str(Path(args.merged_model).resolve()),
              'original_adapter_sha256': sha256(Path(args.adapter_path) / 'adapter_model.safetensors'),
              'original_full_embedding_sha256': sha256(Path(args.adapter_path) / 'full_tied_embedding.safetensors'),
              'merged_export_manifest_sha256': sha256(Path(args.merged_model) / 'sft_export_manifest.json'),
              'original_buffers': original_buffers, 'merged_buffers': merged_buffers,
              'torch_peak_allocated_MiB': torch.cuda.max_memory_allocated() / 1024**2,
              'summary': {'samples': len(rows), 'same_next_token_top1': sum(x['next_token_top1_equal'] for x in rows),
                          'same_full_greedy': sum(x['greedy_tokens_equal'] for x in rows),
                          'same_complete_first_action': sum(x['first_action_equal'] is True for x in rows),
                          'valid_format_before': sum(x['valid_format_before'] for x in rows),
                          'valid_format_after': sum(x['valid_format_after'] for x in rows),
                          'max_abs_logit_difference': max(x['max_abs_logit_difference'] for x in rows),
                          'mean_next_token_KL': sum(x['next_token_KL_before_to_after'] for x in rows) / len(rows),
                          'max_next_token_KL': max(x['next_token_KL_before_to_after'] for x in rows)},
              'comparisons': rows,
              'interpretation': 'This matches the prior evaluator autocast and SDPA context; the no-autocast export probe is a separate numeric comparison. Eight dev actions do not establish equal whole-dataset EM. Keep a merged-policy step-zero independent-dev baseline.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    print(json.dumps({'complete': True, 'output_file': str(output), 'summary': record['summary']}), flush=True)
    return record


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ('base-model', 'adapter-path', 'merged-model', 'validation-file', 'output-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--probe-count', type=int, default=8)
    parser.add_argument('--probe-tokens', type=int, default=128)
    args = parser.parse_args()
    if args.probe_count < 8 or args.probe_tokens < 128:
        raise ValueError('Use at least eight dev questions with a 128-token action budget')
    compare(args)


if __name__ == '__main__':
    main()
