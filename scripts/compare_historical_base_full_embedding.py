#!/usr/bin/env python3
"""Compare unchanged historical Base predictions with the completed SFT run.

The questions and scoring are paired. The historical few-shot/fixed-retrieval
protocol differs from the new agent protocol, so this is a system comparison.
"""

import argparse
from pathlib import Path

from prepare_full_embedding_eval import atomic_json, read_json, read_jsonl, sha256, verify_run
from report_full_embedding_eval import load_predictions
from qa_metrics import score_answer


def historical_records(path, examples):
    indexed = {row['sample_id']: row for row in examples}
    records = {}
    for row in read_jsonl(path):
        sid = row['sample_id']
        if sid not in indexed or sid in records:
            raise ValueError('Historical predictions have an unknown or duplicate sample ID')
        example = indexed[sid]
        for key in ('question', 'data_source', 'golden_answers'):
            if row[key] != example[key]:
                raise ValueError(f'Historical {key} differs from the selected question: {sid}')
        answer = row['answer'] if row['status'] == 'ok' else ''
        primary = score_answer(answer, example['golden_answers'], 'search-r1')
        source = score_answer(answer, example['golden_answers'], example['data_source'])
        if source['em'] != row['em']:
            raise ValueError(f'Historical stored EM disagrees with independent rescoring: {sid}')
        records[sid] = {'total_reward': primary['em'], 'qa_em': source['em'],
                        'qa_f1': source['f1'], 'answer': answer}
    if set(records) != set(indexed):
        raise ValueError('Historical predictions do not cover the identical question set')
    return records


def comparison_group(examples, current, historical):
    count = len(examples)
    if not count:
        return {'selected': 0}
    ids = [row['sample_id'] for row in examples]
    new_correct = sum(current[sid]['total_reward'] for sid in ids)
    output = {'selected': count, 'sft': {
        'correct': new_correct, 'total_reward': new_correct / count,
        'em_percent': 100 * new_correct / count,
        'format_reward': sum(current[sid]['format_reward'] for sid in ids) / count,
        'qa_em': sum(current[sid]['qa_em'] for sid in ids) / count,
    }, 'historical_base': {}}
    for name, rows in historical.items():
        old_correct = sum(rows[sid]['total_reward'] for sid in ids)
        output['historical_base'][name] = {
            'correct': old_correct, 'total_reward': old_correct / count,
            'em_percent': 100 * old_correct / count,
            'source_metric_em_percent': 100 * sum(rows[sid]['qa_em'] for sid in ids) / count,
            'sft_minus_base_percentage_points': 100 * (new_correct - old_correct) / count,
            'paired_sft_only_correct': sum(current[sid]['total_reward'] == 1 and rows[sid]['total_reward'] == 0 for sid in ids),
            'paired_base_only_correct': sum(current[sid]['total_reward'] == 0 and rows[sid]['total_reward'] == 1 for sid in ids),
            'paired_both_correct': sum(current[sid]['total_reward'] == 1 and rows[sid]['total_reward'] == 1 for sid in ids),
        }
    return output


def compare(config_path, historical_dir):
    config, manifest, examples = verify_run(config_path)
    current, _, _, _ = load_predictions(config, manifest, examples)
    if len(current) != len(examples):
        raise ValueError('Wait for all selected questions before publishing a final comparison')
    historical_dir = Path(historical_dir)
    paths = {name: historical_dir / f'predictions-{name}.jsonl' for name in ('no-rag', 'rag')}
    historical = {name: historical_records(path, examples) for name, path in paths.items()}
    monitor_ids = set(manifest.get('training_monitor_subset', {}).get('evaluation_sample_ids', []))
    excluded = [row for row in examples if row['sample_id'] not in monitor_ids]
    groups = {'overall': examples, 'excluding_training_monitor': excluded,
              'training_monitor': [row for row in examples if row['sample_id'] in monitor_ids]}
    result = {
        'complete': True, 'primary_metric': 'Search-R1 normalized answer EM; format reward separate',
        'same_questions': True, 'same_generation_protocol': False,
        'comparison_scope': 'Historical Base used simple few-shot QA, 64 new tokens, fixed top-3 retrieval before answering; SFT uses Search-R1 agent prompt and model-directed retrieval with 512 tokens/action, 2048 total, up to 3 calls. Differences cannot be attributed solely to SFT.',
        'historical_format_reward': None,
        'format_comparison_note': 'Historical direct QA did not require think/search/answer tags; its format reward is not comparable.',
        'historical_predictions_sha256': {name: sha256(path) for name, path in paths.items()},
        'evaluation_data_sha256': manifest['data_sha256'],
        'groups': {name: comparison_group(rows, current, historical) for name, rows in groups.items()},
        'by_source': {source: comparison_group([row for row in examples if row['data_source'] == source], current, historical)
                      for source in sorted({row['data_source'] for row in examples})},
    }
    output = Path(config['output_dir']) / 'historical_base_comparison.json'
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--historical-dir', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.config, args.historical_dir)
    import json
    print(json.dumps(result['groups']['overall'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
