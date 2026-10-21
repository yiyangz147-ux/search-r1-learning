"""CPU checks for GPU isolation, frozen inputs and smoke-to-train gates."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_search_grpo as runner


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_environment_overrides_foreign_gpu_and_ray_settings_without_exposing_secrets(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'RAY_ADDRESS': 'foreign:6379',
                                     'RAY_NAMESPACE': 'foreign', 'WANDB_API_KEY': 'private-test-value'}):
            env = runner.build_environment(self.root, 'run-1', 'smoke')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], ','.join(runner.GPU_UUIDS.values()))
        self.assertNotIn('RAY_ADDRESS', env)
        self.assertNotIn('RAY_NAMESPACE', env)
        self.assertEqual(env['WANDB_MODE'], 'disabled')
        self.assertEqual(env['WANDB_API_KEY'], 'private-test-value')
        self.assertTrue(env['TMPDIR'].startswith(str(self.root)))
        self.assertEqual(env['NO_PROXY'], 'localhost,127.0.0.1,::1,192.0.2.10')
        self.assertNotIn('private-test-value', json.dumps(runner.command(self.root, [])))

    def test_formal_wandb_id_is_stable_and_explicitly_reaches_ray_workers(self):
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled', 'WANDB_RUN_ID': 'foreign-run'}):
            first = runner.build_environment(self.root, 'formal-1', 'train')
            finish = runner.build_environment(self.root, 'formal-1', 'finish')
        self.assertEqual(first['WANDB_MODE'], 'online')
        self.assertEqual(first['WANDB_RESUME'], 'allow')
        self.assertEqual(first['WANDB_RUN_ID'], finish['WANDB_RUN_ID'])
        self.assertNotEqual(first['WANDB_RUN_ID'], 'foreign-run')
        config = {'trainer': {'default_local_dir': '/media/6/formal', 'experiment_name': 'formal'}}
        values = runner.phase_overrides(config, 'train', 'formal-1')
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.SEARCH_GRPO_OUTPUT_DIR="/media/6/formal"', values)
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE="online"', values)
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_ID=' + json.dumps(first['WANDB_RUN_ID']), values)

    def test_vllm_ipc_path_stays_within_unix_socket_limit_and_reaches_workers(self):
        run_id = 'a' * 61
        path = runner.vllm_rpc_path(run_id, 'train')
        self.assertLess(len(path + '/' + '0' * 36), 104)
        self.assertNotEqual(path, runner.vllm_rpc_path(run_id, 'smoke'))
        config = {'trainer': {'default_local_dir': '/media/6/formal', 'experiment_name': 'formal'}}
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RPC_BASE_PATH=' + json.dumps(path),
                      runner.phase_overrides(config, 'train', run_id))

    def test_smoke_has_one_real_iteration_n5_and_separate_output(self):
        config = {'trainer': {'default_local_dir': '/media/6/model-output', 'experiment_name': 'experiment'}}
        values = runner.phase_overrides(config, 'smoke', 'attempt-2')
        for expected in ['data.train_max_samples=4', 'data.val_max_samples=4',
                         'data.train_batch_size=4', 'actor_rollout_ref.actor.ppo_mini_batch_size=4',
                         'actor_rollout_ref.rollout.n=5', 'trainer.total_training_steps=1',
                         'trainer.logger=[console]', 'trainer.resume_mode=disable',
                         'trainer.default_local_dir=/media/6/model-output_smoke/attempt-2']:
            self.assertIn(expected, values)
        command = runner.command(self.root, values)
        self.assertEqual(command[1:3], ['-m', 'verl.trainer.main_ppo'])
        self.assertNotEqual(runner.ray_overrides('A', 'train'), runner.ray_overrides('B', 'train'))
        self.assertNotEqual(runner.ray_overrides('A', 'train'), runner.ray_overrides('A', 'smoke'))
        self.assertIn('++ray_kwargs.ray_init.address=local', values)
        ray_tmp = next(value.split('=', 1)[1] for value in values if '._temp_dir=' in value)
        self.assertLess(len(ray_tmp), 30)

    def test_gpu_inventory_only_queries_approved_cards_and_detects_busy_job(self):
        inventory = '\n'.join(f'{index}, {uuid}, 23000' for index, uuid in runner.GPU_UUIDS.items())
        occupied = f'{runner.GPU_UUIDS[3]}, 12345, 500'
        with patch.object(runner.subprocess, 'check_output', side_effect=[inventory, occupied]) as call:
            result = runner.target_gpu_snapshot()
        self.assertFalse(result['ready'])
        self.assertEqual(result['compute_processes'][0]['pid'], '12345')
        for arguments in call.call_args_list:
            argv = arguments.args[0]
            self.assertEqual(argv[argv.index('-i') + 1], ','.join(runner.GPU_UUIDS.values()))

    def test_gpu_uuid_mapping_change_is_rejected(self):
        inventory = '\n'.join(f'{index}, {uuid}, 23000' for index, uuid in runner.GPU_UUIDS.items())
        inventory = inventory.replace('2, GPU-', '0, GPU-', 1)
        with patch.object(runner.subprocess, 'check_output', return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, 'mapping changed'):
                runner.target_gpu_snapshot()

    def test_manifest_checks_content_and_rejects_path_escape(self):
        path = self.root / 'data.txt'; path.write_text('bound')
        records = runner.verify_manifest_files(self.root, {'complete': True, 'files': {
            'data.txt': {'bytes': 5, 'sha256': runner.sha256(path)},
        }})
        self.assertEqual(records[0]['sha256'], runner.sha256(path))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            runner.verify_manifest_files(self.root, {'complete': True, 'files': {
                '../outside': {'bytes': 0, 'sha256': 'invalid'},
            }})
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            runner.verify_manifest_files(self.root, {'complete': True, 'files': {
                'data.txt': {'bytes': 5, 'sha256': 'wrong'},
            }})

    def test_frozen_binding_rejects_changed_config_or_code(self):
        code = self.root / 'code.py'; code.write_text('print(1)\n')
        config = {'test': True}; overrides = ['test=true']
        binding = {'formal_config': config, 'formal_overrides': overrides, 'files': [runner.fingerprint(code)]}
        binding['binding_sha256'] = runner.hashlib.sha256(runner.canonical(binding).encode()).hexdigest()
        path = self.root / 'run_manifest.json'; runner.atomic_json(path, binding)
        runner.verify_binding(path, config, overrides)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            runner.verify_binding(path, {'test': False}, overrides)
        code.write_text('print(2)\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            runner.verify_binding(path, config, overrides)

    def smoke_artifacts(self):
        output = self.root / 'smoke'; output.mkdir()
        audit = output / 'audit.jsonl'
        rows = [{'group_uid': f'group-{i // 5}', 'sample_id': f'question-{i // 5}'} for i in range(20)]
        audit.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        metrics = {'global_step': 1, 'metrics': {'actor/pg_loss': 0.1, 'actor/grad_norm': 0.2,
                   'train/total_reward': 0.5, 'train/format_reward': 1.0},
                   'audit': {'path': str(audit), 'sha256': runner.sha256(audit), 'count': 20}}
        (output / 'metrics.jsonl').write_text(json.dumps(metrics) + '\n')
        actor = output / 'global_step_1/actor'; actor.mkdir(parents=True)
        (actor / 'optimizer.pt').write_text('fixture')
        runner.atomic_json(output / 'runtime_parameter_audit.json', {
            'embedding_frozen': True, 'only_qv_lora_trainable': True, 'shared_embedding_tied': True})
        runner.atomic_json(output / 'exported_adapter/grpo_adapter_export.json', {
            'complete': True, 'source_binding': {'global_step': 1, 'source_actor': str(actor)},
            'frozen_backbone_weights_checked': 10, 'lora_B_nonzero_count': 1, 'file_sha256': {}})
        return output

    def test_smoke_acceptance_requires_actual_update_group5_checkpoint_and_parameter_audit(self):
        output = self.smoke_artifacts()
        accepted = runner.accept_smoke(output, self.root / 'accepted.json', 'binding-sha')
        self.assertEqual(accepted['trajectory_count'], 20)
        self.assertTrue(accepted['response_masks_independently_audited'])
        parameters = output / 'runtime_parameter_audit.json'
        parameters.write_text('{"embedding_frozen": false}')
        with self.assertRaisesRegex(ValueError, 'embedding_frozen'):
            runner.accept_smoke(output, self.root / 'accepted-2.json', 'binding-sha')

    def test_smoke_missing_or_duplicate_group_members_are_rejected(self):
        output = self.smoke_artifacts()
        audit = output / 'audit.jsonl'
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        rows[0]['group_uid'] = rows[-1]['group_uid']
        audit.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        metrics = json.loads((output / 'metrics.jsonl').read_text())
        metrics['audit']['sha256'] = runner.sha256(audit)
        (output / 'metrics.jsonl').write_text(json.dumps(metrics) + '\n')
        with self.assertRaisesRegex(ValueError, 'four questions'):
            runner.accept_smoke(output, self.root / 'accepted.json', 'binding-sha')

    def test_smoke_rejects_unchanged_adapter_or_failed_frozen_base_export(self):
        output = self.smoke_artifacts()
        proof_file = output / 'exported_adapter/grpo_adapter_export.json'
        proof = json.loads(proof_file.read_text())
        for key in ('lora_B_nonzero_count', 'frozen_backbone_weights_checked'):
            runner.atomic_json(proof_file, {**proof, key: 0})
            with self.assertRaisesRegex(ValueError, 'real updated adapter'):
                runner.accept_smoke(output, self.root / 'accepted.json', 'binding-sha')


if __name__ == '__main__':
    unittest.main()
