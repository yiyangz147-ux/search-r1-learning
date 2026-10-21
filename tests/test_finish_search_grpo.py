"""CPU contracts for dev-only selection and faithful new-GRPO adapter export."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finish_search_grpo import (canonical_lora_key, choose_best, make_comparison,
                               adapter_state_from_shards, expected_adapter_shapes, make_eval_configs, merge_values, wandb_run_id)


def _save_native_cpu_rank(rank, world, init_file, output, weights):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=world)
    mesh = init_device_mesh('cpu', (world,), mesh_dim_names=('fsdp',))
    state = {key: distribute_tensor(value.detach(), mesh, [Shard(0)]) for key, value in weights.items()}
    torch.save(state, str(Path(output) / f'model_world_size_{world}_rank_{rank}.pt'))
    dist.barrier()
    dist.destroy_process_group()


class FinishSelection(unittest.TestCase):
    def test_dev_selection_includes_zero_tie_earlier(self):
        items = [{'global_step': 0, 'val_answer_em': .3}, {'global_step': 50, 'val_answer_em': .3},
                 {'global_step': 100, 'val_answer_em': .29}]
        self.assertEqual(choose_best(items)['global_step'], 0)
        self.assertEqual(choose_best(items + [{'global_step': 150, 'val_answer_em': .4}])['global_step'], 150)
        with self.assertRaises(ValueError):
            choose_best(items[1:])
        with self.assertRaises(ValueError):
            choose_best(items + [{'global_step': 50, 'val_answer_em': .31}])

    def test_adapter_key_never_accepts_old_or_other_targets(self):
        key = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
        self.assertEqual(canonical_lora_key(key), key.replace('.default.', '.'))
        for invalid in (key.replace('q_proj', 'k_proj'), key.replace('.default.', '.old_SFT.'),
                        key.replace('q_proj', 'embed_tokens')):
            with self.assertRaises(ValueError):
                canonical_lora_key(invalid)

    def test_prepare_only_builds_plans_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'prepared/data'
            data.mkdir(parents=True)
            (data / 'data_manifest.json').write_text(json.dumps({'complete': True, 'files': {'test.selected.jsonl': {'sha256': 'a'*64}}}))
            cfg = {'search_r1': {'root': str(root), 'max_new_tokens_per_action': 512,
                                'max_generated_tokens_per_question': 2048, 'max_context_tokens': 8192,
                                'max_searches': 3, 'max_turns': 6, 'max_invalid_actions': 2,
                                'top_k': 3, 'max_chars_per_passage': 1800,
                                'retrieval_url': 'http://127.0.0.1:8000/retrieve',
                                'retrieval_timeout_seconds': 180, 'retrieval_retries': 2},
                   'trainer': {'default_local_dir': str(root / 'output')},
                   'data': {'train_files': str(data / 'train.parquet')},
                   'actor_rollout_ref': {'model': {'path': str(root / 'merged')}}}
            plans = make_eval_configs(cfg)
            self.assertEqual(len(plans), 2)
            for path in plans.values():
                value = json.loads(path.read_text())
                self.assertEqual(value['expected_count'], 1000)
                self.assertEqual(value['model_path'], str((root / 'merged').resolve()))
                self.assertEqual(value['batch_size'], 32)
                self.assertEqual(value['generation_microbatch_size'], 8)
                self.assertFalse(Path(value['output_dir']).exists())
            self.assertEqual(make_eval_configs(cfg), plans)

    def test_paired_metric_uses_merged_initial_not_historical(self):
        def report(em):
            group = {'total_reward': em, 'format_reward': .9, 'correct': int(em*1000), 'count': 1000, 'selected': 1000}
            return {'complete': True, 'data_sha256': 'same', 'selected_sha256': 'same', 'base_model_path': 'merged', 'overall': group, 'by_source': {'nq': dict(group, count=500)},
                    'excluding_training_monitor': {'overall': dict(group, count=968)}}
        value = make_comparison(report(.28), report(.33), {'best': {'global_step': 50}, 'last': {'global_step': 300}})
        self.assertAlmostEqual(value['answer_EM_delta_percentage_points'], 5.)
        self.assertNotIn('historical_dynamic_SFT_EM_context', value)
        self.assertIn('paired merged-SFT initial policy', value['comparability_note'])
        self.assertFalse(value['test_used_for_selection'])
        with self.assertRaises(ValueError):
            make_comparison(dict(report(.28), complete=False), report(.33), {'best': {}, 'last': {}})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires training/SFT torch environment')
class NativeShardContracts(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'native CPU Gloo checkpoint integration runs on server')
    def test_real_four_rank_DTensors_export_new_adapter_and_verify_frozen_base(self):
        import torch
        import multiprocessing
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
            base.save_pretrained(root / 'base')
            actor = root / 'actor'
            actor.mkdir()
            (actor / 'fsdp_config.json').write_text(json.dumps({'world_size': 4, 'FSDP_version': 1}))
            shapes = expected_adapter_shapes(root / 'base', 32)
            adapters = {key: torch.arange(torch.tensor(shape).prod()).reshape(shape).float() / 1234567
                        for key, shape in shapes.items()}
            weights = {}
            for key, value in base.state_dict().items():
                if key.endswith(('.q_proj.weight', '.v_proj.weight')):
                    key = key.replace('.weight', '.base_layer.weight')
                weights['base_model.model.' + key] = value.detach()
            weights.update({key.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight'): value
                            for key, value in adapters.items()})
            context = multiprocessing.get_context('fork')
            workers = [context.Process(target=_save_native_cpu_rank,
                       args=(rank, 4, str(root / 'gloo_init'), str(actor), weights)) for rank in range(4)]
            for process in workers: process.start()
            for process in workers:
                process.join(40)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    self.fail('Synthetic CPU FSDP checkpoint timed out')
                self.assertEqual(process.exitcode, 0)
            exported, proof = adapter_state_from_shards(actor, root / 'base', 32)
            self.assertEqual(set(exported), set(adapters))
            for key, expected in adapters.items():
                self.assertEqual(exported[key].dtype, torch.float32)
                self.assertTrue(torch.equal(exported[key], expected))
            self.assertGreater(proof['frozen_backbone_weights_checked'], 0)
            # A supposedly frozen embedding update must block export, even when
            # all adapter tensors and shard shapes remain valid.
            path = actor / 'model_world_size_4_rank_0.pt'
            corrupt = torch.load(path, map_location='cpu', weights_only=False)
            corrupt['base_model.model.model.embed_tokens.weight']._local_tensor[0, 0] += 1
            torch.save(corrupt, path)
            with self.assertRaisesRegex(ValueError, 'frozen SFT backbone'):
                adapter_state_from_shards(actor, root / 'base', 32)

    def test_dtensor_merge_preserves_FP32_adapter(self):
        import torch
        class Placement:
            dim = 0
            def is_shard(self): return True
            def is_replicate(self): return False
        placement = Placement()
        actual = torch.tensor([[.100001, .200003], [.300007, .400009]], dtype=torch.float32)
        values = [SimpleNamespace(_local_tensor=part, placements=(placement,), shape=actual.shape)
                  for part in actual.chunk(2, dim=0)]
        merged = merge_values(values, actual.shape)
        self.assertEqual(merged.dtype, torch.float32)
        self.assertTrue(torch.equal(merged, actual))
        self.assertFalse(torch.equal(merged, actual.bfloat16().float()))

    def test_ambiguous_plain_shards_are_not_guessed(self):
        import torch
        with self.assertRaises(ValueError):
            merge_values([torch.ones(1, 2), torch.ones(1, 2)], (2, 2))
        tensor = torch.ones(2, 2)
        self.assertTrue(torch.equal(merge_values([tensor, tensor.clone()], tensor.shape), tensor))


if __name__ == '__main__':
    unittest.main()
