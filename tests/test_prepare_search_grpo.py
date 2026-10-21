"""Meaningful CPU contracts: no data leakage, raw Base prompts, full-SFT export."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_search_grpo import (ProtectedQuestions, canonical_rows, choose_validation,
                                 ensure_separate, export_model, plain_tied_model, read_rows, verl_record)


def row(sid, question, source='nq'):
    return {'sample_id': sid, 'question': question, 'golden_answers': ['test answer'],
            'data_source': source}


class DataContracts(unittest.TestCase):
    def test_near_duplicate_threshold_and_punctuation(self):
        protected = ProtectedQuestions([row('seen', 'Who Wrote: the test novel, in 1984?')])
        self.assertTrue(protected.matches(row('new', 'who wrote the test novel in 1984')))
        self.assertFalse(protected.matches(row('new', 'which scientist invented the steam engine')))
        with self.assertRaises(ValueError):
            ensure_separate([row('new', 'Who wrote the test novel in 1984?')], protected, 'dev')

    def test_balanced_deterministic_reserve_and_exclusion(self):
        rows = [row(f'{source}-{i}', f'{source} unique{i} query{i} topic{i}', source)
                for source in ('nq', 'hotpotqa') for i in range(5)]
        protected = ProtectedQuestions([rows[0]])
        selected, eligible, excluded = choose_validation(rows, protected, 42, 2)
        selected2, _, _ = choose_validation(list(reversed(rows)), protected, 42, 2)
        self.assertEqual([r['sample_id'] for r in selected], [r['sample_id'] for r in selected2])
        self.assertEqual(sum(r['data_source'] == 'nq' for r in selected), 2)
        self.assertEqual(sum(r['data_source'] == 'hotpotqa' for r in selected), 2)
        self.assertEqual(len(eligible), 9)
        self.assertEqual(excluded[0]['sample_id'], rows[0]['sample_id'])
        self.assertNotIn(rows[0]['sample_id'], [r['sample_id'] for r in selected])

    def test_verl_record_is_raw_base_not_chat(self):
        source = row('id', 'which test')
        record = verl_record(source, 'train', 0)
        self.assertIsInstance(record['prompt'], str)
        self.assertIn('Question: which test?', record['prompt'])
        self.assertNotIn('<|im_start|>', record['prompt'])
        self.assertEqual(record['agent_name'], 'search_r1_base')
        self.assertEqual(record['reward_model']['ground_truth']['target'], ['test answer'])

    def test_reject_repeated_ids_and_empty_aliases(self):
        with self.assertRaises(ValueError):
            canonical_rows([row('id', 'x'), row('id', 'y')], 'test')
        with self.assertRaises(ValueError):
            canonical_rows([dict(row('id', 'x'), golden_answers=[])], 'test')


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('peft'), 'requires original SFT environment')
class TinyModelExport(unittest.TestCase):
    def test_plain_conversion_preserves_FP32_rotary_buffers(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from full_embedding_lora import configure_full_embedding_training
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
        for name, buffer in model.named_buffers():
            if "inv_freq" in name:
                buffer.data = buffer.data.float()
        configure_full_embedding_training(model)
        before = {name: value.detach().clone() for name, value in model.named_buffers()}
        plain_tied_model(model)
        self.assertEqual(model.get_input_embeddings().weight.dtype, torch.bfloat16)
        self.assertEqual(model.model.rotary_emb.inv_freq.dtype, torch.float32)
        for name, value in model.named_buffers():
            self.assertEqual(value.dtype, before[name].dtype)
            self.assertTrue(torch.equal(value, before[name]))

    def test_full_embedding_and_lora_merge_export_reload(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from full_embedding_lora import configure_full_embedding_training, save_full_embedding_weights
        torch.set_num_threads(2)
        torch.manual_seed(91)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=512, tie_word_embeddings=True,
                            pad_token_id=1, eos_token_id=2, attention_dropout=0.)
        config._attn_implementation = 'eager'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
            base.save_pretrained(root / 'base')
            vocabulary = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, **{f'token{i}': i for i in range(3, 64)}}
            backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
            tokenizer.save_pretrained(root / 'base')
            model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2,
                            lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.))
            configure_full_embedding_training(model)
            with torch.no_grad():
                model.get_input_embeddings().weight[50].add_(.17)
                model.get_input_embeddings().weight[1].add_(.09)
                for name, module in model.named_modules():
                    if hasattr(module, 'lora_B'):
                        module.lora_B['default'].weight.normal_(0., .03)
            model.save_pretrained(root / 'adapter', save_embedding_layers=False)
            save_full_embedding_weights(model, root / 'adapter')
            validation = root / 'val.jsonl'
            validation.write_text(json.dumps(row('probe', 'a simple question')) + '\n')
            args = Namespace(base_model=str(root / 'base'), adapter_path=str(root / 'adapter'),
                             output_dir=str(root / 'export'), validation_file=str(validation),
                             verify_device='cpu', probe_count=1, probe_tokens=2, cpu_threads=2,
                             max_logit_difference=1.)
            result = export_model(args)
            self.assertTrue(result['complete'])
            self.assertTrue(result['shared_embedding']['tied'])
            self.assertGreater(result['shared_embedding']['sample_source_vs_base_abs_max'], 0.)
            self.assertEqual(len(result['merged_lora_targets']), 2)
            self.assertTrue(result['reload_probe_comparisons'][0]['greedy_tokens_equal'])
            self.assertEqual(result['reload_probe_comparisons'][0]['max_abs_logit_difference'], 0.)
            reloaded = Qwen3ForCausalLM.from_pretrained(root / 'export', torch_dtype=torch.bfloat16)
            self.assertTrue(torch.equal(reloaded.get_input_embeddings().weight[50],
                                        model.get_input_embeddings().weight[50].to(torch.bfloat16)))
            self.assertIs(reloaded.get_input_embeddings().weight, reloaded.get_output_embeddings().weight)
            self.assertEqual(export_model(args)['files'], result['files'])


if __name__ == '__main__':
    unittest.main()
