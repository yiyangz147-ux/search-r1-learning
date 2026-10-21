"""Raw Base prompts for Search-R1 GRPO; intentionally no chat templating."""
from __future__ import annotations
import copy
from pathlib import Path
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from search_sft_protocol import build_prompt

class SearchR1BaseDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is not None:
            raise ValueError('This experiment expects a text-only Base tokenizer')
        self.original_data_files=[str(data_files)] if isinstance(data_files,(str,Path)) else list(data_files)
        self.data_files=list(self.original_data_files)
        self.tokenizer=tokenizer
        self.config=config
        self.max_samples=max_samples
        self.max_prompt_length=int(config.get('max_prompt_length',1024))
        self.prompt_key=config.get('prompt_key','prompt')
        self.serialize_dataset=True
        self._read_rows()
    def _read_rows(self):
        self.rows=[]
        for file in self.data_files:
            self.rows.extend(pq.read_table(file).to_pylist())
        if self.max_samples>0:
            self.rows=self.rows[:self.max_samples]
        seen=set()
        self.prompt_ids=[]
        for row in self.rows:
            info=row.get('extra_info') or {}
            sid=info.get('sample_id')
            if not sid or sid in seen:
                raise ValueError('Missing or duplicate sample_id in selected dataset')
            seen.add(sid)
            prompt=row[self.prompt_key]
            if not isinstance(prompt,str) or prompt!=build_prompt(info['question']):
                raise ValueError('Dataset prompt must equal the raw Search-R1 Base prompt')
            if row.get('agent_name')!='search_r1_base':
                raise ValueError('Unexpected agent_name')
            ids=tokenizer_ids(self.tokenizer,prompt)
            if not ids or len(ids)>self.max_prompt_length:
                raise ValueError(f'Prompt length {len(ids)} exceeds {self.max_prompt_length}: {sid}')
            self.prompt_ids.append(ids)
        if not self.rows:
            raise ValueError('Empty selected dataset')
    def __len__(self):return len(self.rows)
    def resume_dataset_state(self):self._read_rows()
    def __getitem__(self,index):
        row=copy.deepcopy(self.rows[index])
        prompt=row.pop(self.prompt_key)
        ids=self.prompt_ids[index]
        length=len(ids)
        pad=self.tokenizer.pad_token_id
        if pad is None:raise ValueError('A pad token is required')
        padding=self.max_prompt_length-length
        input_ids=torch.tensor([pad]*padding+ids,dtype=torch.long)
        attention_mask=torch.tensor([0]*padding+[1]*length,dtype=torch.long)
        position_ids=(attention_mask.cumsum(-1)-1).clamp_min(0)*attention_mask
        row.update(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids,
                   raw_prompt_ids=list(ids),raw_prompt=[{'role':'user','content':prompt}])
        info=row['extra_info']
        row['index']=info.get('index',index)
        row['tools_kwargs']={}
        row['interaction_kwargs']={}
        return row

def tokenizer_ids(tokenizer,text):
    ids=tokenizer.encode(text,add_special_tokens=False)
    return list(ids)
