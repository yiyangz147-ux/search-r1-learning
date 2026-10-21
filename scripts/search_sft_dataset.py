"""Hugging Face Trainer-compatible dataset for the raw Search-R1 continuation.

Base models continue the exported Search-R1 prompt as plain text. For an
instruction model, explicitly select prompt_format="chat" to apply its chat
template only to the original user prompt. Append past assistant actions and
information blocks inside the same continuation. Labels supervise only the
current assistant action.
Use DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100).
"""

from collections import Counter
from pathlib import Path


class SearchSFTDataset:
    def __init__(self, parquet_file, tokenizer, max_length=4096, prompt_format="base"):
        import pyarrow.parquet as pq
        if prompt_format not in {"base", "chat"}:
            raise ValueError("prompt_format must be 'base' or 'chat'")
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for loss masking by character offset")
        self.items = []
        self.dropped = Counter()
        self.dropped_sample_ids = set()
        source_rows = pq.read_table(Path(parquet_file)).to_pylist()
        encoded = []
        for row in source_rows:
            prompt = row["question"]
            if prompt_format == "chat":
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                # Some tokenizer templates pre-open the assistant's think block.
                if prompt.endswith("<think>\n"):
                    prompt = prompt[:-len("<think>\n")]
            prefix = prompt + row["history"]
            text = prefix + row["response"]
            tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = tokenized["input_ids"]
            labels = [token if start >= len(prefix) and end > start else -100
                      for token, (start, end) in zip(ids, tokenized["offset_mapping"])]
            # Search steps end at </search>; the environment resumes generation.
            # Only the final answer teaches EOS.
            if "<answer>" in row["response"]:
                if tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer requires EOS for final-answer supervision")
                ids.append(tokenizer.eos_token_id)
                labels.append(tokenizer.eos_token_id)
            reason = "over_max_length" if len(ids) > max_length else "no_response_tokens" if all(x == -100 for x in labels) else None
            if reason:
                self.dropped[reason] += 1
                self.dropped_sample_ids.add(row["sample_id"])
            encoded.append((row["sample_id"], {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}))
        # Retain or drop an entire question's trajectory, including all search steps.
        self.items = [item for sid, item in encoded if sid not in self.dropped_sample_ids]
        self.input_steps = len(source_rows)
        self.retained_steps = len(self.items)
        if not self.items:
            raise ValueError("No steps fit the token budget; inspect dropped episodes or increase max_length")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
