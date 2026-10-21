"""Check complete Search-R1 episodes with the student's actual Base tokenizer.

The strings, boundary masking and final-answer EOS match SearchSFTDataset's
``prompt_format='base'`` path. No model weights or teacher calls are required.
"""

import hashlib
import importlib.metadata
import json

from prepare_data import PROMPT


def tokenizer_provenance(tokenizer):
    """Fingerprint the loaded tokenizer, rather than trusting a model alias."""
    if tokenizer is None:
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialized = backend.to_str() if backend is not None else None
    special_tokens = json.dumps(getattr(tokenizer, "special_tokens_map", {}),
                                ensure_ascii=False, sort_keys=True, default=str)
    versions = {}
    for package in ("transformers", "tokenizers"):
        try:
            versions[package + "_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package + "_version"] = None
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "backend_sha256": hashlib.sha256(serialized.encode()).hexdigest() if serialized is not None else None,
        "special_tokens_sha256": hashlib.sha256(special_tokens.encode()).hexdigest(),
        **versions,
    }


def episode_steps(sample, events):
    """Yield exactly the Base-format strings later written by SFT export."""
    question = sample["question"]
    if not question.endswith("?"):
        question += "?"
    prompt = PROMPT.format(question=question)
    history = ""
    for event in events:
        response = event["content"] + "\n"
        if event["role"] == "assistant":
            yield {"question": prompt, "history": history, "response": response}
        history += response


def check_episode_length(sample, events, tokenizer, max_length):
    """Reject the entire question if any assistant step cannot be trained.

    ``supervised_tokens`` counts only tokens wholly inside this step's response,
    plus EOS for the final answer. A boundary-straddling token stays masked.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Student length checks require a fast tokenizer")
    if max_length < 1:
        raise ValueError("Student max length must be positive")
    if tokenizer.eos_token_id is None:
        raise ValueError("Student tokenizer requires EOS for final-answer supervision")
    step_lengths, supervised_tokens = [], []
    for step in episode_steps(sample, events):
        prefix = step["question"] + step["history"]
        response = step["response"]
        encoded = tokenizer(prefix + response, add_special_tokens=False,
                            return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) != len(offsets):
            raise ValueError("Student tokenizer returned inconsistent offsets")
        count = sum(start >= len(prefix) and end > start for start, end in offsets)
        eos = int("<answer>" in response)
        step_lengths.append(len(ids) + eos)
        supervised_tokens.append(count + eos)
    reason = ("no_assistant_steps" if not step_lengths else
              "over_max_length" if any(n > max_length for n in step_lengths) else
              "no_response_tokens" if any(n == 0 for n in supervised_tokens) else "ok")
    return {
        "passed": reason == "ok", "reason": reason,
        "max_length": max_length, "step_lengths": step_lengths,
        "max_step_tokens": max(step_lengths, default=0),
        "supervised_tokens": supervised_tokens, "prompt_format": "base",
    }
