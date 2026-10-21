"""Train and checkpoint a complete tied embedding alongside an ordinary LoRA.

PEFT's normal adapter checkpoint deliberately excludes this base-model weight.
Every checkpoint therefore needs BOTH adapter_model.safetensors and the full
tied-embedding files below.  Use ``load_full_embedding_adapter`` for inference,
and ``FullEmbeddingTrainer`` for training/resume/best-checkpoint restoration.

The embedding and LM head retain the *same Parameter object*.  Its master
weight, gradient, and Adam states stay FP32; embedding activations and the head
projection use the backbone dtype.  No modules_to_save copies are introduced.
Only ordinary, unsharded DDP is supported by this small local recipe helper.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MethodType
import uuid

import torch
from torch.nn import functional as F


WEIGHTS_NAME = "full_tied_embedding.safetensors"
METADATA_NAME = "full_tied_embedding.json"
TENSOR_NAME = "shared_embedding_weight"


def _unwrap(model):
    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _embedding_output_cast(module, _args, output):
    # Embedding lookup itself does not participate in autocast.  Without this
    # explicit cast, making its weight FP32 also promotes residual activations.
    return output.to(dtype=module._full_embedding_compute_dtype)


def _mixed_precision_head_forward(module, hidden_states):
    # An explicit differentiable cast also supports generate() outside Trainer's
    # autocast context.  Gradients flow back to the same FP32 master Parameter.
    return F.linear(hidden_states, module.weight.to(dtype=hidden_states.dtype), module.bias)


def assert_full_embedding_tied(model, *, require_trainable=None):
    model = _unwrap(model)
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if not isinstance(embedding, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        raise ValueError("Full embedding training requires plain Embedding and Linear modules")
    if not getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("The base config must have tie_word_embeddings=true")
    if head.bias is not None:
        raise ValueError("This tied-embedding recipe expects an unbiased LM head")
    if embedding.weight is not head.weight or embedding.weight.data_ptr() != head.weight.data_ptr():
        raise ValueError("Input embedding and output head must share one Parameter object")
    if embedding.weight.dtype != torch.float32:
        raise ValueError("The trainable tied embedding master weight must remain FP32")
    if require_trainable is not None and embedding.weight.requires_grad != require_trainable:
        raise ValueError("The tied embedding has the wrong requires_grad flag")
    return embedding.weight


def configure_full_embedding_training(model, *, compute_dtype=torch.bfloat16, trainable=True):
    """Call after get_peft_model, before creating DDP or the optimizer.

    Repeated calls preserve the Parameter object, optimizer references, and a
    single output-casting hook.  The rest of PEFT's trainability flags are kept.
    """
    model = _unwrap(model)
    if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported embedding activation dtype")
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if not isinstance(embedding, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        raise ValueError("Do not wrap embed_tokens or lm_head with LoRA/modules_to_save")
    if embedding.weight is not head.weight or not getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("Refusing to train separate input/output embedding weights")
    if head.bias is not None or embedding.weight.device.type == "meta":
        raise ValueError("Expected a loaded, unbiased tied head")
    # Assign data, not a new Parameter: the input and output modules must continue
    # sharing the same object, and a resume must retain optimizer references.
    if embedding.weight.dtype != torch.float32:
        embedding.weight.data = embedding.weight.data.to(dtype=torch.float32)
    embedding.weight.requires_grad_(trainable)
    embedding._full_embedding_compute_dtype = compute_dtype
    if not hasattr(embedding, "_full_embedding_cast_handle"):
        embedding._full_embedding_cast_handle = embedding.register_forward_hook(_embedding_output_cast)
    if not hasattr(head, "_full_embedding_original_forward"):
        head._full_embedding_original_forward = head.forward
        head.forward = MethodType(_mixed_precision_head_forward, head)
    assert_full_embedding_tied(model, require_trainable=trainable)
    return {"shape": list(embedding.weight.shape), "parameters": embedding.weight.numel(),
            "master_dtype": "float32", "compute_dtype": _dtype_name(compute_dtype),
            "tied": True, "trainable": trainable}


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_full_embedding_weights(model, output_dir):
    """Save one FP32 matrix once; caller decides which DDP rank writes.

    Contains no collective operations.  A metadata file is written last, after
    the weight file has been atomically installed and SHA-256 fingerprinted.
    """
    from safetensors.torch import save_file

    model = _unwrap(model)
    weight = assert_full_embedding_tied(model)
    embedding = model.get_input_embeddings()
    if not hasattr(embedding, "_full_embedding_compute_dtype"):
        raise ValueError("Configure full embedding training before checkpointing")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / WEIGHTS_NAME
    temporary = output_dir / f".{WEIGHTS_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        # CPU copy avoids retaining another full-size tensor on the GPU.  clone
        # also gives a CPU-only test/save a stable snapshot of the live weight.
        tensor = weight.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        save_file({TENSOR_NAME: tensor}, str(temporary), metadata={"format": "pt"})
        del tensor
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {"format_version": 1, "weight_file": WEIGHTS_NAME,
                "tensor_name": TENSOR_NAME, "shape": list(weight.shape),
                "master_dtype": "float32", "tied": True,
                "compute_dtype": _dtype_name(embedding._full_embedding_compute_dtype),
                "bytes": path.stat().st_size, "sha256": _sha256(path)}
    _atomic_json(output_dir / METADATA_NAME, metadata)
    return metadata


def load_full_embedding_weights(model, checkpoint_dir, *, trainable=True):
    """Restore in place so any existing optimizer references remain valid."""
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / METADATA_NAME).read_text(encoding="utf-8"))
    path = checkpoint_dir / WEIGHTS_NAME
    if (metadata.get("format_version") != 1 or metadata.get("weight_file") != WEIGHTS_NAME
            or metadata.get("tensor_name") != TENSOR_NAME or metadata.get("master_dtype") != "float32"
            or metadata.get("tied") is not True or path.stat().st_size != metadata.get("bytes")
            or _sha256(path) != metadata.get("sha256")):
        raise ValueError("Missing, incompatible, or corrupted full tied embedding checkpoint")
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtypes.get(metadata.get("compute_dtype"))
    if compute_dtype is None:
        raise ValueError("Unsupported saved embedding compute dtype")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {TENSOR_NAME}:
        raise ValueError("The full embedding checkpoint has unexpected tensors")
    saved = tensors[TENSOR_NAME]
    model = _unwrap(model)
    if (saved.dtype != torch.float32 or list(saved.shape) != metadata.get("shape")
            or saved.shape != model.get_input_embeddings().weight.shape):
        raise ValueError("The full embedding checkpoint dtype or shape does not match this model")
    configure_full_embedding_training(model, compute_dtype=compute_dtype, trainable=trainable)
    weight = assert_full_embedding_tied(model, require_trainable=trainable)
    with torch.no_grad():
        weight.copy_(saved)
    return metadata


def load_full_embedding_adapter(base_model, checkpoint_dir, *, is_trainable=False, **peft_kwargs):
    """Load the LoRA AND its required complete tied embedding for inference.

    Pass a fresh original Base model loaded with the same backbone dtype as the
    saved recipe (BF16 for this project).  Loading plain PeftModel.from_pretrained
    by itself would silently omit the trained full embedding.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir),
                                     is_trainable=is_trainable, **peft_kwargs)
    load_full_embedding_weights(model, checkpoint_dir, trainable=is_trainable)
    return model


class FullEmbeddingTrainerMixin:
    """Place before transformers.Trainer in the subclass MRO.

    Supports the standard non-sharded single-node DDP recipe.  Trainer's normal
    optimizer/scheduler/RNG checkpoints remain unchanged; these overrides add
    the shared master weight to model saving, resume, and best-checkpoint load.
    """

    def _save(self, output_dir=None, state_dict=None):
        if getattr(self, "is_deepspeed_enabled", False) or getattr(self, "is_fsdp_enabled", False):
            raise ValueError("FullEmbeddingTrainer supports unsharded DDP only")
        super()._save(output_dir=output_dir, state_dict=state_dict)
        save_full_embedding_weights(self.model, output_dir or self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Validate the companion files before modifying adapter parameters.
        target = self.model if model is None else model
        load_full_embedding_weights(target, resume_from_checkpoint, trainable=True)
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        assert_full_embedding_tied(target, require_trainable=True)
        return result

    def _load_best_model(self):
        result = super()._load_best_model()
        load_full_embedding_weights(self.model, self.state.best_model_checkpoint, trainable=True)
        return result

    def _load_rng_state(self, checkpoint):
        # This installed Transformers 4.51.3 calls weights_only=True even under
        # torch 2.5.1, but its built-in NumPy allowlist is enabled only for 2.6+.
        # Keep restricted loading; scope the exact RNG-state NumPy types instead
        # of switching torch.load to unrestricted pickle deserialization.
        import numpy as np

        from importlib import import_module

        core_name = "numpy._core" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core"
        multiarray = import_module(core_name + ".multiarray")
        allowed = [multiarray._reconstruct, np.ndarray, np.dtype,
                   type(np.dtype(np.uint32))]
        with torch.serialization.safe_globals(allowed):
            return super()._load_rng_state(checkpoint)


# Imported here so low-level utilities above are straightforward to reuse in
# inference scripts; the public Trainer subclass matches the existing recipe.
from transformers import Trainer


class FullEmbeddingTrainer(FullEmbeddingTrainerMixin, Trainer):
    pass
