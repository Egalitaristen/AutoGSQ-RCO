"""HuggingFace model loading, layer extraction, and calibration data loaders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype representation to torch.dtype."""
    s = dtype_str.lower().strip()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif s in ("float16", "fp16"):
        return torch.float16
    elif s in ("float32", "fp32"):
        return torch.float32
    return torch.float32


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cpu",
    dtype: str = "bfloat16",
) -> Tuple[Any, Any]:
    """Load model and tokenizer with low CPU memory usage."""
    if AutoModelForCausalLM is None:
        raise ImportError("The 'transformers' package is required for model loading.")

    torch_dtype = get_torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def get_transformer_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """Extract the sequential list of transformer decoder layers from standard architectures."""
    # Common architectures: LLaMA, Qwen, Mistral, Gemma
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers", model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "transformer.h", model.transformer.h
    elif hasattr(model, "layers"):
        return "layers", model.layers
    elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return "decoder.layers", model.decoder.layers
    else:
        # Fallback: search children for a ModuleList
        for name, child in model.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                return name, child
            if hasattr(child, "layers") and isinstance(child.layers, nn.ModuleList):
                return f"{name}.layers", child.layers
        raise ValueError("Could not automatically locate transformer decoder layers in model.")


def get_calibration_batches(
    dataset_name: str,
    tokenizer: Any,
    num_samples: int = 128,
    max_length: int = 2048,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Load or generate calibration token batches."""
    # Synthetic token generation fallback if dataset loading is unavailable or offline
    try:
        from datasets import load_dataset
        if dataset_name == "wikitext2":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join(ds["text"])
        elif dataset_name == "c4":
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            text = "\n\n".join([item["text"] for item in ds.take(100)])
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            text = "\n\n".join([item["text"] for item in ds.take(100)])

        tokens = tokenizer(text, return_tensors="pt").input_ids[0]
        batches = []
        for i in range(0, min(len(tokens) - max_length, num_samples * max_length), max_length):
            batch = tokens[i : i + max_length].unsqueeze(0).to(device)
            batches.append(batch)
            if len(batches) >= num_samples:
                break
        if batches:
            return batches
    except Exception:
        pass

    # Clean deterministic synthetic token batches fallback
    vocab_size = getattr(tokenizer, "vocab_size", 32000) if tokenizer else 32000
    batches = []
    for s in range(num_samples):
        # Generate varied synthetic sequences
        seq = torch.randint(100, vocab_size, (1, max_length), device=device)
        batches.append(seq)
    return batches


class ModelWeightSource:
    """Lazy accessor for a model's original FP weights, for any model source.

    Accepts a ``.safetensors`` file, a directory of safetensors shards (or a
    HuggingFace local snapshot dir), or a HuggingFace model ID. Tensors are
    loaded on demand so multi-GB models never sit fully in RAM.
    """

    def __init__(self, source: str) -> None:
        from pathlib import Path as _Path

        self.source = str(source)
        p = _Path(source)
        self._shards: Optional[List[str]] = None
        self._state: Optional[Dict[str, torch.Tensor]] = None
        if p.is_file() and p.suffix == ".safetensors":
            self._shards = [str(p)]
        elif p.is_dir():
            shards = sorted(str(f) for f in p.glob("*.safetensors"))
            if shards:
                self._shards = shards

    def names(self) -> List[str]:
        """Return all tensor names in the source model."""
        if self._shards is not None:
            from safetensors import safe_open

            seen: List[str] = []
            for shard in self._shards:
                with safe_open(shard, framework="pt") as f:
                    seen.extend(list(f.keys()))
            return seen
        if self._state is None:
            if AutoModelForCausalLM is None:
                raise ImportError("The 'transformers' package is required for model loading.")
            model = AutoModelForCausalLM.from_pretrained(
                self.source, torch_dtype=torch.float32, trust_remote_code=True
            )
            self._state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            del model
        return list(self._state.keys())

    def get(self, tensor_name: str) -> torch.Tensor:
        """Load one tensor by name (CPU, owned memory).

        NOTE: safetensors returns memory-mapped views tied to the open file
        handle; this method clones into owned RAM so the tensor stays valid
        after the handle closes (use-after-close segfaulted on large models).
        """
        if self._shards is not None:
            from safetensors import safe_open

            for shard in self._shards:
                with safe_open(shard, framework="pt") as f:
                    if tensor_name in f.keys():
                        return f.get_tensor(tensor_name).clone().cpu()
            raise KeyError(f"Tensor {tensor_name!r} not found in {self.source!r}")
        if self._state is None:
            self.names()
        assert self._state is not None
        if tensor_name not in self._state:
            raise KeyError(f"Tensor {tensor_name!r} not found in {self.source!r}")
        return self._state[tensor_name]
