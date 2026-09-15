"""Tensor grouping, parameter counting, and regex filtering."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Pattern, Tuple, Union
import torch
import torch.nn as nn


# Default patterns for weights that should remain in high precision (FP16/Q8_0)
DEFAULT_EXCLUDE_PATTERNS = [
    r"embed_tokens",
    r"wte",
    r"lm_head",
    r"norm",
    r"layernorm",
]

# Default patterns for standard transformer linear weights to quantize
DEFAULT_QUANT_PATTERNS = [
    r"self_attn\.(q|k|v|o)_proj",
    r"mlp\.(gate|up|down)_proj",
    r"attention\.(query|key|value|dense)",
]


def count_parameters(tensor: Union[torch.Tensor, Tuple[int, ...]]) -> int:
    """Return number of parameters for a tensor or shape tuple."""
    if isinstance(tensor, torch.Tensor):
        return tensor.numel()
    prod = 1
    for dim in tensor:
        prod *= dim
    return prod


def should_quantize_tensor(
    name: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> bool:
    """Determine whether a tensor should undergo low-bit mixed precision quantization."""
    excludes = exclude_patterns if exclude_patterns is not None else DEFAULT_EXCLUDE_PATTERNS
    for pattern in excludes:
        if re.search(pattern, name, re.IGNORECASE):
            return False

    if include_patterns:
        return any(re.search(pattern, name, re.IGNORECASE) for pattern in include_patterns)

    # By default, quantize 2D weight matrices that end with .weight
    return name.endswith(".weight")


def get_tensor_groups(
    named_tensors: Dict[str, torch.Tensor],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[str], torch.Tensor]:
    """Extract quantizable tensor names and their parameter weights.

    Returns:
        group_names: list of N tensor names
        weights: (N,) tensor containing parameter counts
    """
    group_names: List[str] = []
    param_counts: List[int] = []

    for name, tensor in named_tensors.items():
        if tensor.ndim >= 2 and should_quantize_tensor(name, include_patterns, exclude_patterns):
            group_names.append(name)
            param_counts.append(tensor.numel())

    weights_tensor = torch.tensor(param_counts, dtype=torch.float64)
    return group_names, weights_tensor
