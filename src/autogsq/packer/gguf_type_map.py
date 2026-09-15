"""Mapping between RCO-allocated bit-widths and llama.cpp GGUF quant types."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union


# Canonical GGUF intent for our candidate labels. The manifest records intent;
# the writer ladder (closest_writable_type) maps it to what can actually be
# emitted today, truthfully. As packers land, files become byte-exact with
# no manifest changes (K-family and IQ2_XXS native; IQ1_S still falls back).
LABEL_DEFAULTS: Dict[str, str] = {
    "2": "IQ2_XXS",
    "3": "Q3_K_M",
    "4": "Q4_K_M",
    "ternary": "IQ1_S",
}


# Standard GGUF quant types with typical effective bits-per-weight (bpw) and block size
GGUF_TYPE_SPECS: Dict[str, Dict[str, Union[float, int, str]]] = {
    "IQ1_S": {"bpw": 1.56, "block_size": 256, "family": "iq"},
    "IQ1_M": {"bpw": 1.75, "block_size": 256, "family": "iq"},
    "IQ2_XXS": {"bpw": 2.06, "block_size": 256, "family": "iq"},
    "IQ2_XS": {"bpw": 2.31, "block_size": 256, "family": "iq"},
    "IQ2_S": {"bpw": 2.50, "block_size": 256, "family": "iq"},
    "IQ2_M": {"bpw": 2.70, "block_size": 256, "family": "iq"},
    "Q2_K": {"bpw": 2.56, "block_size": 256, "family": "k_quant"},
    "IQ3_XXS": {"bpw": 3.06, "block_size": 256, "family": "iq"},
    "IQ3_S": {"bpw": 3.44, "block_size": 256, "family": "iq"},
    "IQ3_M": {"bpw": 3.66, "block_size": 256, "family": "iq"},
    "Q3_K_S": {"bpw": 3.44, "block_size": 256, "family": "k_quant"},
    "Q3_K_M": {"bpw": 3.91, "block_size": 256, "family": "k_quant"},
    "Q3_K_L": {"bpw": 4.25, "block_size": 256, "family": "k_quant"},
    "Q4_0": {"bpw": 4.50, "block_size": 32, "family": "legacy"},
    "Q4_1": {"bpw": 5.00, "block_size": 32, "family": "legacy"},
    "Q4_K_S": {"bpw": 4.50, "block_size": 256, "family": "k_quant"},
    "Q4_K_M": {"bpw": 4.80, "block_size": 256, "family": "k_quant"},
    "IQ4_XS": {"bpw": 4.25, "block_size": 256, "family": "iq"},
    "IQ4_NL": {"bpw": 4.50, "block_size": 32, "family": "iq"},
    "Q5_0": {"bpw": 5.50, "block_size": 32, "family": "legacy"},
    "Q5_1": {"bpw": 6.00, "block_size": 32, "family": "legacy"},
    "Q5_K_S": {"bpw": 5.50, "block_size": 256, "family": "k_quant"},
    "Q5_K_M": {"bpw": 5.50, "block_size": 256, "family": "k_quant"},
    "Q6_K": {"bpw": 6.56, "block_size": 256, "family": "k_quant"},
    "Q8_0": {"bpw": 8.50, "block_size": 32, "family": "legacy"},
    "F16": {"bpw": 16.00, "block_size": 1, "family": "float"},
    "BF16": {"bpw": 16.00, "block_size": 1, "family": "float"},
}


def get_available_gguf_types() -> List[str]:
    """Return sorted list of all supported GGUF quant type names."""
    return list(GGUF_TYPE_SPECS.keys())


def nearest_gguf_type(
    bits: Union[float, int, str],
    group_size: int = 128,
    symmetric: bool = True,
    tolerance: float = 0.15,
    fallback: str = "round_down",
) -> str:
    """Map an RCO-selected bit-width to the nearest compatible GGUF quant type.

    Args:
        bits: chosen bit-width (e.g. 2, 2.75, 3, 4, or 'ternary')
        group_size: quantization group size
        symmetric: whether symmetric grid was used
        tolerance: maximum allowed drift between target bits and GGUF type bpw
        fallback: policy if nearest exceeds tolerance ('round_down', 'round_up', 'error')

    Returns:
        String name of the closest GGUF quant type (e.g. 'IQ2_XS', 'Q4_K_M')
    """
    if str(bits).lower() == "ternary":
        target_bpw = 1.6
    else:
        target_bpw = float(bits)

    # Candidate labels map to canonical intent directly; the writer ladder
    # resolves intent to actually-emittable bytes (family-aware, honest).
    if str(bits) in LABEL_DEFAULTS:
        return LABEL_DEFAULTS[str(bits)]

    # Calculate distance to all available GGUF types
    candidates = []
    for name, spec in GGUF_TYPE_SPECS.items():
        bpw = float(spec["bpw"])
        diff = abs(bpw - target_bpw)
        candidates.append((diff, bpw, name))

    candidates.sort(key=lambda x: x[0])
    best_diff, best_bpw, best_name = candidates[0]

    if best_diff <= tolerance:
        return best_name

    # Handle fallback if difference exceeds tolerance
    if fallback == "error":
        raise ValueError(
            f"No GGUF quant type found within tolerance {tolerance} for target {target_bpw} bpw. "
            f"Closest is {best_name} ({best_bpw} bpw, diff={best_diff:.3f})."
        )
    elif fallback == "round_down":
        # Pick closest candidate with bpw <= target_bpw (or minimum available)
        down_candidates = [c for c in candidates if c[1] <= target_bpw]
        if down_candidates:
            return down_candidates[0][2]
        return candidates[0][2]
    elif fallback == "round_up":
        # Pick closest candidate with bpw >= target_bpw (or maximum available)
        up_candidates = [c for c in candidates if c[1] >= target_bpw]
        if up_candidates:
            return up_candidates[0][2]
        return candidates[0][2]
    else:
        raise ValueError(f"Unknown fallback policy: {fallback}")
