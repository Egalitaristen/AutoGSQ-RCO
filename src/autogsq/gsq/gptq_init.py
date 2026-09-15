"""GPTQ (Hessian-based) and RTN (Round-To-Nearest) initialization for GSQ."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def compute_group_scale_zero(
    weight: torch.Tensor,
    group_size: int,
    n_bits: int,
    sym: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group scales and zero-points for a weight tensor.

    Args:
        weight: (out_features, in_features)
        group_size: group size along in_features
        n_bits: target bit-width (2..8)
        sym: symmetric quantization

    Returns:
        scales: (out_features, n_groups)
        zeros: (out_features, n_groups)
    """
    out_features, in_features = weight.shape
    n_groups = math.ceil(in_features / group_size)

    pad_size = n_groups * group_size - in_features
    if pad_size > 0:
        w_pad = nn.functional.pad(weight, (0, pad_size))
    else:
        w_pad = weight

    w_grouped = w_pad.view(out_features, n_groups, group_size)
    qmax = (1 << n_bits) - 1

    if sym:
        # Symmetric quantization: zero is centered
        half_qmax = 1 << (n_bits - 1)
        max_val = w_grouped.abs().amax(dim=-1).clamp(min=1e-5)
        scales = max_val / float(half_qmax)
        zeros = torch.full_like(scales, float(half_qmax))
    else:
        # Asymmetric quantization
        min_val = w_grouped.amin(dim=-1)
        max_val = w_grouped.amax(dim=-1)
        scales = (max_val - min_val).clamp(min=1e-5) / float(qmax)
        zeros = torch.round(-min_val / scales).clamp(0, qmax)

    return scales, zeros


def rtn_init(
    weight: torch.Tensor,
    group_size: int = 128,
    n_bits: int = 4,
    sym: bool = True,
) -> Dict[str, torch.Tensor]:
    """Simple Round-To-Nearest (RTN) initialization."""
    scales, zeros = compute_group_scale_zero(weight, group_size, n_bits, sym=sym)
    out_features, in_features = weight.shape

    # Expand scales and zeros
    expanded_scale = scales.repeat_interleave(group_size, dim=1)[:, :in_features]
    expanded_zero = zeros.repeat_interleave(group_size, dim=1)[:, :in_features]

    qmax = (1 << n_bits) - 1
    qweight = torch.round(weight / expanded_scale + expanded_zero).clamp(0, qmax)
    dequant = (qweight - expanded_zero) * expanded_scale

    return {
        "qweight": qweight.to(torch.uint8),
        "scales": scales,
        "zeros": zeros,
        "dequant_weight": dequant,
        "bits": n_bits,
        "group_size": group_size,
        "sym": sym,
    }


class GPTQAccumulator:
    """Accumulates calibration activations and computes the inverse Hessian for GPTQ."""

    def __init__(self, in_features: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.in_features = in_features
        self.device = device
        self.dtype = dtype
        self.H = torch.zeros((in_features, in_features), device=device, dtype=dtype)
        self.nsamples = 0

    def add_batch(self, x: torch.Tensor) -> None:
        """Add calibration activation batch.

        Args:
            x: activation tensor, (..., in_features)
        """
        x = x.view(-1, self.in_features).to(device=self.device, dtype=self.dtype)
        self.H.addmm_(x.t(), x)
        self.nsamples += x.shape[0]

    def get_inverse_hessian(self, damping: float = 0.01) -> torch.Tensor:
        """Compute (H + lambda * I)^(-1) using Cholesky factorization."""
        H = self.H.clone()
        if self.nsamples > 0:
            H.mul_(2.0 / self.nsamples)

        diag = torch.diag(H)
        mean_diag = torch.mean(diag).item()
        if mean_diag == 0:
            mean_diag = 1.0
        damp = damping * mean_diag
        H.add_(torch.eye(self.in_features, device=self.device, dtype=self.dtype) * damp)

        try:
            chol = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(chol)
        except Exception:
            # Fallback to pseudo-inverse if Cholesky fails due to non-positive definiteness
            Hinv = torch.linalg.pinv(H)

        return Hinv


def gptq_init(
    weight: torch.Tensor,
    Hinv: Optional[torch.Tensor] = None,
    group_size: int = 128,
    n_bits: int = 4,
    sym: bool = True,
    block_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """GPTQ quantization with second-order error compensation.

    If Hinv is None, falls back to RTN.
    """
    if Hinv is None:
        return rtn_init(weight, group_size=group_size, n_bits=n_bits, sym=sym)

    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype

    W = weight.clone().to(torch.float32)
    Hinv = Hinv.to(device=device, dtype=torch.float32)

    # Initial group scales and zeros
    scales, zeros = compute_group_scale_zero(weight, group_size, n_bits, sym=sym)
    qmax = (1 << n_bits) - 1

    Q = torch.zeros_like(W)

    # Block-wise quantization with column updates
    for col_start in range(0, in_features, block_size):
        col_end = min(col_start + block_size, in_features)
        count = col_end - col_start

        W1 = W[:, col_start:col_end].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)

        Hinv_block = Hinv[col_start:col_end, col_start:col_end]
        inv_diag = torch.diag(Hinv_block)

        for j in range(count):
            c = col_start + j
            g_idx = c // group_size
            s = scales[:, g_idx]
            z = zeros[:, g_idx]

            w_col = W1[:, j]
            q_col = torch.round(w_col / s + z).clamp(0, qmax)
            q_dequant = (q_col - z) * s
            Q1[:, j] = q_col

            err = w_col - q_dequant
            Err1[:, j] = err

            diag_val = inv_diag[j].clamp(min=1e-8)
            if j + 1 < count:
                step = Hinv_block[j, j + 1 :] / diag_val
                W1[:, j + 1 :].sub_(err.unsqueeze(1) @ step.unsqueeze(0))

        Q[:, col_start:col_end] = Q1

        # Update remaining weight matrix columns outside the block
        if col_end < in_features:
            Hinv_rem = Hinv[col_start:col_end, col_end:]
            diag_block = inv_diag.clamp(min=1e-8).unsqueeze(1)
            scale_step = Hinv_rem / diag_block
            W[:, col_end:].sub_(Err1 @ scale_step)

    # Final dequantized weight
    expanded_scale = scales.repeat_interleave(group_size, dim=1)[:, :in_features]
    expanded_zero = zeros.repeat_interleave(group_size, dim=1)[:, :in_features]
    dequant = ((Q - expanded_zero) * expanded_scale).to(dtype)

    return {
        "qweight": Q.to(torch.uint8),
        "scales": scales.to(dtype),
        "zeros": zeros.to(dtype),
        "dequant_weight": dequant,
        "bits": n_bits,
        "group_size": group_size,
        "sym": sym,
    }
