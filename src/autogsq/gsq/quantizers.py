"""Gumbel-Softmax Quantizers for GSQ (2-bit, integer, and ternary)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape: torch.Size, device: torch.device, dtype: torch.dtype = torch.float32, eps: float = 1e-20) -> torch.Tensor:
    """Sample from standard Gumbel(0, 1) distribution."""
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def expand_group_param(param: torch.Tensor, in_features: int, group_size: int) -> torch.Tensor:
    """Expand (out_features, n_groups) parameter to (out_features, in_features)."""
    expanded = param.repeat_interleave(group_size, dim=1)
    if expanded.shape[1] > in_features:
        expanded = expanded[:, :in_features]
    return expanded


class GumbelQuantizerBase(nn.Module):
    """Base class for Gumbel-Softmax quantizers."""

    def __init__(
        self,
        weight_init: torch.Tensor,
        group_size: int = 128,
        logits_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight_init.shape
        self.group_size = int(group_size)
        self.n_groups = math.ceil(self.in_features / self.group_size)
        self.logits_dtype = logits_dtype
        self.orig_dtype = weight_init.dtype

        # Register target weight as buffer for reference/loss computation
        self.register_buffer("weight_orig", weight_init.detach().clone())

    def forward(self, temperature: float, logit_scale: float) -> torch.Tensor:
        """Returns the soft-quantized weight tensor w_tilde for this forward pass."""
        raise NotImplementedError

    def harden(self) -> Dict[str, Any]:
        """Collapse logits to argmax and return final integer codes + scales + metadata."""
        raise NotImplementedError


class GumbelQuantizer2Bit(GumbelQuantizerBase):
    """2-bit GSQ quantizer with codebook {-2, -1, 0, 1} * s."""

    CODEBOOK = (-2.0, -1.0, 0.0, 1.0)

    def __init__(
        self,
        weight_init: torch.Tensor,
        group_size: int = 128,
        logits_dtype: torch.dtype = torch.float32,
        init_codes: Optional[torch.Tensor] = None,
        warm_start: bool = False,
    ) -> None:
        super().__init__(weight_init, group_size=group_size, logits_dtype=logits_dtype)

        # Initialize scales per group: mean absolute weight / scale of codebook
        # max abs code is 2.0
        w_reshaped = self._reshape_for_grouping(weight_init)
        init_scale = (w_reshaped.abs().mean(dim=-1) * 2.0).clamp(min=1e-5)
        self.scale = nn.Parameter(init_scale.to(self.orig_dtype))

        # 4 logits per weight element: (out_features, in_features, 4)
        # Initialize logits towards nearest code in codebook
        expanded_scale = expand_group_param(self.scale.detach(), self.in_features, self.group_size)
        grid = init_codes.to(weight_init.dtype) if init_codes is not None else None
        scaled_w = (grid if grid is not None else weight_init) / expanded_scale.clamp(min=1e-5)

        codes_tensor = torch.tensor(self.CODEBOOK, device=weight_init.device, dtype=scaled_w.dtype)
        # Distance to each code: shape (out_features, in_features, 4)
        dist = (scaled_w.unsqueeze(-1) - codes_tensor).abs()
        if warm_start:
            # Reference GPTQ-centered warm start: l = strength*l_GPTQ + std*eps,
            # with l_GPTQ = -0.5 * dist^2 mean-centered over codes.
            base = -0.5 * dist * dist
            base = base - base.mean(dim=-1, keepdim=True)
            init_logits = 6.0 * base + 0.01 * torch.randn_like(base)
        else:
            init_logits = -dist * 2.0
        self.logits = nn.Parameter(init_logits.to(logits_dtype))

        self.register_buffer(
            "codebook_tensor",
            torch.tensor(self.CODEBOOK, dtype=logits_dtype),
            persistent=False,
        )

    def _reshape_for_grouping(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_size = self.n_groups * self.group_size - self.in_features
        if pad_size > 0:
            padded = F.pad(tensor, (0, pad_size))
        else:
            padded = tensor
        return padded.view(self.out_features, self.n_groups, self.group_size)

    def forward(self, temperature: float = 1.0, logit_scale: float = 100.0) -> torch.Tensor:
        """Soft-quantized weight tensor via 4-way Gumbel-Softmax."""
        device = self.logits.device
        dtype = self.logits.dtype

        gumbel_noise = sample_gumbel(self.logits.shape, device, dtype)
        scaled_logits = (self.logits * logit_scale + gumbel_noise) / max(temperature, 1e-5)
        probs = F.softmax(scaled_logits, dim=-1)

        cb = self.codebook_tensor.to(device=device, dtype=dtype)
        soft_grid = torch.sum(probs * cb, dim=-1)

        scale_expanded = expand_group_param(self.scale, self.in_features, self.group_size)
        return (soft_grid * scale_expanded.to(dtype)).to(self.orig_dtype)

    def harden(self) -> Dict[str, Any]:
        """Hard quantization by taking argmax over logits."""
        with torch.no_grad():
            best_idx = torch.argmax(self.logits, dim=-1)  # indices 0, 1, 2, 3
            cb = self.codebook_tensor.to(device=self.logits.device, dtype=self.orig_dtype)
            grid_vals = cb[best_idx]

            scale_expanded = expand_group_param(self.scale, self.in_features, self.group_size)
            dequant = grid_vals * scale_expanded

            # Integer codes: 0, 1, 2, 3 stored as uint8
            qweight = best_idx.to(torch.uint8)
            zeros = torch.zeros_like(self.scale)

            return {
                "dequant_weight": dequant,
                "qweight": qweight,
                "scales": self.scale.detach().clone(),
                "zeros": zeros,
                "perm": None,
                "bits": 2,
                "group_size": self.group_size,
                "sym": True,
                "shape": (self.out_features, self.in_features),
                "dtype": str(self.orig_dtype),
                "schema": 1,
            }


class GumbelQuantizerInt(GumbelQuantizerBase):
    """General b-bit quantizer (b > 2) with 5-level offset around initial integer grid."""

    OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)

    def __init__(
        self,
        weight_init: torch.Tensor,
        group_size: int = 128,
        n_bits: int = 4,
        logits_dtype: torch.dtype = torch.float32,
        init_codes: Optional[torch.Tensor] = None,
        init_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(weight_init, group_size=group_size, logits_dtype=logits_dtype)
        self.n_bits = int(n_bits)
        self.qmax = (1 << self.n_bits) - 1
        self.half_qmax = 1 << (self.n_bits - 1)

        # Scale initialization (GPTQ scales win when provided)
        if init_scale is not None:
            self.scale = nn.Parameter(init_scale.to(self.orig_dtype).detach().clone())
        else:
            w_reshaped = self._reshape_for_grouping(weight_init)
            init_scale = (w_reshaped.abs().amax(dim=-1) / float(self.half_qmax)).clamp(min=1e-5)
            self.scale = nn.Parameter(init_scale.to(self.orig_dtype))

        # Initial integer grid: GPTQ codes when provided, else RTN rounding.
        if init_codes is not None:
            base_int = init_codes.to(weight_init.device).clamp(0, self.qmax)
        else:
            expanded_scale = expand_group_param(self.scale.detach(), self.in_features, self.group_size)
            scaled_w = weight_init / expanded_scale.clamp(min=1e-5)
            # Shift to non-negative range [0, 2^b - 1]
            base_int = torch.round(scaled_w + self.half_qmax).clamp(0, self.qmax)
        self.register_buffer("base_int", base_int, persistent=False)

        # 5 logits per weight element representing offsets {-2, -1, 0, 1, 2}
        init_logits = torch.zeros((self.out_features, self.in_features, 5), dtype=logits_dtype)
        init_logits[..., 2] = 2.0  # bias towards 0 offset initially
        self.logits = nn.Parameter(init_logits)

        self.register_buffer(
            "offset_tensor",
            torch.tensor(self.OFFSETS, dtype=logits_dtype),
            persistent=False,
        )

        # Valid-shift mask: edge grid points cannot select out-of-range shifts.
        with torch.no_grad():
            off = torch.tensor(self.OFFSETS, device=base_int.device).view(1, 1, -1)
            shifted = base_int.unsqueeze(-1).to(torch.float32) + off
            valid = (shifted >= 0) & (shifted <= self.qmax)
        self.register_buffer("valid_mask", valid, persistent=False)

    def _reshape_for_grouping(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_size = self.n_groups * self.group_size - self.in_features
        if pad_size > 0:
            padded = F.pad(tensor, (0, pad_size))
        else:
            padded = tensor
        return padded.view(self.out_features, self.n_groups, self.group_size)

    def forward(self, temperature: float = 1.0, logit_scale: float = 100.0) -> torch.Tensor:
        device = self.logits.device
        dtype = self.logits.dtype

        gumbel_noise = sample_gumbel(self.logits.shape, device, dtype)
        scaled_logits = (self.logits * logit_scale + gumbel_noise) / max(temperature, 1e-5)
        if hasattr(self, "valid_mask") and self.valid_mask is not None:
            scaled_logits = scaled_logits.masked_fill(~self.valid_mask.to(device), -1e9)
        probs = F.softmax(scaled_logits, dim=-1)

        offsets = self.offset_tensor.to(device=device, dtype=dtype)
        # Candidate integer values: base_int.unsqueeze(-1) + offsets
        candidates = (self.base_int.unsqueeze(-1).to(dtype) + offsets).clamp(0, self.qmax)
        # Shift back from unsigned [0, 2^b - 1] to signed symmetric grid
        signed_candidates = candidates - float(self.half_qmax)
        soft_grid = torch.sum(probs * signed_candidates, dim=-1)

        scale_expanded = expand_group_param(self.scale, self.in_features, self.group_size)
        return (soft_grid * scale_expanded.to(dtype)).to(self.orig_dtype)

    def harden(self) -> Dict[str, Any]:
        with torch.no_grad():
            best_idx = torch.argmax(self.logits, dim=-1)
            offsets = self.offset_tensor.to(device=self.logits.device, dtype=self.logits.dtype)
            chosen_offset = offsets[best_idx]
            hard_codes = (self.base_int + chosen_offset).clamp(0, self.qmax)

            signed_grid = hard_codes - float(self.half_qmax)
            scale_expanded = expand_group_param(self.scale, self.in_features, self.group_size)
            dequant = signed_grid.to(self.orig_dtype) * scale_expanded

            zeros = torch.zeros_like(self.scale)
            return {
                "dequant_weight": dequant,
                "qweight": hard_codes.to(torch.uint8),
                "scales": self.scale.detach().clone(),
                "zeros": zeros,
                "perm": None,
                "bits": self.n_bits,
                "group_size": self.group_size,
                "sym": True,
                "shape": (self.out_features, self.in_features),
                "dtype": str(self.orig_dtype),
                "schema": 1,
            }


class GumbelQuantizerTernary(GumbelQuantizerBase):
    """Factorized ternary quantizer with codebook {-1, 0, +1} * s.

    Uses separate mask logit (active vs zero) and sign logit (+/-).
    """

    def __init__(
        self,
        weight_init: torch.Tensor,
        group_size: int = 128,
        logits_dtype: torch.dtype = torch.float32,
        init_mask: Optional[torch.Tensor] = None,
        init_sign: Optional[torch.Tensor] = None,
        init_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(weight_init, group_size=group_size, logits_dtype=logits_dtype)

        # Scale initialization (GPTQ scales win when provided)
        if init_scale is not None:
            self.scale = nn.Parameter(init_scale.to(self.orig_dtype).detach().clone())
        else:
            w_reshaped = self._reshape_for_grouping(weight_init)
            init_scale = w_reshaped.abs().mean(dim=-1).clamp(min=1e-5)
            self.scale = nn.Parameter(init_scale.to(self.orig_dtype))

        # Mask logits: positive indicates active (non-zero), negative indicates zero
        expanded_scale = expand_group_param(self.scale.detach(), self.in_features, self.group_size)
        if init_mask is not None:
            mask_init = init_mask.to(logits_dtype)
        else:
            abs_scaled = weight_init.abs() / expanded_scale.clamp(min=1e-5)
            # Threshold: items with abs > 0.5 get positive mask logits
            mask_init = (abs_scaled - 0.5) * 2.0
        self.mask_logits = nn.Parameter(mask_init.to(logits_dtype))

        # Sign logits: positive indicates +1, negative indicates -1
        if init_sign is not None:
            sign_init = init_sign.to(logits_dtype)
        else:
            sign_init = torch.sign(weight_init) * 2.0
        self.sign_logits = nn.Parameter(sign_init.to(logits_dtype))

    def _reshape_for_grouping(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_size = self.n_groups * self.group_size - self.in_features
        if pad_size > 0:
            padded = F.pad(tensor, (0, pad_size))
        else:
            padded = tensor
        return padded.view(self.out_features, self.n_groups, self.group_size)

    def forward(self, temperature: float = 1.0, logit_scale: float = 100.0) -> torch.Tensor:
        device = self.mask_logits.device
        dtype = self.mask_logits.dtype

        # Binary Gumbel-Sigmoid for mask: logistic noise = g1 - g2.
        # Reference uses 2.0 logit factors: sigmoid((2*l*kappa + noise) / tau).
        g_mask = sample_gumbel(self.mask_logits.shape, device, dtype) - sample_gumbel(
            self.mask_logits.shape, device, dtype
        )
        p_mask = torch.sigmoid((2.0 * self.mask_logits * logit_scale + g_mask) / max(temperature, 1e-5))

        # Binary Gumbel-Sigmoid for sign:
        g_sign = sample_gumbel(self.sign_logits.shape, device, dtype) - sample_gumbel(
            self.sign_logits.shape, device, dtype
        )
        p_sign = torch.sigmoid((2.0 * self.sign_logits * logit_scale + g_sign) / max(temperature, 1e-5))
        # Continuous expectation of sign in [-1, +1]
        soft_sign = 2.0 * p_sign - 1.0

        soft_weight = p_mask * soft_sign
        scale_expanded = expand_group_param(self.scale, self.in_features, self.group_size)
        return (soft_weight * scale_expanded.to(dtype)).to(self.orig_dtype)

    def harden(self) -> Dict[str, Any]:
        with torch.no_grad():
            hard_mask = (self.mask_logits > 0).to(self.orig_dtype)
            hard_sign = torch.where(
                self.sign_logits >= 0,
                torch.ones_like(self.sign_logits),
                -torch.ones_like(self.sign_logits),
            ).to(self.orig_dtype)

            ternary_val = hard_mask * hard_sign  # values in {-1, 0, +1}
            scale_expanded = expand_group_param(self.scale, self.in_features, self.group_size)
            dequant = ternary_val * scale_expanded

            # Store codes mapped to uint8: {-1, 0, 1} -> {0, 1, 2} with zeros = 1
            qweight = (ternary_val + 1.0).to(torch.uint8)
            zeros = torch.ones_like(self.scale)

            return {
                "dequant_weight": dequant,
                "qweight": qweight,
                "scales": self.scale.detach().clone(),
                "zeros": zeros,
                "perm": None,
                "bits": 2,  # 2 bits allocated on storage for ternary
                "group_size": self.group_size,
                "sym": True,
                "shape": (self.out_features, self.in_features),
                "dtype": str(self.orig_dtype),
                "schema": 1,
            }
