"""Unit tests for GSQ quantizers (2-bit, integer, and factorized ternary)."""

import pytest
import torch
import torch.nn.functional as F

from autogsq.gsq.quantizers import (
    GumbelQuantizer2Bit,
    GumbelQuantizerInt,
    GumbelQuantizerTernary,
    expand_group_param,
)
from autogsq.gsq.schedules import get_temperature, get_logit_scale


def test_expand_group_param():
    """Test group parameter expansion across column dimension."""
    param = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 rows, 2 groups
    group_size = 4
    in_features = 7  # Not a multiple of group_size
    expanded = expand_group_param(param, in_features, group_size)
    assert expanded.shape == (2, 7)
    assert torch.equal(expanded[0], torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))


def test_gsq_2bit_quantizer():
    """Test GumbelQuantizer2Bit codebook, forward pass, and gradient flow."""
    torch.manual_seed(42)
    weight = torch.randn(16, 64, dtype=torch.float32)
    quantizer = GumbelQuantizer2Bit(weight, group_size=32)

    # Check shapes
    assert quantizer.logits.shape == (16, 64, 4)
    assert quantizer.scale.shape == (16, 2)

    # Forward pass produces tensor of same shape
    soft_w = quantizer.forward(temperature=1.0, logit_scale=100.0)
    assert soft_w.shape == weight.shape
    assert soft_w.dtype == weight.dtype

    # Gradient flow test
    loss = F.mse_loss(soft_w, weight)
    loss.backward()
    assert quantizer.logits.grad is not None
    assert torch.isfinite(quantizer.logits.grad).all()
    assert quantizer.scale.grad is not None
    assert torch.isfinite(quantizer.scale.grad).all()

    # Harden test
    res = quantizer.harden()
    assert "dequant_weight" in res
    assert "qweight" in res
    assert res["bits"] == 2
    assert res["qweight"].shape == weight.shape
    assert res["qweight"].dtype == torch.uint8
    # 2-bit codes must be in [0, 3]
    assert (res["qweight"] >= 0).all() and (res["qweight"] <= 3).all()


def test_gsq_int_quantizer():
    """Test GumbelQuantizerInt (e.g. 4-bit) 5-level offset quantization."""
    torch.manual_seed(42)
    weight = torch.randn(8, 128, dtype=torch.float32)
    quantizer = GumbelQuantizerInt(weight, group_size=32, n_bits=4)

    assert quantizer.logits.shape == (8, 128, 5)
    assert quantizer.scale.shape == (8, 4)

    soft_w = quantizer.forward(temperature=1.0, logit_scale=100.0)
    assert soft_w.shape == weight.shape

    loss = F.mse_loss(soft_w, weight)
    loss.backward()
    assert quantizer.logits.grad is not None
    assert torch.isfinite(quantizer.logits.grad).all()

    res = quantizer.harden()
    assert res["bits"] == 4
    assert (res["qweight"] >= 0).all() and (res["qweight"] <= 15).all()


def test_gsq_ternary_quantizer():
    """Test GumbelQuantizerTernary factorized mask and sign logits."""
    torch.manual_seed(42)
    weight = torch.randn(8, 64, dtype=torch.float32)
    quantizer = GumbelQuantizerTernary(weight, group_size=32)

    assert quantizer.mask_logits.shape == (8, 64)
    assert quantizer.sign_logits.shape == (8, 64)

    soft_w = quantizer.forward(temperature=1.0, logit_scale=100.0)
    assert soft_w.shape == weight.shape

    loss = F.mse_loss(soft_w, weight)
    loss.backward()
    assert quantizer.mask_logits.grad is not None
    assert quantizer.sign_logits.grad is not None
    assert torch.isfinite(quantizer.mask_logits.grad).all()
    assert torch.isfinite(quantizer.sign_logits.grad).all()

    res = quantizer.harden()
    assert res["qweight"].shape == weight.shape
    # Ternary uint8 codes: {0, 1, 2}
    assert (res["qweight"] >= 0).all() and (res["qweight"] <= 2).all()


def test_annealing_collapse():
    """Test that at low temperature and high scale, sampling collapses to deterministic argmax."""
    torch.manual_seed(42)
    weight = torch.randn(8, 32, dtype=torch.float32)
    quantizer = GumbelQuantizer2Bit(weight, group_size=32)

    # Set extreme logits on first element
    with torch.no_grad():
        quantizer.logits[0, 0] = torch.tensor([10.0, -10.0, -10.0, -10.0])

    # Sample multiple times at tau = 0.01, kappa = 500
    samples = []
    for _ in range(5):
        w_sample = quantizer.forward(temperature=0.01, logit_scale=500.0)
        samples.append(w_sample[0, 0].item())

    # All samples should be nearly identical (deterministic collapse)
    for s in samples[1:]:
        assert abs(s - samples[0]) < 1e-4
