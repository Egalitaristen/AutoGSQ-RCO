"""Surrogate and end-to-end loss functions for RCO allocation."""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_reconstruction_loss(
    weight_candidates: List[torch.Tensor],
    activation_inputs: torch.Tensor,
    original_output: torch.Tensor,
    alpha_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute soft expected layer reconstruction MSE under option probabilities alpha_probs.

    Args:
        weight_candidates: list of K candidate weight tensors, each (out_features, in_features)
        activation_inputs: (batch_size, in_features)
        original_output: (batch_size, out_features)
        alpha_probs: (K,) probability distribution over K options

    Returns:
        Scalar reconstruction loss
    """
    total_loss = torch.tensor(0.0, device=original_output.device, dtype=original_output.dtype)
    for k, w_k in enumerate(weight_candidates):
        pred_output = F.linear(activation_inputs, w_k)
        mse = F.mse_loss(pred_output, original_output)
        total_loss = total_loss + alpha_probs[k] * mse
    return total_loss


def cross_entropy_eval(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard cross entropy loss for perplexity evaluation."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss
