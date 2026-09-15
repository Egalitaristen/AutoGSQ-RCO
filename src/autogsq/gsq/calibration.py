"""Calibration activations and Hessian capture for paper-quality GSQ.

Paper-quality GSQ needs two things our pipeline previously lacked:

1. Real calibration activations flowing through the *original* model
   (``calibration_data`` was accepted but never used).
2. Per-linear Hessians ``H = X^T X`` (and their damped inverses) built from
   those activations, which drive both the GPTQ warm-start and the
   activation-space reconstruction loss
   ``||X w~-X w||^2 = (w~-w)^T H (w~-w)``.

The ``H``-form loss is exact and needs no stored activations: only the
``(in_features, in_features)`` matrix per linear is kept. Collection is
streamed one layer at a time (hooks live only on the current layer's
linears) so multi-GB models never blow up RAM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


def prepare_calibration_batches(
    tokenizer: Any,
    dataset_name: str = "wikitext2",
    nsamples: int = 128,
    max_length: int = 2048,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Token-id batches for calibration (real datasets, synthetic fallback)."""
    from autogsq.common.models import get_calibration_batches

    return get_calibration_batches(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        num_samples=nsamples,
        max_length=max_length,
        device=device,
    )


def collect_layer_hessians(
    model: nn.Module,
    layer: nn.Module,
    layer_name: str,
    calib_batches: List[torch.Tensor],
    device: torch.device,
    damping: float = 0.01,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run calibration batches and return ``{tensor_name: {"H", "Hinv"}}``.

    Args:
        model: full original model (run under no_grad; never modified).
        layer: the decoder block whose linears get hooked.
        layer_name: prefix, e.g. ``"model.layers.0"``.
        calib_batches: token-id batches ``(1, seq)`` on ``device``.
        device: where Hessian accumulation happens.
        damping: relative damp ``lambda * mean(diag(H))`` for inversion.
        dtype: accumulation dtype (float32 for numerical safety).

    Returns:
        Mapping ``f"{layer_name}.{module}.{weight}"`` -> ``{"H", "Hinv"}``.
    """
    from autogsq.gsq.gptq_init import GPTQAccumulator

    linears: List[Tuple[str, nn.Linear]] = [
        (n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)
    ]
    if not linears:
        return {}

    accums: Dict[str, GPTQAccumulator] = {}
    for mod_name, mod in linears:
        in_f = mod.in_features
        accums[mod_name] = GPTQAccumulator(in_f, device, dtype)

    handles = []

    def _make_hook(acc: GPTQAccumulator):
        def _hook(_mod: nn.Module, inputs: Tuple[torch.Tensor, ...], _out: Any) -> None:
            x = inputs[0].detach()
            # Linear inputs are (..., in_features); fold leading dims.
            acc.add_batch(x.reshape(-1, acc.in_features).to(device))
        return _hook

    for mod_name, mod in linears:
        handles.append(mod.register_forward_hook(_make_hook(accums[mod_name])))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in calib_batches:
                model(batch.to(device))
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train(True)

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for mod_name, acc in accums.items():
        if acc.nsamples == 0:
            continue
        H = acc.H.clone()
        H.mul_(2.0 / max(acc.nsamples, 1))
        out[f"{layer_name}.{mod_name}.weight"] = {
            "H": H.clone(),
            "Hinv": acc.get_inverse_hessian(damping=damping),
        }
    return out
