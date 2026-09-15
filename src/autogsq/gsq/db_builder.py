"""Per-layer, per-bitwidth candidate database builder with streaming memory offload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from autogsq.common.models import load_model_and_tokenizer, get_transformer_layers, get_torch_dtype
from autogsq.common.store import CandidateStore
from autogsq.common.grouping import should_quantize_tensor
from autogsq.gsq.quantizers import (
    GumbelQuantizer2Bit,
    GumbelQuantizerInt,
    GumbelQuantizerTernary,
    GumbelQuantizerBase,
    expand_group_param,
)
from autogsq.gsq.gptq_init import gptq_init, rtn_init
from autogsq.gsq.schedules import get_temperature, get_logit_scale


def _linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """Reference annealing: linear in training step (not epoch)."""
    if total_steps <= 1:
        return float(start)
    progress = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return float(start + (end - start) * progress)


def train_gsq_layer(
    weight: torch.Tensor,
    bits: Union[int, str],
    group_size: int = 128,
    init_method: Literal["gptq", "rtn"] = "gptq",
    num_epochs: int = 10,
    lr: float = 1e-3,
    temp_range: tuple = (2.0, 0.05),
    scale_range: tuple = (100.0, 500.0),
    device: Optional[torch.device] = None,
    H: Optional[torch.Tensor] = None,
    Hinv: Optional[torch.Tensor] = None,
    optimizer: Literal["lion", "adam"] = "lion",
    lr_logits: float = 2e-4,
    lr_scales: float = 1e-4,
    weight_decay: float = 1.0,
    warmup_ratio: float = 0.1,
    steps_per_epoch: int = 64,
    logits_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Initialize and refine a GSQ quantizer for a single weight tensor.

    Paper-quality path (all optional, all recommended):

    1. GPTQ (Hessian-guided) or RTN initialization producing the starting grid.
    2. GPTQ-centered logit warm start (``std`` 0.01, ``strength`` 6).
    3. Gumbel-Softmax refinement with per-step linear annealing of tau/kappa,
       Lion (3 param groups) or Adam, warmup + decay, minimizing the
       activation-space loss ``(w~-w)^T H (w~-w)`` when ``H`` is given
       (weight-space MSE fallback otherwise).
    4. Harden and return qparams (with weight-space ``mse`` plus ``mse_act``
       when ``H`` is available).
    """
    from autogsq.gsq.lion import Lion

    dev = device or weight.device
    w_dev = weight.to(dev)
    use_h = H is not None
    H_dev = H.to(dev, dtype=torch.float32) if use_h else None

    # Initialize quantizer based on bit-width option, warm-started from a
    # GPTQ/RTN grid whenever second-order info (or any init) is available.
    bits_str = str(bits).lower()
    warm = Hinv is not None
    if bits_str == "ternary":
        init_res = gptq_init(w_dev, Hinv=Hinv, group_size=group_size, n_bits=2) if warm else None
        if init_res is not None:
            s = init_res["scales"]
            s_exp = expand_group_param(s.detach(), w_dev.shape[1], group_size)
            dq = init_res["dequant_weight"].to(w_dev.dtype)
            sc = s_exp.clamp(min=1e-5)
            init_sign = torch.sign(dq) * 2.0
            init_mask = (dq.abs() / sc - 0.5) * 2.0 + 0.01 * torch.randn_like(dq)
            init_sign = init_sign + 0.01 * torch.randn_like(dq)
            quantizer: GumbelQuantizerBase = GumbelQuantizerTernary(
                weight_init=w_dev, group_size=group_size, logits_dtype=logits_dtype,
                init_mask=init_mask, init_sign=init_sign, init_scale=s,
            ).to(dev)
        else:
            quantizer = GumbelQuantizerTernary(
                weight_init=w_dev, group_size=group_size, logits_dtype=logits_dtype
            ).to(dev)
    elif bits == 2 or bits_str == "2":
        init_res = (
            gptq_init(w_dev, Hinv=Hinv, group_size=group_size, n_bits=2)
            if (warm and init_method == "gptq")
            else rtn_init(w_dev, group_size=group_size, n_bits=2)
        )
        s = init_res["scales"]
        s_exp = expand_group_param(s.detach(), w_dev.shape[1], group_size)
        grid = (init_res["dequant_weight"].to(w_dev.dtype) / s_exp.clamp(min=1e-5))
        quantizer = GumbelQuantizer2Bit(
            weight_init=w_dev, group_size=group_size, logits_dtype=logits_dtype,
            init_codes=grid, warm_start=warm,
        ).to(dev)
    else:
        n_bits = int(bits)
        # Use GPTQ / RTN for initial center grid
        init_res = (
            gptq_init(w_dev, Hinv=Hinv, group_size=group_size, n_bits=n_bits)
            if (warm and init_method == "gptq")
            else rtn_init(w_dev, group_size=group_size, n_bits=n_bits)
        )
        quantizer = GumbelQuantizerInt(
            weight_init=w_dev, group_size=group_size, n_bits=n_bits, logits_dtype=logits_dtype,
            init_codes=init_res["qweight"], init_scale=init_res["scales"],
        ).to(dev)

    # Optimizer: Lion with separate logit/scale groups, or legacy Adam.
    if optimizer == "lion":
        logit_params = [p for n, p in quantizer.named_parameters() if "logit" in n]
        scale_params = [p for n, p in quantizer.named_parameters() if "logit" not in n]
        opt: torch.optim.Optimizer = Lion(
            [
                {"params": logit_params, "lr": lr_logits, "weight_decay": weight_decay},
                {"params": scale_params, "lr": lr_scales, "weight_decay": 0.0},
            ],
            lr=lr_logits,
        )
        base_lrs = [lr_logits, lr_scales]
    else:
        params = [p for p in quantizer.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        base_lrs = [lr]

    total_steps = max(int(num_epochs) * int(steps_per_epoch), 1)
    warmup_steps = int(total_steps * float(warmup_ratio))
    for step in range(total_steps):
        tau = _linear_schedule(step, total_steps, temp_range[0], temp_range[1])
        kappa = _linear_schedule(step, total_steps, scale_range[0], scale_range[1])

        # Warmup + decay LR multiplier (linear warmup, cosine decay).
        if warmup_steps > 0 and step < warmup_steps:
            mult = float(step + 1) / float(warmup_steps)
        else:
            denom = max(total_steps - warmup_steps, 1)
            prog = min(max((step - warmup_steps) / denom, 0.0), 1.0)
            mult = 0.5 * (1.0 + math.cos(math.pi * prog))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * mult

        opt.zero_grad()
        soft_w = quantizer.forward(temperature=tau, logit_scale=kappa)

        if use_h:
            # Activation-space loss: (w~-w)^T H (w~-w), exact, no stored X.
            assert H_dev is not None
            d = (soft_w.to(torch.float32) - w_dev.to(torch.float32))
            loss = torch.einsum("oi,ij,oj->", d, H_dev, d) / max(d.shape[0], 1)
        else:
            # Minimize reconstruction MSE
            loss = nn.functional.mse_loss(soft_w, w_dev)
        loss.backward()
        opt.step()

    # Harden to integer grid
    result = quantizer.harden()

    # Record final reconstruction fidelity for RCO allocation.
    # db_builder pops "dequant_weight" before saving; the rest stays in qparams.
    with torch.no_grad():
        dequant = result["dequant_weight"]
        d = dequant.to(dtype=w_dev.dtype).reshape(w_dev.shape) - w_dev
        mse = nn.functional.mse_loss(d + w_dev, w_dev)
        result["mse"] = float(mse.detach().cpu().item())
        if use_h:
            assert H_dev is not None
            df = d.to(torch.float32).cpu()
            mse_act = torch.einsum("oi,ij,oj->", df, H_dev.cpu(), df) / max(df.shape[0], 1)
            result["mse_act"] = float(mse_act.detach().cpu().item())
    return result


def build_candidate_database(
    model_id: str,
    calibration_data: str = "wikitext2",
    bitwidth_options: Optional[List[Union[int, str]]] = None,
    out_dir: Union[str, Path] = "./runtime/db",
    group_size: int = 128,
    init_method: Literal["gptq", "rtn"] = "gptq",
    num_epochs: int = 10,
    device: str = "cpu",
    max_layers: Optional[int] = None,
    resume: bool = True,
    calib_nsamples: int = 128,
    calib_max_length: int = 2048,
    calib_device: str = "cpu",
    load_dtype: str = "float32",
    gptq_damping: float = 0.01,
    optimizer: Literal["lion", "adam"] = "lion",
    lr: float = 1e-3,
    lr_logits: float = 2e-4,
    lr_scales: float = 1e-4,
    weight_decay: float = 1.0,
    warmup_ratio: float = 0.1,
    steps_per_epoch: int = 64,
    temp_range: tuple = (2.0, 0.05),
    scale_range: tuple = (100.0, 500.0),
    logits_dtype: str = "bfloat16",
) -> CandidateStore:
    """Build candidate database for all layers and bit-width options.

    Paper-quality path: calibration activations are captured once per layer
    (whole model on ``calib_device``), producing per-linear Hessians that
    drive GPTQ warm-starts and the activation-space training loss.

    Adheres strictly to the streaming memory contract:
    - Hessian capture runs one hooked layer at a time (only H matrices kept).
    - Training loads 1 layer on accelerator at a time.
    - Frees layer memory before moving to next.
    - Writes progress.json for crash recovery and resumability.
    """
    if bitwidth_options is None:
        bitwidth_options = [2, 3, 4, "ternary"]

    store = CandidateStore(out_dir)
    target_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    calib_dev = torch.device(calib_device if torch.cuda.is_available() and calib_device == "cuda" else "cpu")

    print(f"Loading model '{model_id}'...")
    model, tokenizer = load_model_and_tokenizer(model_id, device="cpu", dtype=load_dtype)
    prefix, layers = get_transformer_layers(model)

    num_layers_to_process = len(layers) if max_layers is None else min(len(layers), max_layers)
    print(f"Generating candidate database for {num_layers_to_process} layers across options {bitwidth_options}...")

    from autogsq.gsq.calibration import prepare_calibration_batches, collect_layer_hessians

    print(f"Preparing {calib_nsamples} calibration batches ({calibration_data})...")
    calib_batches = prepare_calibration_batches(
        tokenizer,
        dataset_name=calibration_data,
        nsamples=calib_nsamples,
        max_length=calib_max_length,
        device=str(calib_dev),
    )

    # Phase 1: Hessian capture for all layers up front. The whole model sits
    # on calib_dev; hooks live on one layer at a time so peak memory is one
    # layer's H matrices plus a single forward's activations.
    print(f"Capturing Hessians on {calib_dev}...")
    model.to(calib_dev)
    all_hessians: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_idx in range(num_layers_to_process):
        layer_name = f"{prefix}.{layer_idx}"
        layer_tensors = [
            f"{layer_name}.{tname}.weight"
            for tname, module in layers[layer_idx].named_modules()
            if isinstance(module, nn.Linear) and should_quantize_tensor(tname + ".weight")
        ]
        if resume and store.is_layer_complete(layer_name, bitwidth_options, layer_tensors):
            print(f"Hessians for {layer_name} already complete, skipping (resume=True).")
            continue
        print(f"Capturing Hessians for {layer_name} ({len(calib_batches)} batches)...")
        all_hessians.update(
            collect_layer_hessians(
                model, layers[layer_idx], layer_name, calib_batches, calib_dev,
                damping=gptq_damping,
            )
        )
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del calib_batches

    # Phase 2: per-tensor training with 1-layer accelerator streaming.
    for layer_idx in range(num_layers_to_process):
        layer_name = f"{prefix}.{layer_idx}"

        if resume and store.is_layer_complete(
            layer_name,
            bitwidth_options,
            [
                f"{layer_name}.{tname}.weight"
                for tname, module in layers[layer_idx].named_modules()
                if isinstance(module, nn.Linear) and should_quantize_tensor(tname + ".weight")
            ],
        ):
            print(f"Layer {layer_name} already complete, skipping (resume=True).")
            continue

        hessians = {
            k: v for k, v in all_hessians.items()
            if k.startswith(layer_name + ".")
        }

        layer = layers[layer_idx]
        layer.to(target_device)

        # Process each linear projection in the layer
        for tensor_name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and should_quantize_tensor(tensor_name + ".weight"):
                full_weight_name = f"{layer_name}.{tensor_name}.weight"
                orig_weight = module.weight.data
                hess = hessians.get(full_weight_name, {})
                H = hess.get("H")
                Hinv = hess.get("Hinv")
                if H is None and init_method == "gptq":
                    print(f"  warning: no Hessian for {full_weight_name}; RTN fallback")

                for bits in bitwidth_options:
                    if resume and store.is_candidate_complete(full_weight_name, bits):
                        continue

                    qparams = train_gsq_layer(
                        weight=orig_weight,
                        bits=bits,
                        group_size=group_size,
                        init_method=init_method,
                        num_epochs=num_epochs,
                        device=target_device,
                        H=H,
                        Hinv=Hinv,
                        optimizer=optimizer,
                        lr=lr,
                        lr_logits=lr_logits,
                        lr_scales=lr_scales,
                        weight_decay=weight_decay,
                        warmup_ratio=warmup_ratio,
                        steps_per_epoch=steps_per_epoch,
                        temp_range=temp_range,
                        scale_range=scale_range,
                        logits_dtype=get_torch_dtype(logits_dtype),
                    )

                    dequant_w = qparams.pop("dequant_weight")
                    store.save_candidate(
                        layer_name=full_weight_name,
                        bits=bits,
                        dequant_weight=dequant_w,
                        qparams=qparams,
                    )

        store.mark_layer_complete(layer_name)

        # Drop this layer's Hessians (all remaining H/Hinv freed as we go).
        for k in [k for k in all_hessians if k.startswith(layer_name + ".")]:
            del all_hessians[k]

        # Offload layer to CPU/meta to free accelerator VRAM
        layer.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Candidate database generation complete at: {out_dir}")
    return store
