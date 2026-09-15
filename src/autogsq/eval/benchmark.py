"""Perplexity evaluation and model benchmarking harness."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from autogsq.common.models import get_calibration_batches
from autogsq.common.store import CandidateStore


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any = None,
    dataset_name: str = "wikitext2",
    max_length: int = 512,
    num_samples: int = 8,
    device: str = "cpu",
    batches: Optional[List[torch.Tensor]] = None,
) -> float:
    """Evaluate cross-entropy perplexity on token sequences.

    PPL = exp(mean(cross_entropy_loss))
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)

    if batches is None:
        batches = get_calibration_batches(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_length=max_length,
            device=str(dev),
        )

    nlls = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in batches:
            input_ids = batch.to(dev)
            outputs = model(input_ids)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            # Shift for autoregressive next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if torch.isfinite(loss):
                nlls.append(loss.item())

    if not nlls:
        return float("nan")

    mean_nll = sum(nlls) / len(nlls)
    # Clamp to avoid overflow
    ppl = math.exp(min(mean_nll, 20.0))
    return ppl


def evaluate_perplexity_strided(
    model: nn.Module,
    tokenizer: Any = None,
    dataset_name: str = "wikitext2",
    split: str = "test",
    max_length: int = 2048,
    stride: int = 512,
    max_windows: int = 0,
    device: str = "cpu",
    text: Optional[str] = None,
) -> float:
    """Strided perplexity over a long token stream (lm-eval style).

    Concatenates the split text, slides a ``max_length`` window by ``stride``,
    and scores each token exactly once (first window fully, then only the
    trailing ``stride`` tokens). This matches literature methodology, unlike
    non-overlapping chunk eval which inflates absolute PPL.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)

    if text is None:
        from datasets import load_dataset

        if dataset_name == "wikitext2":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        elif dataset_name == "c4":
            ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
            ds = [item for _, item in zip(range(200), ds)]
            text = "\n\n".join(
                item["text"] for item in (ds if isinstance(ds, list) else ds.take(200))
            )
            ds = None
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split=split, streaming=True)
            text = "\n\n".join([item["text"] for item in ds.take(200)])
            ds = None
        if ds is not None:
            text = "\n\n".join(ds["text"])

    assert tokenizer is not None and text is not None
    # Cache tokenized test streams (full runs only): retokenizing 300k+
    # tokens and re-resolving the dataset on every variant wastes minutes.
    cache_key = f"{dataset_name}_{split}_{len(text)}"
    input_ids = None
    cache_file = None
    if max_windows == 0:
        try:
            from pathlib import Path as _P

            cache_dir = _P("runtime") / ".tokcache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{cache_key}.pt"
            if cache_file.is_file():
                input_ids = torch.load(str(cache_file), map_location="cpu")
        except Exception:
            input_ids = None
    if input_ids is None:
        input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if cache_file is not None:
            try:
                torch.save(input_ids.cpu(), str(cache_file))
            except Exception:
                pass
    n_tokens = input_ids.size(0)

    nll_sum = 0.0
    n_scored = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    n_windows = 0
    with torch.no_grad():
        for begin in range(0, n_tokens, stride):
            if max_windows and n_windows >= max_windows:
                break
            end = min(begin + max_length, n_tokens)
            if end - begin < 2:
                break
            window = input_ids[begin:end].unsqueeze(0).to(dev)
            outputs = model(window)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = window[..., 1:].contiguous()
            # Score only previously-unscored positions.
            scored_from = 0 if begin == 0 else max_length - stride
            scored_from = min(scored_from, shift_logits.size(1) - 1)
            nll_sum += loss_fct(
                shift_logits[:, scored_from:, :].reshape(-1, shift_logits.size(-1)),
                shift_labels[:, scored_from:].reshape(-1),
            ).item()
            n_scored += shift_logits.size(1) - scored_from
            n_windows += 1
            if end == n_tokens:
                break

    if n_scored == 0:
        return float("nan")
    return math.exp(min(nll_sum / n_scored, 20.0))


def patch_model_weights(model: nn.Module, new_weights: Dict[str, torch.Tensor]) -> None:
    """In-place update named weight parameters of a model."""
    named_params = dict(model.named_parameters())
    for name, tensor in new_weights.items():
        if name in named_params:
            named_params[name].data.copy_(tensor.to(named_params[name].device, named_params[name].dtype))


def run_benchmark(
    model: nn.Module,
    tokenizer: Any,
    candidate_db_dir: Optional[Union[str, Path]] = None,
    allocation_manifest: Optional[Dict[str, Any]] = None,
    dataset_name: str = "wikitext2",
    num_samples: int = 8,
    max_length: int = 512,
    device: str = "cpu",
    console: Optional[Console] = None,
    eval_mode: str = "windowed",
    stride: int = 512,
    split: str = "test",
    max_windows: int = 0,
    skip_uniform: bool = False,
) -> Dict[str, Any]:
    """Benchmark baseline model, uniform baselines, and AutoGSQ-RCO mixed allocation."""
    c = console or Console()
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    strided = eval_mode == "strided"

    # Generate test batches once so all variants are evaluated on identical sequences
    test_batches = None
    if not strided:
        test_batches = get_calibration_batches(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_length=max_length,
            device=str(dev),
        )

    def _ppl() -> float:
        if strided:
            return evaluate_perplexity_strided(
                model, tokenizer, dataset_name=dataset_name, split=split,
                max_length=max_length, stride=stride, max_windows=max_windows,
                device=str(dev),
            )
        return evaluate_perplexity(model, batches=test_batches, device=str(dev))

    results: Dict[str, Any] = {}

    # 1. Baseline FP16/BF16
    c.print("[cyan]Evaluating Baseline (Unquantized)...[/cyan]")
    base_ppl = _ppl()
    results["baseline"] = {
        "name": "Base (Unquantized)",
        "bpw": 16.0,
        "ppl": base_ppl,
        "delta_ppl": 0.0,
        "compression": 1.0,
    }

    # Backup original weights
    orig_weights = {name: param.detach().clone() for name, param in model.named_parameters()}

    store = CandidateStore(candidate_db_dir) if candidate_db_dir else None

    # 2. AutoGSQ-RCO Mixed Precision Allocation
    if store and allocation_manifest and "per_tensor" in allocation_manifest:
        c.print("[magenta]Evaluating AutoGSQ-RCO Mixed-Precision Allocation...[/magenta]")
        per_tensor = allocation_manifest["per_tensor"]
        mixed_weights = {}

        for t_name, info in per_tensor.items():
            bits = info.get("bits", 4)
            try:
                w_dequant = store.load_dequant_weight(t_name, bits, device=dev)
                mixed_weights[t_name] = w_dequant
            except Exception:
                pass

        if mixed_weights:
            patch_model_weights(model, mixed_weights)
            mixed_ppl = _ppl()
            achieved_bpw = float(allocation_manifest.get("achieved_bpw", 2.75))
            results["autogsq_rco"] = {
                "name": "AutoGSQ-RCO (Mixed)",
                "bpw": achieved_bpw,
                "ppl": mixed_ppl,
                "delta_ppl": mixed_ppl - base_ppl,
                "compression": 16.0 / max(achieved_bpw, 1e-4),
            }
            # Restore base weights
            patch_model_weights(model, orig_weights)

    # 3. Uniform 2-bit Baseline (if available in candidate store)
    if store and not skip_uniform:
        uniform_2bit_weights = {}
        for t_name in orig_weights.keys():
            try:
                uniform_2bit_weights[t_name] = store.load_dequant_weight(t_name, 2, device=dev)
            except Exception:
                pass
        if uniform_2bit_weights:
            c.print("[yellow]Evaluating Uniform 2-Bit Baseline...[/yellow]")
            patch_model_weights(model, uniform_2bit_weights)
            u2_ppl = _ppl()
            results["uniform_2bit"] = {
                "name": "Uniform 2-bit",
                "bpw": 2.0,
                "ppl": u2_ppl,
                "delta_ppl": u2_ppl - base_ppl,
                "compression": 8.0,
            }
            patch_model_weights(model, orig_weights)

    # Display Rich Comparison Table
    table = Table(title="AutoGSQ-RCO Benchmarking Results", header_style="bold green")
    table.add_column("Configuration", style="bold")
    table.add_column("BPW", justify="right")
    table.add_column("Compression", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Delta PPL", justify="right")

    for key, data in results.items():
        delta_str = f"{data['delta_ppl']:+.2f}" if data['delta_ppl'] != 0 else "-"
        table.add_row(
            data["name"],
            f"{data['bpw']:.2f}",
            f"{data['compression']:.1f}x",
            f"{data['ppl']:.2f}",
            delta_str,
        )

    c.print(table)
    return results
