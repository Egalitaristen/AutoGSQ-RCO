"""Backfill per-candidate reconstruction MSE into existing candidate databases.

Databases built before MSE tracking (see ``train_gsq_layer``) lack the
``"mse"`` key in their ``.qparams.pt`` sidecars that RCO allocation needs
as its fidelity signal. This module recomputes it from the stored dequant
weights and the original model weights, without rebuilding the database.

Works for any model: the weight source can be a safetensors file, a
directory of safetensors shards, or a HuggingFace model ID / local dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def _resolve_safetensors_files(model_source: Union[str, Path]) -> List[Path]:
    """Return safetensors files for a file path, directory, or raise."""
    p = Path(model_source)
    if p.is_file() and p.suffix == ".safetensors":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
        if files:
            return files
    raise ValueError(
        f"No safetensors file(s) found at {model_source!r}; "
        "pass a .safetensors file, a directory containing shards, or use "
        "model_id= with a HuggingFace ID."
    )


def _load_original_weight(
    tensor_name: str,
    shard_files: List[Path],
    model_id: Optional[str] = None,
) -> torch.Tensor:
    """Load one original weight tensor, preferring local safetensors shards."""
    if shard_files:
        from safetensors import safe_open

        for shard in shard_files:
            with safe_open(str(shard), framework="pt") as f:
                if tensor_name in f.keys():
                    # Clone: mmap views dangle after the handle closes.
                    return f.get_tensor(tensor_name).clone().cpu()
        raise KeyError(f"Tensor {tensor_name!r} not found in {len(shard_files)} shard(s)")
    # Fallback: load full model state dict (CPU) for a HF ID / local dir.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=False
    )
    state = model.state_dict()
    if tensor_name not in state:
        raise KeyError(f"Tensor {tensor_name!r} not found in model {model_id!r}")
    return state[tensor_name].detach().cpu()


def backfill_mse(
    db_dir: Union[str, Path],
    model_source: Optional[Union[str, Path]] = None,
    model_id: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Add ``"mse"`` to every completed candidate's qparams sidecar.

    Args:
        db_dir: candidate database directory (with progress.json).
        model_source: .safetensors file or directory of shards holding the
            original FP weights. Defaults to ``db_dir`` itself when it
            contains safetensors, else falls back to ``model_id``.
        model_id: HuggingFace ID / local model dir used only when no local
            safetensors shards are available.
        overwrite: recompute even when ``"mse"`` already exists.

    Returns:
        Summary dict with counts and per-candidate MSE values.
    """
    from autogsq.common.store import CandidateStore

    store = CandidateStore(db_dir)
    progress = store.load_progress()
    completed = progress.get("completed_candidates", [])

    shard_files: List[Path] = []
    if model_source is not None:
        shard_files = _resolve_safetensors_files(model_source)
    else:
        # Convenience: model shards sometimes live next to the DB.
        candidate = Path(db_dir).glob("*.safetensors")
        shard_files = sorted(candidate)

    summary: Dict[str, Any] = {
        "backfilled": 0,
        "skipped": 0,
        "errors": [],
        "mse": {},
    }

    for entry in completed:
        if "::" not in entry:
            continue
        tensor_name, bits = entry.split("::", 1)
        key = f"{tensor_name}::{bits}"
        try:
            _, qparams_path = store.get_candidate_paths(tensor_name, bits)
            qparams = torch.load(str(qparams_path), map_location="cpu")
            if "mse" in qparams and not overwrite:
                summary["skipped"] += 1
                summary["mse"][key] = float(qparams["mse"])
                continue
            dequant = store.load_dequant_weight(tensor_name, bits, device=torch.device("cpu"))
            orig = _load_original_weight(tensor_name, shard_files, model_id=model_id)
            mse = F.mse_loss(
                dequant.to(dtype=orig.dtype).reshape(orig.shape), orig
            )
            qparams["mse"] = float(mse.detach().cpu().item())
            torch.save(qparams, str(qparams_path))
            summary["backfilled"] += 1
            summary["mse"][key] = qparams["mse"]
        except Exception as exc:  # keep going; report at the end
            summary["errors"].append(f"{key}: {exc}")

    return summary
