"""On-disk candidate database manager and progress tracking for AutoGSQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


class CandidateStore:
    """Manages on-disk storage for quantized candidate weights, qparams sidecars, and progress."""

    def __init__(self, db_dir: Union[str, Path]) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.db_dir / "progress.json"

    def get_candidate_paths(self, layer_name: str, bits: Union[int, str]) -> Tuple[Path, Path]:
        """Return paths for (dequant_pth, qparams_pt)."""
        clean_name = layer_name.replace("/", "_")
        dequant_path = self.db_dir / f"{clean_name}.{bits}.pth"
        qparams_path = self.db_dir / f"{clean_name}.{bits}.qparams.pt"
        return dequant_path, qparams_path

    def is_candidate_complete(self, layer_name: str, bits: Union[int, str]) -> bool:
        """Check if candidate files exist on disk and progress records it."""
        dequant_path, qparams_path = self.get_candidate_paths(layer_name, bits)
        if not (dequant_path.is_file() and qparams_path.is_file()):
            return False

        progress = self.load_progress()
        completed = progress.get("completed_candidates", [])
        key = f"{layer_name}::{bits}"
        return key in completed

    def save_candidate(
        self,
        layer_name: str,
        bits: Union[int, str],
        dequant_weight: torch.Tensor,
        qparams: Dict[str, Any],
    ) -> None:
        """Save dequantized weight tensor and qparams sidecar to disk."""
        dequant_path, qparams_path = self.get_candidate_paths(layer_name, bits)

        # Validate qparams schema (§6.2)
        required_keys = {"qweight", "scales", "zeros", "bits", "group_size", "sym", "shape"}
        missing = required_keys - set(qparams.keys())
        if missing:
            raise ValueError(f"Missing required qparams fields: {missing}")

        # Save dequantized weights (FP/BF)
        torch.save(dequant_weight.detach().cpu(), dequant_path)

        # Save qparams dictionary
        torch.save(qparams, qparams_path)

        # Update progress.json
        self.mark_candidate_complete(layer_name, bits)

    def load_dequant_weight(
        self,
        layer_name: str,
        bits: Union[int, str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Load dequantized weight tensor from disk."""
        dequant_path, _ = self.get_candidate_paths(layer_name, bits)
        if not dequant_path.is_file():
            raise FileNotFoundError(f"Dequantized tensor not found: {dequant_path}")
        return torch.load(dequant_path, map_location=device or "cpu")

    def load_qparams(self, layer_name: str, bits: Union[int, str]) -> Dict[str, Any]:
        """Load qparams sidecar dictionary from disk."""
        _, qparams_path = self.get_candidate_paths(layer_name, bits)
        if not qparams_path.is_file():
            raise FileNotFoundError(f"qparams sidecar not found: {qparams_path}")
        return torch.load(qparams_path, map_location="cpu")

    def load_progress(self) -> Dict[str, Any]:
        """Load progress record."""
        if not self.progress_file.is_file():
            return {"completed_candidates": [], "completed_layers": []}
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"completed_candidates": [], "completed_layers": []}

    def mark_candidate_complete(self, layer_name: str, bits: Union[int, str]) -> None:
        """Record completed candidate in progress.json."""
        progress = self.load_progress()
        completed = progress.setdefault("completed_candidates", [])
        key = f"{layer_name}::{bits}"
        if key not in completed:
            completed.append(key)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def mark_layer_complete(self, layer_name: str) -> None:
        """Record completed layer in progress.json."""
        progress = self.load_progress()
        layers = progress.setdefault("completed_layers", [])
        if layer_name not in layers:
            layers.append(layer_name)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    def is_layer_complete(self, layer_name: str, bitwidth_options: List[Union[int, str]]) -> bool:
        """Check if all bit-width candidates for a layer are complete."""
        return all(self.is_candidate_complete(layer_name, bits) for bits in bitwidth_options)
