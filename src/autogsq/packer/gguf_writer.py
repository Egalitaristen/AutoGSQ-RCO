"""GGUF assembly, stitching, and allocation manifest injector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    gguf = None
    GGUFWriter = None
    GGMLQuantizationType = None

from autogsq.packer.calibration_repack import repack_tensor_to_gguf, get_ggml_type_code
from autogsq.packer.gguf_type_map import nearest_gguf_type


def _repack_one(payload: tuple) -> tuple:
    """Process-pool worker: repack one tensor (top-level for spawn pickling).

    Payload: (name, data_numpy, chosen_type). Returns
    (name, packed_numpy, ggml_type_code, actual_type).
    """
    name, data, chosen_type = payload
    packed, code, actual = repack_tensor_to_gguf(data, chosen_type)
    return name, np.asarray(packed), code, actual


class GGUFPacker:
    """Stitches candidate tensors into a llama.cpp-compatible GGUF file."""

    def __init__(self, architecture: str = "llama") -> None:
        self.architecture = architecture

    def stitch_gguf(
        self,
        tensors: Dict[str, Union[torch.Tensor, np.ndarray]],
        allocation: Dict[str, Dict[str, Any]],
        out_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        model_id: str = "custom_model",
        tensor_name_map: Optional[Dict[str, str]] = None,
        hf_config: Optional[Dict[str, Any]] = None,
        tokenizer_dir: Optional[Union[str, Path]] = None,
        vocab_size: Optional[int] = None,
        jobs: int = 1,
    ) -> Path:
        """Pack tensors into a GGUF file based on RCO allocation.

        Args:
            tensors: mapping of tensor_name -> tensor (dequantized or original FP/BF)
            allocation: mapping of tensor_name -> {'bits': ..., 'gguf_type': ...}
            out_path: destination .gguf file path
            metadata: additional metadata key-values
            model_id: model name/identifier
            tensor_name_map: HF name -> GGUF canonical name (unmapped pass through,
                which preserves extras such as grafted MTP heads)
            hf_config: HF config.json dict; when given, arch KV is written and
                the GGUF architecture tag is detected from it
            tokenizer_dir: local dir with tokenizer.json; when given, full
                tokenizer KV (tokens/merges/types/scores/ids/chat template)
                is written — required for llama.cpp to load the file
            vocab_size: token-list length; when given, token_embd/output rows
                are trimmed to it (models often ship wider embeddings than
                the tokenizer; llama.cpp requires exact agreement)
            jobs: parallel repack workers (process pool over tensors; the
                GGUF stream itself is still written serially in order, so
                output bytes are identical for any jobs value)

        Returns:
            Path to written GGUF file

        Note:
            The ``allocation`` mapping is updated in place: each entry's
            ``gguf_type`` is set to the type actually written, which may be a
            nearest writable fallback (see ``repack_tensor_to_gguf``).
        """
        from pathlib import Path as _Path
        from autogsq.packer.hf_gguf_map import (
            detect_arch,
            write_arch_kv,
            write_tokenizer_kv,
        )

        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        name_map_early = tensor_name_map or {}
        # Embedding/output rows must equal the token-list length (llama.cpp
        # enforces it). Trailing rows beyond vocab_size are unused reserves.
        # Done before KV writing so the trim is recorded in-file, not just
        # in the sidecar.
        metadata = dict(metadata or {})
        if vocab_size is not None:
            for key, t in list(tensors.items()):
                gname = name_map_early.get(key, key)
                if gname in ("token_embd.weight", "output.weight") and t.shape[0] > vocab_size:
                    note = f"trimming {gname} rows {t.shape[0]} -> {vocab_size}"
                    print(note)
                    sl = t[:vocab_size]
                    tensors[key] = sl.clone() if isinstance(sl, torch.Tensor) else sl.copy()
                    metadata["autogsq_vocab_trim"] = note

        if GGUFWriter is None:
            raise ImportError("The 'gguf' Python package is required to write GGUF files.")

        arch = detect_arch(hf_config, self.architecture) if hf_config else self.architecture
        writer = GGUFWriter(str(out_file), arch)
        writer.add_name(model_id)

        kv_notes = []
        if hf_config:
            kv_notes.extend(write_arch_kv(writer, hf_config, arch))
        if tokenizer_dir is not None and (_Path(tokenizer_dir) / "tokenizer.json").is_file():
            mt = str((hf_config or {}).get("model_type", "") or "")
            kv_notes.extend(write_tokenizer_kv(writer, tokenizer_dir, model_type=mt))
        writer.add_type("model")
        # Honest file_type from the dominant written tensor type.
        from collections import Counter as _Counter
        from autogsq.packer.calibration_repack import closest_writable_type as _closest

        _ftype_map = {"Q4_0": 2, "Q4_1": 3, "Q8_0": 7, "Q5_0": 8, "Q5_1": 9}
        _counts = _Counter(
            _closest(str((allocation.get(n, {}) or {}).get("gguf_type", "F16")))
            for n in tensors
        )
        _counts.pop("F32", None)
        _counts.pop("F16", None)
        _counts.pop("BF16", None)
        _dominant = _counts.most_common(1)[0][0] if _counts else "F16"
        writer.add_file_type(_ftype_map.get(_dominant, 1))
        if kv_notes:
            writer.add_string("autogsq.kv_written", ",".join(kv_notes))

        # Add custom metadata
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, str):
                    writer.add_string(k, v)
                elif isinstance(v, float):
                    writer.add_float32(k, float(v))
                elif isinstance(v, int):
                    writer.add_uint32(k, int(v))

        # Add allocation summary to KV metadata
        writer.add_string("autogsq.version", "1.0.0")

        name_map = tensor_name_map or {}

        # Repack each tensor (parallel over tensors when jobs > 1; the GGUF
        # stream below stays serial and ordered, so bytes are jobs-invariant).
        repacked: Dict[str, tuple] = {}
        if jobs is not None and jobs > 1:
            import os as _os
            from concurrent.futures import ProcessPoolExecutor as _Pool

            def _as_np(t: Any) -> Any:
                if isinstance(t, torch.Tensor):
                    return t.detach().cpu().to(torch.float32).numpy()
                return np.asarray(t)

            payloads = [
                (name, _as_np(tdata), (allocation.get(name, {}) or {}).get("gguf_type", "F16"))
                for name, tdata in tensors.items()
            ]
            workers = max(1, min(int(jobs), _os.cpu_count() or 1, len(payloads)))
            with _Pool(max_workers=workers) as pool:
                for name, arr, code, actual in pool.map(_repack_one, payloads, chunksize=1):
                    repacked[name] = (arr, code, actual)
        else:
            for name, tensor_data in tensors.items():
                alloc_info = allocation.get(name, {})
                chosen_type = alloc_info.get("gguf_type", "F16")
                repacked_arr, ggml_type_code, actual_type = repack_tensor_to_gguf(tensor_data, chosen_type)
                if isinstance(repacked_arr, torch.Tensor):
                    repacked_arr = repacked_arr.cpu().numpy()
                repacked[name] = (np.asarray(repacked_arr), ggml_type_code, actual_type)

        for name, tensor_data in tensors.items():
            alloc_info = allocation.get(name, {})
            gguf_name = name_map.get(name, name)

            # Block quantization repack
            repacked_arr, ggml_type_code, actual_type = repacked[name]
            if isinstance(alloc_info, dict):
                alloc_info["gguf_type"] = actual_type

            # Ensure proper array properties
            if isinstance(repacked_arr, torch.Tensor):
                repacked_arr = repacked_arr.cpu().numpy()

            raw_type = GGMLQuantizationType(ggml_type_code) if GGMLQuantizationType else ggml_type_code
            # gguf-py derives the logical shape from the byte layout itself
            # (quant_shape_from_byte_shape), so packed arrays must follow the
            # (rows, superblocks * type_size) convention — which ours do.
            writer.add_tensor(gguf_name, repacked_arr, raw_dtype=raw_type)
            if isinstance(alloc_info, dict) and gguf_name != name:
                alloc_info["gguf_name"] = gguf_name

        # Write to disk
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # Inject allocation manifests sidecar
        self.inject_metadata(out_file, allocation, model_id=model_id, metadata=metadata)

        return out_file

    def inject_metadata(
        self,
        gguf_path: Union[str, Path],
        allocation_manifest: Dict[str, Any],
        model_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """Write sidecar .rco-allocation.json and .rco-allocation.txt files.

        Schema matches Section 6.3:
        {
          "target_bpw": ...,
          "achieved_bpw": ...,
          "model_id": ...,
          "per_tensor": {
             "tensor_name": {"bits": ..., "gguf_type": ...}
          },
          "eval": {...}
        }
        """
        gguf_file = Path(gguf_path)
        base_path = gguf_file.with_suffix("")

        json_path = Path(f"{base_path}.rco-allocation.json")
        txt_path = Path(f"{base_path}.rco-allocation.txt")

        # Normalize manifest dictionary
        target_bpw = 0.0
        achieved_bpw = 0.0
        if metadata:
            target_bpw = float(metadata.get("target_bpw", 0.0))
            achieved_bpw = float(metadata.get("achieved_bpw", 0.0))

        # Format per_tensor map
        per_tensor: Dict[str, Dict[str, Any]] = {}
        for tensor_name, info in allocation_manifest.items():
            if isinstance(info, dict):
                bits_val = info.get("bits", 4)
                gguf_type_val = info.get("gguf_type") or nearest_gguf_type(bits_val)
                per_tensor[tensor_name] = {"bits": bits_val, "gguf_type": gguf_type_val}
            else:
                bits_val = info
                gguf_type_val = nearest_gguf_type(bits_val)
                per_tensor[tensor_name] = {"bits": bits_val, "gguf_type": gguf_type_val}

        manifest_data = {
            "target_bpw": target_bpw,
            "achieved_bpw": achieved_bpw,
            "model_id": model_id,
            "per_tensor": per_tensor,
            "eval": metadata.get("eval", {}) if metadata else {},
        }

        # Write JSON sidecar
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Write TXT sidecar (one line per tensor for easy diffing)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# AutoGSQ-RCO Allocation Manifest\n")
            f.write(f"# Model: {model_id}\n")
            f.write(f"# Target BPW: {target_bpw} | Achieved BPW: {achieved_bpw}\n\n")
            for t_name, t_info in sorted(per_tensor.items()):
                f.write(f"{t_name}: {t_info['bits']} bits -> {t_info['gguf_type']}\n")

        return json_path, txt_path
