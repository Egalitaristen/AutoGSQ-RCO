"""Unit tests for GGUF type mapping, block repacking, and file roundtrip."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from autogsq.packer.gguf_type_map import nearest_gguf_type
from autogsq.packer.calibration_repack import (
    quantize_q8_0,
    quantize_q4_0,
    repack_tensor_to_gguf,
)
from autogsq.packer.gguf_writer import GGUFPacker

try:
    import gguf
    from gguf import GGUFReader
except ImportError:
    gguf = None
    GGUFReader = None


def test_nearest_gguf_type():
    """Verify bit-width mapping and tolerance fallbacks."""
    # Near 2.06 bpw
    assert nearest_gguf_type(2.06, tolerance=0.1) == "IQ2_XXS"
    # Near 4.5 bpw
    assert nearest_gguf_type(4.5, tolerance=0.1) in ("Q4_0", "Q4_K_S", "IQ4_NL")
    # Ternary
    assert nearest_gguf_type("ternary") in ("IQ1_S", "IQ1_M", "IQ2_XXS")

    # Drift exceeding tolerance with error fallback
    with pytest.raises(ValueError):
        nearest_gguf_type(10.5, tolerance=0.1, fallback="error")

    # Round down fallback
    assert nearest_gguf_type(10.5, tolerance=0.1, fallback="round_down") in ("Q8_0", "Q6_K")


def test_quantize_q8_0_and_q4_0():
    """Verify block quantization functions produce correct byte layouts."""
    np.random.seed(42)
    data = np.random.randn(64).astype(np.float32)

    # Q8_0: 64 elements = 2 blocks of 32 elements. Each block is 34 bytes -> 68 bytes total.
    q8_bytes = quantize_q8_0(data)
    assert q8_bytes.nbytes == 68

    # Q4_0: 64 elements = 2 blocks of 32 elements. Each block is 18 bytes -> 36 bytes total.
    q4_bytes = quantize_q4_0(data)
    assert q4_bytes.nbytes == 36

    # 2-D inputs keep row-major block layout: (rows, bytes_per_row).
    q8_2d = quantize_q8_0(np.random.randn(4, 64).astype(np.float32))
    assert q8_2d.shape == (4, 68)
    q4_2d = quantize_q4_0(np.random.randn(4, 64).astype(np.float32))
    assert q4_2d.shape == (4, 36)


@pytest.mark.skipif(gguf is None, reason="gguf package not installed")
def test_gguf_stitch_and_reader_roundtrip(tmp_path: Path):
    """Verify assembling a GGUF file and reading it back with gguf.GGUFReader."""
    packer = GGUFPacker(architecture="llama")

    tensors = {
        "model.layers.0.self_attn.q_proj": torch.randn(64, 64, dtype=torch.float32),
        "model.layers.0.mlp.down_proj": torch.randn(64, 64, dtype=torch.float32),
    }

    allocation = {
        "model.layers.0.self_attn.q_proj": {"bits": 4, "gguf_type": "Q4_0"},
        "model.layers.0.mlp.down_proj": {"bits": 8, "gguf_type": "Q8_0"},
    }

    out_file = tmp_path / "test_model.gguf"
    packer.stitch_gguf(
        tensors=tensors,
        allocation=allocation,
        out_path=out_file,
        metadata={"target_bpw": 2.75, "achieved_bpw": 2.749},
        model_id="test_llama_model",
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 0

    # Verify sidecar files (§6.3)
    json_sidecar = tmp_path / "test_model.rco-allocation.json"
    txt_sidecar = tmp_path / "test_model.rco-allocation.txt"
    assert json_sidecar.is_file()
    assert txt_sidecar.is_file()

    with open(json_sidecar, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["target_bpw"] == 2.75
        assert "model.layers.0.self_attn.q_proj" in manifest["per_tensor"]

    # Read back with GGUFReader
    reader = GGUFReader(str(out_file))
    tensor_names = [t.name for t in reader.tensors]
    assert "model.layers.0.self_attn.q_proj" in tensor_names
    assert "model.layers.0.mlp.down_proj" in tensor_names

    # Check metadata in GGUF
    assert "general.architecture" in reader.fields
    arch_val = reader.fields["general.architecture"].contents()
    assert arch_val == "llama"
