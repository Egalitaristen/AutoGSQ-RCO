"""Regression tests for packer bugs surfaced during K-quant/IQ bring-up.

Each test pins a real failure mode: wrong superblock layouts, dropped
special tokens, pair-format merges, or ladder mis-mapping. All run on
synthetic data in seconds (no models, no GPU).
"""

import hashlib
import json

import numpy as np
import pytest
import torch

import autogsq.packer.calibration_repack as C
from autogsq.packer.hf_gguf_map import _read_tokenizer_bundle

try:
    import gguf
    from gguf import GGMLQuantizationType as QT

    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

needs_gguf = pytest.mark.skipif(not HAS_GGUF, reason="gguf package required")

K_CASES = [
    ("Q2_K", C.quantize_q2_K, C.dequantize_q2_K, 84),
    ("Q3_K", C.quantize_q3_K, C.dequantize_q3_K, 110),
    ("Q4_K", C.quantize_q4_K, C.dequantize_q4_K, 144),
    ("Q5_K", C.quantize_q5_K, C.dequantize_q5_K, 176),
    ("Q6_K", C.quantize_q6_K, C.dequantize_q6_K, 210),
]
IQ_CASES = [
    ("IQ2_XXS", C.quantize_iq2_xxs, C.dequantize_iq2_xxs, 66),
    ("IQ1_S", C.quantize_iq1_s, C.dequantize_iq1_s, 50),
]


@needs_gguf
@pytest.mark.parametrize("name,pack,depack,block_bytes", K_CASES + IQ_CASES)
def test_native_packer_bit_exact_vs_gguf_py(name, pack, depack, block_bytes):
    """Our bytes through gguf-py's independent dequantizer: zero difference.

    Guards struct field order, nibble packing, scale layout, and codebook
    tables in one assertion per type.
    """
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((1, 256)) * 0.04).astype(np.float32)
    p = pack(a)
    assert p.shape == (1, block_bytes), (name, p.shape)
    mine = depack(p, 1).reshape(a.shape)
    theirs = gguf.quants.dequantize(p.reshape(-1), QT[name]).reshape(a.shape)
    assert float(np.abs(mine - theirs).max()) == 0.0


def test_k_quant_requires_superblock_dim():
    with pytest.raises(AssertionError):
        C.quantize_q4_K(np.zeros((1, 100), dtype=np.float32))


def test_ladder_native_types_and_honest_fallback():
    assert C.closest_writable_type("Q4_K_M", 1024) == "Q4_K"
    assert C.closest_writable_type("IQ2_XXS", 256) == "IQ2_XXS"
    assert C.closest_writable_type("IQ1_S", 256) == "IQ1_S"
    # Last dim not a superblock multiple: must NOT claim a K/IQ type.
    assert C.closest_writable_type("Q4_K_M", 576) != "Q4_K"
    assert C.closest_writable_type("Q4_K_M", 576) in C.writable_gguf_types()


def _write_tokdir(d, vocab, added, merges):
    tok = {"model": {"vocab": vocab, "merges": merges}, "added_tokens": added}
    (d / "tokenizer.json").write_text(json.dumps(tok))
    (d / "tokenizer_config.json").write_text(json.dumps({}))


def test_tokenizer_bundle_merges_pairs_and_added_tokens(tmp_path):
    """Newer tokenizers store merges as pairs and specials outside vocab.

    Dropping either breaks files at load (SentencePiece mode on BPE vocab)
    or at tokenize time (missing specials).
    """
    _write_tokdir(
        tmp_path,
        {"hello": 0, "world": 1},
        [{"id": 2, "content": "<|end|>", "special": True}],
        [["hello", "world"]],
    )
    tokens, merges, tok_cfg, _ = _read_tokenizer_bundle(tmp_path)
    assert tokens == ["hello", "world", "<|end|>"]
    assert merges == ["hello world"]
    assert tok_cfg["__added_special_ids"] == [2]


def test_jobs_byte_identity(tmp_path):
    """Parallel repack must emit byte-identical files (order preserved)."""
    from autogsq.packer.gguf_writer import GGUFPacker

    torch.manual_seed(0)
    tensors = {f"blk.{i}.w.weight": torch.randn(256, 512) for i in range(4)}
    mk = lambda: {k: {"bits": 4, "gguf_type": "Q4_K_M"} for k in tensors}
    f1, f2 = tmp_path / "j1.gguf", tmp_path / "j2.gguf"
    GGUFPacker(architecture="llama").stitch_gguf(
        {k: v.clone() for k, v in tensors.items()}, mk(), f1, metadata={}, model_id="t"
    )
    GGUFPacker(architecture="llama").stitch_gguf(
        {k: v.clone() for k, v in tensors.items()}, mk(), f2, metadata={}, model_id="t", jobs=2
    )
    assert hashlib.sha256(f1.read_bytes()).hexdigest() == hashlib.sha256(f2.read_bytes()).hexdigest()
