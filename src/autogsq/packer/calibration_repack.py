"""Calibration repack and block quantization for GGUF formats."""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

try:
    import gguf
    from gguf import GGMLQuantizationType
except ImportError:
    gguf = None
    GGMLQuantizationType = None


# Name to GGMLQuantizationType mapping
NAME_TO_GGML_TYPE: Dict[str, int] = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q4_K_M": 12,
    "Q4_K_S": 12,
    "Q5_K": 13,
    "Q5_K_M": 13,
    "Q5_K_S": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "IQ2_XXS": 16,
    "IQ2_XS": 17,
    "IQ3_XXS": 18,
    "IQ1_S": 19,
    "IQ4_NL": 20,
    "IQ3_S": 21,
    "IQ2_S": 22,
    "IQ4_XS": 23,
    "IQ2_M": 24,
    "IQ3_M": 26,
    "BF16": 29,
    # Named variants share their family's GGML code.
    "Q3_K_S": 11,
    "Q3_K_M": 11,
    "Q3_K_L": 11,
}

# Approximate bpw for writable types missing from GGUF_TYPE_SPECS.
_WRITABLE_BPW_OVERRIDE: Dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q5_1": 6.0,
    "Q5_0": 5.5,
    "Q4_1": 5.0,
    "Q4_0": 4.5,
    "Q4_K": 4.5,
    "IQ2_XXS": 2.06,
    "IQ1_S": 1.56,
}


def _probe_writable_types() -> List[str]:
    """Detect which GGUF types this writer can actually emit.

    gguf-py's numpy quantizers only implement the simple types; K-quants and
    IQ types raise. Probing once at import keeps every fallback honest.
    """
    candidates = ["F32", "F16", "BF16", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0"]
    if gguf is None or not hasattr(gguf, "quants") or not hasattr(gguf.quants, "quantize"):
        return ["Q8_0", "Q4_0", "F16", "F32"]
    writable = ["F32", "F16"]  # handled via direct casts, always available
    probe = np.zeros((8, 256), dtype=np.float32)  # 2-D: satisfies block-shape checks
    for name in candidates:
        if name in writable:
            continue
        try:
            code = get_ggml_type_code(name)
            enum = GGMLQuantizationType(code)
            gguf.quants.quantize(probe, enum)
            writable.append(name)
        except Exception:
            continue
    return writable


_WRITABLE_CACHE: Optional[List[str]] = None


def writable_gguf_types() -> List[str]:
    """Types this writer can actually emit (probed once, lazily).

    Lazy because probing during module import is unreliable (gguf.quants
    may not be fully initialized at that point); results are cached.
    Includes this module's own K-quant packers beyond gguf-py's set.
    """
    global _WRITABLE_CACHE
    if _WRITABLE_CACHE is None:
        probed = _probe_writable_types()
        _WRITABLE_CACHE = probed + [t for t in OUR_PACKERS if t not in probed]
    return _WRITABLE_CACHE


def _type_bpw(type_name: str) -> float:
    """Effective bpw for fallback comparison (specs table, then overrides)."""
    from autogsq.packer.gguf_type_map import GGUF_TYPE_SPECS

    spec = GGUF_TYPE_SPECS.get(type_name)
    if spec is not None:
        return float(spec["bpw"])
    return _WRITABLE_BPW_OVERRIDE.get(type_name.upper().strip(), 16.0)


def closest_writable_type(requested: str, n_cols: Optional[int] = None) -> str:
    """Map a requested GGUF type to the nearest type this writer can emit.

    K-quant superblocks need the last dim to be a multiple of 256; when it
    is not (or unknown), K types are excluded from candidates.
    """
    writable = writable_gguf_types()
    req = requested.upper().strip()
    if req in writable and (req not in _K_TYPES or (n_cols is not None and can_pack_kquant(n_cols))):
        return req
    cands = [
        n for n in writable
        if n not in _K_TYPES or (n_cols is not None and can_pack_kquant(n_cols))
    ] or writable
    try:
        target = _type_bpw(req)
    except Exception:
        target = 4.5

    def _same_family(cand: str) -> bool:
        return req == cand or req.startswith(cand + "_") or cand.startswith(req + "_")

    ranked = sorted(
        cands, key=lambda n: (0 if _same_family(n) else 1, abs(_type_bpw(n) - target), _type_bpw(n))
    )
    return ranked[0] if ranked else "Q4_0"


def get_ggml_type_code(type_name: str) -> int:
    """Return integer GGMLQuantizationType enum value."""
    clean_name = type_name.upper().strip()
    if clean_name in NAME_TO_GGML_TYPE:
        return NAME_TO_GGML_TYPE[clean_name]
    if hasattr(GGMLQuantizationType, clean_name):
        return int(getattr(GGMLQuantizationType, clean_name))
    return 1  # Default to F16


def quantize_q8_0(data: np.ndarray) -> np.ndarray:
    """Quantize 1D float array into standard GGUF Q8_0 blocks (block size 32).

    Each block contains:
      - 2 bytes: scale d (float16)
      - 32 bytes: int8 quants (32 bytes)
    Total: 34 bytes per 32 elements.

    Returns a uint8 array shaped (rows, bytes_per_row) matching the layout
    of gguf-py's native quantizers (2-D inputs keep their row count).
    """
    arr2d = data.astype(np.float32)
    rows = arr2d.shape[0] if arr2d.ndim == 2 else 1
    flat = arr2d.ravel()
    n_blocks = len(flat) // 32
    if len(flat) % 32 != 0:
        pad_len = (n_blocks + 1) * 32 - len(flat)
        flat = np.pad(flat, (0, pad_len))
        n_blocks += 1

    blocks = flat.reshape(n_blocks, 32)
    # Scale per block: max(abs(x)) / 127
    max_abs = np.max(np.abs(blocks), axis=1)
    d = (max_abs / 127.0).astype(np.float16)

    # Avoid div by zero
    d_safe = np.where(d == 0, np.float16(1.0), d).astype(np.float32)
    qs = np.round(blocks / d_safe[:, None]).clip(-128, 127).astype(np.int8)

    # Pack into byte buffer: 34 bytes per block
    packed = bytearray(n_blocks * 34)
    d_bytes = d.tobytes()
    qs_bytes = qs.tobytes()

    for b in range(n_blocks):
        out_offset = b * 34
        packed[out_offset : out_offset + 2] = d_bytes[b * 2 : b * 2 + 2]
        packed[out_offset + 2 : out_offset + 34] = qs_bytes[b * 32 : b * 32 + 32]

    return np.frombuffer(packed, dtype=np.uint8).reshape(rows, -1)


def quantize_q4_0(data: np.ndarray) -> np.ndarray:
    """Quantize float array into standard GGUF Q4_0 blocks (block size 32).

    Each block contains:
      - 2 bytes: scale d (float16)
      - 16 bytes: 32 4-bit nibbles (qs)
    Total: 18 bytes per 32 elements.

    Returns a uint8 array shaped (rows, bytes_per_row) matching the layout
    of gguf-py's native quantizers (2-D inputs keep their row count).
    """
    arr2d = data.astype(np.float32)
    rows = arr2d.shape[0] if arr2d.ndim == 2 else 1
    flat = arr2d.ravel()
    n_blocks = len(flat) // 32
    if len(flat) % 32 != 0:
        pad_len = (n_blocks + 1) * 32 - len(flat)
        flat = np.pad(flat, (0, pad_len))
        n_blocks += 1

    blocks = flat.reshape(n_blocks, 32)
    max_abs = np.max(np.abs(blocks), axis=1)
    d = (max_abs / -8.0).astype(np.float16)
    d_safe = np.where(d == 0, np.float16(1.0), d).astype(np.float32)

    # Values scaled into [-8, 7] and shifted to [0, 15]
    qs = np.round(blocks / d_safe[:, None] + 8.0).clip(0, 15).astype(np.uint8)

    packed = bytearray(n_blocks * 18)
    d_bytes = d.tobytes()

    for b in range(n_blocks):
        out_offset = b * 18
        packed[out_offset : out_offset + 2] = d_bytes[b * 2 : b * 2 + 2]
        # Pack pairs of 4-bit values into bytes
        b_qs = qs[b]
        low_nibbles = b_qs[0:16]
        high_nibbles = b_qs[16:32]
        packed_bytes = (low_nibbles | (high_nibbles << 4)).tobytes()
        packed[out_offset + 2 : out_offset + 18] = packed_bytes

    return np.frombuffer(packed, dtype=np.uint8).reshape(rows, -1)


def _fp16_bytes(values: np.ndarray) -> bytes:
    """Float32 array -> little-endian fp16 bytes (matches GGML_FP32_TO_FP16)."""
    return np.asarray(values, dtype=np.float32).astype("<f2").tobytes()


def _unpack_scale_min_k4(scales: np.ndarray, j: int) -> tuple:
    """Mirror of ggml get_scale_min_k4: 6-bit (sc, m) for sub-block j."""
    q = scales
    if j < 4:
        return int(q[j] & 63), int(q[j + 4] & 63)
    return (
        int((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
        int((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
    )


def quantize_q4_K(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF Q4_K superblocks (256 elems -> 144 bytes).

    Bit-exact format mirror of ggml ``quantize_row_q4_K``: 8 sub-blocks of 32
    with per-sub-block 6-bit scale/min under fp16 super-scales
    (``x = d*sc*q - dmin*m``), nibble-packed codes. The sub-block grid search
    uses min/max (reference refines with Lloyd iterations; same format,
    marginally higher error, far below Q4_0 thanks to sub-block granularity
    and asymmetric mins).

    Returns uint8 array shaped (n_rows, n_superblocks_per_row * 144), the
    gguf byte-shape convention (mirror of ``quant_shape_to_byte_shape``).
    Input last dim must be a multiple of 256 (see :func:`can_pack_kquant`).
    """
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"Q4_K needs last dim % 256 == 0, got {n_cols}"
    n_super = n_cols // 256
    out = np.zeros((n_rows, n_super * 144), dtype=np.uint8)

    for r in range(n_rows):
        for s in range(n_super):
            blk = flat_rows[r, s * 256 : (s + 1) * 256].reshape(8, 32)
            mn = blk.min(axis=1)
            mx = blk.max(axis=1)
            mn = np.minimum(mn, 0.0)  # mirror make_qkx3: min clamped <= 0
            rng = (mx - mn).astype(np.float64)
            sub_s = np.where(rng > 0, rng / 15.0, 0.0)
            sub_m = (-mn).astype(np.float64)

            # Second level: 6-bit uniform over max (mirror make_qp_quants core).
            smax = float(sub_s.max())
            mmax = float(sub_m.max())
            d_block = smax / 63.0 if smax > 0 else 0.0
            m_block = mmax / 63.0 if mmax > 0 else 0.0
            Ls = np.clip(np.round(sub_s / d_block), 0, 63).astype(np.int64) if d_block > 0 else np.zeros(8, dtype=np.int64)
            Lm = np.clip(np.round(sub_m / m_block), 0, 63).astype(np.int64) if m_block > 0 else np.zeros(8, dtype=np.int64)

            scales = np.zeros(12, dtype=np.uint8)
            for j in range(8):
                ls, lm = int(Ls[j]), int(Lm[j])
                if j < 4:
                    scales[j] = ls
                    scales[j + 4] = lm
                else:
                    scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
                    scales[j - 4] |= ((ls >> 4) << 6)
                    scales[j] |= ((lm >> 4) << 6)

            d_f16 = np.float16(d_block).astype(np.float32).item()
            dm_f16 = np.float16(m_block).astype(np.float32).item()
            codes = np.zeros(256, dtype=np.uint8)
            for j in range(8):
                sc, m = _unpack_scale_min_k4(scales, j)
                d = d_f16 * sc
                if not d:
                    continue
                dm = dm_f16 * m
                q = np.clip(np.round((blk[j] + dm) / d), 0, 15).astype(np.uint8)
                codes[j * 32 : (j + 1) * 32] = q

            qs = np.zeros(128, dtype=np.uint8)
            for j in range(0, 256, 64):
                for l in range(32):
                    qs[(j // 64) * 32 + l] = codes[j + l] | (codes[j + l + 32] << 4)

            o = out[r, s * 144 : (s + 1) * 144]
            o[0:2] = np.frombuffer(_fp16_bytes(np.array([d_block])), dtype=np.uint8)
            o[2:4] = np.frombuffer(_fp16_bytes(np.array([m_block])), dtype=np.uint8)
            o[4:16] = scales
            o[16:144] = qs

    return out


def dequantize_q4_K(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize Q4_K superblock bytes (self-verification mirror)."""
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_super = raw.shape[1] // 144
    out = np.zeros((n_rows, n_super * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for s in range(n_super):
            o = raw[r, s * 144 : (s + 1) * 144]
            d = float(f16(o[0:2])[0])
            dm = float(f16(o[2:4])[0])
            scales = o[4:16]
            qs = o[16:144]
            codes = np.zeros(256, dtype=np.float32)
            for j in range(0, 256, 64):
                for l in range(32):
                    b = int(qs[(j // 64) * 32 + l])
                    codes[j + l] = b & 0xF
                    codes[j + l + 32] = (b >> 4) & 0xF
            for j in range(8):
                sc, m = _unpack_scale_min_k4(scales, j)
                out[r, s * 256 + j * 32 : s * 256 + (j + 1) * 32] = (
                    d * sc * codes[j * 32 : (j + 1) * 32] - dm * m
                )
    return out


def can_pack_kquant(n_cols: int) -> bool:
    """K-quants require the last dim to be a multiple of the 256 superblock."""
    return n_cols % 256 == 0


def _nearest_int(a: np.ndarray) -> np.ndarray:
    """Round half away from zero (mirror of ggml ``nearest_int``)."""
    x = np.asarray(a, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def _pack_k45_block(blk: np.ndarray) -> tuple:
    """Shared Q4_K/Q5_K front end: 8x32 min/max grid + 6-bit scale/min pack.

    Returns (scales[12] uint8, d_block, m_block, d_f16, dm_f16). Same
    estimation as :func:`quantize_q4_K` (min/max grid; the reference refines
    with Lloyd iterations — same format, marginally higher error).
    """
    mn = blk.min(axis=1)
    mx = blk.max(axis=1)
    mn = np.minimum(mn, 0.0)
    rng = (mx - mn).astype(np.float64)
    sub_s = np.where(rng > 0, rng / 15.0, 0.0)
    sub_m = (-mn).astype(np.float64)

    smax = float(sub_s.max())
    mmax = float(sub_m.max())
    d_block = smax / 63.0 if smax > 0 else 0.0
    m_block = mmax / 63.0 if mmax > 0 else 0.0
    Ls = np.clip(_nearest_int(sub_s / d_block), 0, 63) if d_block > 0 else np.zeros(8, dtype=np.int64)
    Lm = np.clip(_nearest_int(sub_m / m_block), 0, 63) if m_block > 0 else np.zeros(8, dtype=np.int64)

    scales = np.zeros(12, dtype=np.uint8)
    for j in range(8):
        ls, lm = int(Ls[j]), int(Lm[j])
        if j < 4:
            scales[j] = ls
            scales[j + 4] = lm
        else:
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
            scales[j - 4] |= ((ls >> 4) << 6)
            scales[j] |= ((lm >> 4) << 6)

    d_f16 = np.float16(d_block).astype(np.float32).item()
    dm_f16 = np.float16(m_block).astype(np.float32).item()
    return scales, d_block, m_block, d_f16, dm_f16


def quantize_q5_K(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF Q5_K superblocks (256 elems -> 176 bytes).

    Format mirror of ggml ``quantize_row_q5_K``: identical scale/min front
    end to Q4_K, 5-bit codes (``x = d*sc*q - dmin*m``) with the high bit
    split into ``qh`` exactly per the reference packing (masks 1,2 <<= 2
    per 64-group). Layout per superblock: d(2), dmin(2), scales(12),
    qh(32), qs(128).

    Returns uint8 array shaped (n_rows, n_superblocks_per_row * 176).
    Input last dim must be a multiple of 256 (see :func:`can_pack_kquant`).
    """
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"Q5_K needs last dim % 256 == 0, got {n_cols}"
    n_super = n_cols // 256
    out = np.zeros((n_rows, n_super * 176), dtype=np.uint8)

    for r in range(n_rows):
        for s in range(n_super):
            blk = flat_rows[r, s * 256 : (s + 1) * 256].reshape(8, 32)
            scales, d_block, m_block, d_f16, dm_f16 = _pack_k45_block(blk)

            codes = np.zeros(256, dtype=np.int64)
            for j in range(8):
                sc, m = _unpack_scale_min_k4(scales, j)
                d = d_f16 * sc
                if not d:
                    continue
                dm = dm_f16 * m
                codes[j * 32 : (j + 1) * 32] = np.clip(
                    _nearest_int((blk[j] + dm) / d), 0, 31
                )

            qh = np.zeros(32, dtype=np.uint8)
            qs = np.zeros(128, dtype=np.uint8)
            m1, m2 = 1, 2
            for n in range(0, 256, 64):
                for j in range(32):
                    l1 = int(codes[n + j])
                    if l1 > 15:
                        l1 -= 16
                        qh[j] |= m1
                    l2 = int(codes[n + j + 32])
                    if l2 > 15:
                        l2 -= 16
                        qh[j] |= m2
                    qs[(n // 64) * 32 + j] = l1 | (l2 << 4)
                m1 <<= 2
                m2 <<= 2

            o = out[r, s * 176 : (s + 1) * 176]
            o[0:2] = np.frombuffer(_fp16_bytes(np.array([d_block])), dtype=np.uint8)
            o[2:4] = np.frombuffer(_fp16_bytes(np.array([m_block])), dtype=np.uint8)
            o[4:16] = scales
            o[16:48] = qh
            o[48:176] = qs

    return out


def dequantize_q5_K(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize Q5_K superblock bytes (self-verification mirror)."""
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_super = raw.shape[1] // 176
    out = np.zeros((n_rows, n_super * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for s in range(n_super):
            o = raw[r, s * 176 : (s + 1) * 176]
            d = float(f16(o[0:2])[0])
            dm = float(f16(o[2:4])[0])
            scales = o[4:16]
            qh = o[16:48]
            qs = o[48:176]
            u1, u2 = 1, 2
            for g in range(4):
                sc0, m0 = _unpack_scale_min_k4(scales, 2 * g)
                sc1, m1 = _unpack_scale_min_k4(scales, 2 * g + 1)
                d1, mm1 = d * sc0, dm * m0
                d2, mm2 = d * sc1, dm * m1
                for l in range(32):
                    b = int(qs[g * 32 + l])
                    out[r, s * 256 + g * 64 + l] = d1 * ((b & 0xF) + (16 if qh[l] & u1 else 0)) - mm1
                    out[r, s * 256 + g * 64 + 32 + l] = d2 * ((b >> 4) + (16 if qh[l] & u2 else 0)) - mm2
                u1 <<= 2
                u2 <<= 2
    return out


def _make_qx_quants_16_32(x: np.ndarray) -> tuple:
    """Faithful port of ggml ``make_qx_quants(16, 32, x, L, rmse_type=1)``.

    Returns (scale, L[16] int array centered at nmax=32, i.e. codes 0..63).
    Includes the +/-9 iscale refinement loop.
    """
    n, nmax = 16, 32
    x = np.asarray(x, dtype=np.float64)
    amax, vmax = 0.0, 0.0
    for v in x:
        if abs(v) > amax:
            amax, vmax = abs(v), float(v)
    if amax < 1e-15:
        return 0.0, np.zeros(16, dtype=np.int64)
    iscale = -nmax / vmax

    def _codes(isc: float) -> np.ndarray:
        return np.clip(_nearest_int(isc * x), -nmax, nmax - 1) + nmax

    L = _codes(iscale)
    lz = L - nmax
    w = x * x
    sumlx = float((w * x * lz).sum())
    suml2 = float((w * lz * lz).sum())
    scale = sumlx / suml2 if suml2 else 0.0
    best = scale * sumlx
    for i in range(-9, 10):
        if i == 0:
            continue
        isc = -(nmax + 0.1 * i) / vmax
        Li = _codes(isc)
        lzi = Li - nmax
        slx = float((w * x * lzi).sum())
        sl2 = float((w * lzi * lzi).sum())
        if sl2 > 0 and slx * slx > best * sl2:
            L, scale, best = Li, slx / sl2, (slx / sl2) * slx
    return scale, L


def quantize_q6_K(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF Q6_K superblocks (256 elems -> 210 bytes).

    Format mirror of ggml ``quantize_row_q6_K``: 16 sub-blocks of 16 with
    ``make_qx_quants`` scales, int8-quantized sub-scales under one fp16
    super-scale (``x = d*sc*(q - 32)``), bit-split code packing per the
    reference. Layout per superblock: ql(128), qh(64), scales(16), d(2).

    Returns uint8 array shaped (n_rows, n_superblocks_per_row * 210).
    Input last dim must be a multiple of 256 (see :func:`can_pack_kquant`).
    """
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"Q6_K needs last dim % 256 == 0, got {n_cols}"
    n_super = n_cols // 256
    out = np.zeros((n_rows, n_super * 210), dtype=np.uint8)

    for r in range(n_rows):
        for s in range(n_super):
            blk = flat_rows[r, s * 256 : (s + 1) * 256]
            L = np.zeros(256, dtype=np.int64)
            sub = np.zeros(16, dtype=np.float64)
            for ib in range(16):
                sc, Li = _make_qx_quants_16_32(blk[ib * 16 : (ib + 1) * 16])
                sub[ib] = sc
                L[ib * 16 : (ib + 1) * 16] = Li

            max_scale, max_abs = 0.0, 0.0
            for v in sub:
                if abs(v) > max_abs:
                    max_abs, max_scale = abs(v), float(v)
            if max_abs < 1e-15:
                continue  # block stays zeroed, d = 0

            iscale = -128.0 / max_scale
            d_block = 1.0 / iscale
            d_f16 = np.float16(d_block).astype(np.float32).item()
            sc8 = np.clip(_nearest_int(iscale * sub), -128, 127).astype(np.int64)

            for j in range(16):
                dd = d_f16 * float(sc8[j])
                if not dd:
                    continue
                seg = blk[j * 16 : (j + 1) * 16]
                L[j * 16 : (j + 1) * 16] = np.clip(_nearest_int(seg / dd), -32, 31) + 32

            ql = np.zeros(128, dtype=np.uint8)
            qh = np.zeros(64, dtype=np.uint8)
            for j in (0, 128):
                base_ql = 0 if j == 0 else 64
                base_qh = 0 if j == 0 else 32
                for l in range(32):
                    q1 = int(L[j + l]) & 0xF
                    q2 = int(L[j + l + 32]) & 0xF
                    q3 = int(L[j + l + 64]) & 0xF
                    q4 = int(L[j + l + 96]) & 0xF
                    ql[base_ql + l] = q1 | (q3 << 4)
                    ql[base_ql + l + 32] = q2 | (q4 << 4)
                    qh[base_qh + l] = ((int(L[j + l]) >> 4) | ((int(L[j + l + 32]) >> 4) << 2)
                                       | ((int(L[j + l + 64]) >> 4) << 4) | ((int(L[j + l + 96]) >> 4) << 6))
            o = out[r, s * 210 : (s + 1) * 210]
            o[0:128] = ql
            o[128:192] = qh
            o[192:208] = (sc8.astype(np.int8).view(np.uint8))
            o[208:210] = np.frombuffer(_fp16_bytes(np.array([d_block])), dtype=np.uint8)

    return out


def dequantize_q6_K(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize Q6_K superblock bytes (self-verification mirror)."""
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_super = raw.shape[1] // 210
    out = np.zeros((n_rows, n_super * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for s in range(n_super):
            o = raw[r, s * 210 : (s + 1) * 210]
            ql, qh = o[0:128], o[128:192]
            sc = o[192:208].astype(np.int8).astype(np.float32)
            d = float(f16(o[208:210])[0])
            for n in (0, 128):
                bq = 0 if n == 0 else 64
                bh = 0 if n == 0 else 32
                bs = 0 if n == 0 else 8
                for l in range(32):
                    is_ = l // 16
                    q1 = int((ql[bq + l] & 0xF) | (((qh[bh + l] >> 0) & 3) << 4)) - 32
                    q2 = int((ql[bq + l + 32] & 0xF) | (((qh[bh + l] >> 2) & 3) << 4)) - 32
                    q3 = int((ql[bq + l] >> 4) | (((qh[bh + l] >> 4) & 3) << 4)) - 32
                    q4 = int((ql[bq + l + 32] >> 4) | (((qh[bh + l] >> 6) & 3) << 4)) - 32
                    out[r, s * 256 + n + l] = d * sc[bs + is_] * q1
                    out[r, s * 256 + n + 32 + l] = d * sc[bs + is_ + 2] * q2
                    out[r, s * 256 + n + 64 + l] = d * sc[bs + is_ + 4] * q3
                    out[r, s * 256 + n + 96 + l] = d * sc[bs + is_ + 6] * q4
    return out


def quantize_q2_K(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF Q2_K superblocks (256 elems -> 84 bytes).

    Format mirror of ggml ``quantize_row_q2_K``: 16 sub-blocks of 16 with
    min/max grid estimation (``x = d*sc*q - dmin*m``), 4-bit second-level
    scale/min nibbles, 2-bit codes. Layout per superblock: scales(16),
    qs(64), d(2), dmin(2).

    Returns uint8 array shaped (n_rows, n_superblocks_per_row * 84).
    Input last dim must be a multiple of 256 (see :func:`can_pack_kquant`).
    """
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"Q2_K needs last dim % 256 == 0, got {n_cols}"
    n_super = n_cols // 256
    out = np.zeros((n_rows, n_super * 84), dtype=np.uint8)

    for r in range(n_rows):
        for s in range(n_super):
            blk = flat_rows[r, s * 256 : (s + 1) * 256].reshape(16, 16)
            mn = np.minimum(blk.min(axis=1), 0.0)
            mx = blk.max(axis=1)
            rng = (mx - mn).astype(np.float64)
            sub_s = np.where(rng > 0, rng / 3.0, 0.0)
            sub_m = (-mn).astype(np.float64)

            smax = float(sub_s.max())
            mmax = float(sub_m.max())
            d_block = smax / 15.0 if smax > 0 else 0.0
            m_block = mmax / 15.0 if mmax > 0 else 0.0
            Ls = np.clip(_nearest_int(sub_s / d_block), 0, 15) if d_block > 0 else np.zeros(16, dtype=np.int64)
            Lm = np.clip(_nearest_int(sub_m / m_block), 0, 15) if m_block > 0 else np.zeros(16, dtype=np.int64)
            scales = (Ls | (Lm << 4)).astype(np.uint8)

            d_f16 = np.float16(d_block).astype(np.float32).item()
            dm_f16 = np.float16(m_block).astype(np.float32).item()
            codes = np.zeros(256, dtype=np.int64)
            for j in range(16):
                sc = int(scales[j] & 0xF)
                m = int(scales[j] >> 4)
                d = d_f16 * sc
                if not d:
                    continue
                dm = dm_f16 * m
                codes[j * 16 : (j + 1) * 16] = np.clip(
                    _nearest_int((blk[j] + dm) / d), 0, 3
                )

            qs = np.zeros(64, dtype=np.uint8)
            for j in (0, 128):
                for l in range(32):
                    qs[j // 4 + l] = (int(codes[j + l]) | (int(codes[j + l + 32]) << 2)
                                      | (int(codes[j + l + 64]) << 4) | (int(codes[j + l + 96]) << 6))

            o = out[r, s * 84 : (s + 1) * 84]
            o[0:16] = scales
            o[16:80] = qs
            o[80:82] = np.frombuffer(_fp16_bytes(np.array([d_block])), dtype=np.uint8)
            o[82:84] = np.frombuffer(_fp16_bytes(np.array([m_block])), dtype=np.uint8)

    return out


def dequantize_q2_K(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize Q2_K superblock bytes (self-verification mirror)."""
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_super = raw.shape[1] // 84
    out = np.zeros((n_rows, n_super * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for s in range(n_super):
            o = raw[r, s * 84 : (s + 1) * 84]
            scales = o[0:16]
            qs = o[16:80]
            d = float(f16(o[80:82])[0])
            dm = float(f16(o[82:84])[0])
            # Direct mirror of dequantize_row_q2_K.
            is_ = 0
            for n in (0, 128):
                shift = 0
                qb = n // 4
                for j in range(4):
                    sc0 = int(scales[is_])
                    sc1 = int(scales[is_ + 1])
                    is_ += 2
                    dl0, ml0 = d * (sc0 & 0xF), dm * (sc0 >> 4)
                    dl1, ml1 = d * (sc1 & 0xF), dm * (sc1 >> 4)
                    for l in range(16):
                        out[r, s * 256 + n + j * 32 + l] = (
                            dl0 * ((int(qs[qb + l]) >> shift) & 3) - ml0
                        )
                        out[r, s * 256 + n + j * 32 + 16 + l] = (
                            dl1 * ((int(qs[qb + 16 + l]) >> shift) & 3) - ml1
                        )
                    shift += 2
    return out


def _make_q3_quants_16_4(x: np.ndarray) -> tuple:
    """Faithful port of ggml ``make_q3_quants(16, 4, x, L, do_rmse=true)``.

    Returns (scale, L[16] int array centered at nmax=4, i.e. codes 0..7).
    """
    nmax = 4
    x = np.asarray(x, dtype=np.float64)
    amax, vmax = 0.0, 0.0
    for v in x:
        if abs(v) > amax:
            amax, vmax = abs(v), float(v)
    if amax < 1e-15:
        return 0.0, np.zeros(16, dtype=np.int64)
    iscale = -nmax / vmax
    L = np.clip(_nearest_int(iscale * x), -nmax, nmax - 1)
    w = x * x
    sumlx = float((w * x * L).sum())
    suml2 = float((w * L * L).sum())
    for _ in range(5):
        n_changed = 0
        for i in range(16):
            wi = w[i]
            slx = sumlx - wi * x[i] * L[i]
            if slx > 0:
                sl2 = suml2 - wi * L[i] * L[i]
                new_l = int(np.clip(_nearest_int(x[i] * sl2 / slx), -nmax, nmax - 1)) if slx else int(L[i])
                if new_l != L[i]:
                    s2x = slx + wi * x[i] * new_l
                    s22 = sl2 + wi * new_l * new_l
                    if s22 > 0 and s2x * s2x * suml2 > sumlx * sumlx * s22:
                        L[i], sumlx, suml2 = new_l, s2x, s22
                        n_changed += 1
        if not n_changed:
            break
    return (sumlx / suml2 if suml2 > 0 else 0.0), L + nmax


def quantize_q3_K(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF Q3_K superblocks (256 elems -> 110 bytes).

    Format mirror of ggml ``quantize_row_q3_K``: 16 sub-blocks of 16 with
    ``make_q3_quants`` scales, 6-bit packed sub-scales under one fp16
    super-scale (``x = d*sc*(q - 4)`` with the low bit split into hmask),
    2-bit codes. Layout per superblock: hmask(32), qs(64), scales(12), d(2).

    Returns uint8 array shaped (n_rows, n_superblocks_per_row * 110).
    Input last dim must be a multiple of 256 (see :func:`can_pack_kquant`).
    """
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"Q3_K needs last dim % 256 == 0, got {n_cols}"
    n_super = n_cols // 256
    out = np.zeros((n_rows, n_super * 110), dtype=np.uint8)

    for r in range(n_rows):
        for s in range(n_super):
            blk = flat_rows[r, s * 256 : (s + 1) * 256]
            L = np.zeros(256, dtype=np.int64)
            sub = np.zeros(16, dtype=np.float64)
            for j in range(16):
                sc, Lj = _make_q3_quants_16_4(blk[j * 16 : (j + 1) * 16])
                sub[j] = sc
                L[j * 16 : (j + 1) * 16] = Lj

            amax, max_scale = 0.0, 0.0
            for v in sub:
                if abs(v) > amax:
                    amax, max_scale = abs(v), float(v)

            scales = np.zeros(12, dtype=np.uint8)
            if max_scale:
                iscale = -32.0 / max_scale
                d_block = 1.0 / iscale
                for j in range(16):
                    l = int(np.clip(_nearest_int(iscale * sub[j]), -32, 31)) + 32
                    if j < 8:
                        scales[j] = l & 0xF
                    else:
                        scales[j - 8] |= ((l & 0xF) << 4)
                    scales[j % 4 + 8] |= ((l >> 4) << (2 * (j // 4)))
            else:
                d_block = 0.0
            d_f16 = np.float16(d_block).astype(np.float32).item()

            for j in range(16):
                low = int(scales[j] & 0xF) if j < 8 else int(scales[j - 8] >> 4)
                sc = (low | ((((int(scales[8 + j % 4]) >> (2 * (j // 4))) & 3) << 4))) - 32
                dd = d_f16 * sc
                if not dd:
                    continue
                seg = blk[j * 16 : (j + 1) * 16]
                L[j * 16 : (j + 1) * 16] = np.clip(_nearest_int(seg / dd), -4, 3) + 4

            hmask = np.zeros(32, dtype=np.uint8)
            for j in range(256):
                if L[j] > 3:
                    hmask[j % 32] |= (1 << (j // 32))
                    L[j] -= 4

            qs = np.zeros(64, dtype=np.uint8)
            for j in (0, 128):
                for l in range(32):
                    qs[j // 4 + l] = (int(L[j + l]) | (int(L[j + l + 32]) << 2)
                                      | (int(L[j + l + 64]) << 4) | (int(L[j + l + 96]) << 6))

            o = out[r, s * 110 : (s + 1) * 110]
            o[0:32] = hmask
            o[32:96] = qs
            o[96:108] = scales
            o[108:110] = np.frombuffer(_fp16_bytes(np.array([d_block])), dtype=np.uint8)

    return out


def dequantize_q3_K(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize Q3_K superblock bytes (self-verification mirror)."""
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_super = raw.shape[1] // 110
    out = np.zeros((n_rows, n_super * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for s in range(n_super):
            o = raw[r, s * 110 : (s + 1) * 110]
            hm = o[0:32]
            qs = o[32:96]
            scb = o[96:108]
            d = float(f16(o[108:110])[0])
            sc = np.zeros(16, dtype=np.float64)
            for j in range(16):
                low = int(scb[j] & 0xF) if j < 8 else int(scb[j - 8] >> 4)
                sc[j] = (low | ((((int(scb[8 + j % 4]) >> (2 * (j // 4))) & 3) << 4))) - 32
            is_ = 0
            for n in (0, 128):
                shift = 0
                m = 1 << (n // 32)
                qb = n // 4
                for j in range(4):
                    dl0 = d * sc[is_]
                    dl1 = d * sc[is_ + 1]
                    is_ += 2
                    for l in range(16):
                        c0 = (int(qs[qb + l]) >> shift) & 3
                        c1 = (int(qs[qb + 16 + l]) >> shift) & 3
                        v0 = dl0 * (c0 - (0 if hm[l] & m else 4))
                        v1 = dl1 * (c1 - (0 if hm[16 + l] & m else 4))
                        out[r, s * 256 + n + j * 32 + l] = v0
                        out[r, s * 256 + n + j * 32 + 16 + l] = v1
                    shift += 2
                    m <<= 1
    return out


def quantize_iq2_xxs(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF IQ2_XXS blocks (256 elems -> 66 bytes).

    Format mirror of ggml ``quantize_row_iq2_xxs``: 8 groups of 32, each a
    single scale nibble over 4 grid-coded 8-groups. Per 8-group the encoder
    folds signs (with the reference's parity fix) and searches all 256
    ``IQ2XXS_GRID`` rows for the least-squares fit; the 32-group scale is
    the joint least-squares fit, nibble-quantized exactly per the reference
    (``l = round(0.5*(scale/d - 1))``, ``db = d*(0.5+l)*0.25``).
    Layout per block: d(2), qs(64) as 32 little-endian uint16 with grid
    indices in the low bytes, sign fragments in the high bytes and the
    scale nibble in the top 4 bits (see reference for the bit packing).

    The reference uses importance weights in its search; this port fits
    unweighted (same format, small quality gap, still far below Q2_K).

    Returns uint8 array shaped (n_rows, n_blocks_per_row * 66).
    Input last dim must be a multiple of 256.
    """
    from autogsq.packer.iq_tables import IQ2XXS_GRID, KMASK_IQ2XS

    G = IQ2XXS_GRID.astype(np.float64)  # (256, 8)
    G2 = (G * G).sum(axis=1)  # (256,)
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"IQ2_XXS needs last dim % 256 == 0, got {n_cols}"
    n_blocks = n_cols // 256
    out = np.zeros((n_rows, n_blocks * 66), dtype=np.uint8)

    for r in range(n_rows):
        for b in range(n_blocks):
            blk = flat_rows[r, b * 256 : (b + 1) * 256].astype(np.float64)
            bsigns = np.zeros(32, dtype=np.int64)  # 7-bit sign idx per 8-group
            gscales = np.zeros(8, dtype=np.float64)  # joint scale per 32-group
            grid8 = np.zeros(32, dtype=np.int64)  # grid row per 8-group
            xvals = np.zeros((8, 4, 8), dtype=np.float64)

            for ib in range(8):
                seg = blk[ib * 32 : (ib + 1) * 32]
                for k in range(4):
                    g8 = seg[k * 8 : (k + 1) * 8]
                    neg = g8 < 0
                    xv = np.abs(g8)
                    s = 0
                    for i in range(8):
                        if neg[i]:
                            s |= (1 << i)
                    nflip = int(neg.sum())
                    if nflip % 2:
                        imin = int(np.argmin(xv * xv))
                        xv[imin] = -xv[imin]
                        s ^= (1 << imin)
                    xvals[ib, k] = np.abs(xv)
                    bsigns[ib * 4 + k] = s & 127

                # Exhaustive grid search per 8-group at free scale.
                best_rows = np.zeros(4, dtype=np.int64)
                for k in range(4):
                    xv = xvals[ib, k]
                    if xv.max() < 1e-15:
                        best_rows[k] = 0
                        continue
                    xg = xv @ G.T  # (256,)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s_opt = np.where(G2 > 0, xg / G2, 0.0)
                    x2 = float(xv @ xv)
                    err = x2 - s_opt * xg
                    best_rows[k] = int(np.argmin(err))
                # Joint least-squares scale over the 32-group.
                num, den = 0.0, 0.0
                for k in range(4):
                    g = G[best_rows[k]]
                    xv = xvals[ib, k]
                    num += float(xv @ g)
                    den += float(g @ g)
                scale = num / den if den > 0 else 0.0
                # One refine round: re-pick rows at the joint scale, re-fit.
                if scale > 0:
                    for k in range(4):
                        xv = xvals[ib, k]
                        err = ((xv[None, :] - scale * G) ** 2).sum(axis=1)
                        best_rows[k] = int(np.argmin(err))
                    num, den = 0.0, 0.0
                    for k in range(4):
                        g = G[best_rows[k]]
                        xv = xvals[ib, k]
                        num += float(xv @ g)
                        den += float(g @ g)
                    scale = num / den if den > 0 else 0.0
                gscales[ib] = scale
                grid8[ib * 4 : ib * 4 + 4] = best_rows

            max_scale = float(gscales.max())
            if max_scale <= 0:
                continue  # block stays zeroed
            d_block = max_scale / 31.0
            id_ = 1.0 / d_block

            # Explicit byte-level pack per the reference dequant layout:
            # bytes 8*ib..8*ib+3 = grid idx; bytes 4..7 of each 8-slice hold
            # sign fragments (7 bits each, little-endian) + scale nibble.
            raw = np.zeros(64, dtype=np.uint8)
            for ib in range(8):
                l = int(np.clip(round(0.5 * (id_ * gscales[ib] - 1)), 0, 15))
                for k in range(4):
                    raw[8 * ib + k] = int(grid8[ib * 4 + k]) & 0xFF
                w1 = 0
                for k in range(4):
                    w1 |= (int(bsigns[ib * 4 + k]) & 127) << (7 * k)
                w1 |= (l & 15) << 28
                raw[8 * ib + 4] = w1 & 0xFF
                raw[8 * ib + 5] = (w1 >> 8) & 0xFF
                raw[8 * ib + 6] = (w1 >> 16) & 0xFF
                raw[8 * ib + 7] = (w1 >> 24) & 0xFF

            o = out[r, b * 66 : (b + 1) * 66]
            o[0:2] = np.frombuffer(_fp16_bytes(np.array([d_block])), dtype=np.uint8)
            o[2:66] = raw

    return out


def dequantize_iq2_xxs(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize IQ2_XXS block bytes (self-verification mirror)."""
    from autogsq.packer.iq_tables import IQ2XXS_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS

    G = IQ2XXS_GRID.astype(np.float32)
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_blocks = raw.shape[1] // 66
    out = np.zeros((n_rows, n_blocks * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for b in range(n_blocks):
            o = raw[r, b * 66 : (b + 1) * 66]
            d = float(f16(o[0:2])[0])
            qs = o[2:66]
            for ib in range(8):
                aux = qs[8 * ib : 8 * ib + 8]
                aux32_1 = (int(aux[4]) | (int(aux[5]) << 8) | (int(aux[6]) << 16) | (int(aux[7]) << 24))
                db = d * (0.5 + (aux32_1 >> 28)) * 0.25
                for l in range(4):
                    grid = G[int(aux[l])]
                    signs = int(KSIGNS_IQ2XS[(aux32_1 >> (7 * l)) & 127])
                    for j in range(8):
                        s = -1.0 if (signs & int(KMASK_IQ2XS[j])) else 1.0
                        out[r, b * 256 + ib * 32 + l * 8 + j] = db * grid[j] * s
    return out


def quantize_iq1_s(data: np.ndarray) -> np.ndarray:
    """Quantize float array into GGUF IQ1_S blocks (256 elems -> 50 bytes).

    Format mirror of ggml ``quantize_row_iq1_s``: 8 groups of 32, each a
    3-bit scale selector plus sign over 4 grid-coded 8-groups. Grids are
    ternary {-1, 0, 1} rows with a +/-0.125 delta shift
    (``y = dl*(grid + delta)``, ``dl = d*(2*l+1)``). Per 8-group the
    encoder searches all 2048 ``IQ1S_GRID`` rows (least-squares, both
    delta signs); the 32-group scale is the joint fit; the global ``d``
    carries the reference's 1.125 fudge factor while the nibble base does
    not — exactly as in the reference. Layout per block: d(2), qs(32),
    qh(8 x uint16: row-high-bits in 0..11, scale selector in 12..14,
    delta sign in bit 15).

    The reference uses importance weights and a sorted-split ternary
    pre-pass with neighbour search; this port fits unweighted exhaustive
    (same format; an importance-weighted variant scored worse on uniform
    fidelity, so unweighted stands pending PPL-arbitrated tuning).

    Returns uint8 array shaped (n_rows, n_blocks_per_row * 50).
    Input last dim must be a multiple of 256.
    """
    from autogsq.packer.iq1s_table import IQ1S_GRID, IQ1S_DELTA

    G = IQ1S_GRID.astype(np.float64)  # (2048, 8)
    Gd = {s: G + s * IQ1S_DELTA for s in (1.0, -1.0)}
    Gn = {s: (Gd[s] * Gd[s]).sum(axis=1) for s in (1.0, -1.0)}
    arr = np.asarray(data, dtype=np.float32)
    flat_rows = arr.reshape(-1, arr.shape[-1])
    n_rows, n_cols = flat_rows.shape
    assert n_cols % 256 == 0, f"IQ1_S needs last dim % 256 == 0, got {n_cols}"
    n_blocks = n_cols // 256
    out = np.zeros((n_rows, n_blocks * 50), dtype=np.uint8)

    for r in range(n_rows):
        for b in range(n_blocks):
            blk = flat_rows[r, b * 256 : (b + 1) * 256].astype(np.float64)
            rows = np.zeros((8, 4), dtype=np.int64)
            gscales = np.zeros(8, dtype=np.float64)
            gshifts = np.ones(8, dtype=np.float64)

            for ib in range(8):
                seg = blk[ib * 32 : (ib + 1) * 32].reshape(4, 8)
                best = (float("inf"), None, None, None)  # err, rows, scale, shift
                for shift in (1.0, -1.0):
                    M = Gd[shift]
                    N = Gn[shift]
                    cand = np.zeros(4, dtype=np.int64)
                    for k in range(4):
                        xv = seg[k]
                        if np.abs(xv).max() < 1e-12:
                            cand[k] = 0
                            continue
                        xg = xv @ M.T
                        s_opt = xg / np.where(N > 0, N, 1.0)
                        x2 = float(xv @ xv)
                        cand[k] = int(np.argmin(x2 - s_opt * xg))
                    num = sum(float(seg[k] @ M[cand[k]]) for k in range(4))
                    den = sum(float(N[cand[k]]) for k in range(4))
                    scale = num / den if den > 0 else 0.0
                    if scale > 0:
                        for k in range(4):
                            err = ((seg[k][None, :] - scale * M) ** 2).sum(axis=1)
                            cand[k] = int(np.argmin(err))
                        num = sum(float(seg[k] @ M[cand[k]]) for k in range(4))
                        den = sum(float(N[cand[k]]) for k in range(4))
                        scale = num / den if den > 0 else 0.0
                    err = sum(float(((seg[k] - scale * M[cand[k]]) ** 2).sum()) for k in range(4))
                    if err < best[0]:
                        best = (err, cand.copy(), scale, shift)
                rows[ib] = best[1]
                gscales[ib] = best[2]
                gshifts[ib] = best[3]

            max_scale = float(gscales.max())
            if max_scale <= 0:
                continue  # block stays zeroed
            d_raw = max_scale / 15.0
            id_ = 1.0 / d_raw
            d_f16 = np.float16(d_raw * 1.125).astype(np.float32).item()

            qs = np.zeros(32, dtype=np.uint8)
            qh = np.zeros(8, dtype=np.uint16)
            for ib in range(8):
                l = int(np.clip(round(0.5 * (id_ * gscales[ib] - 1)), 0, 7))
                dl_actual = d_f16 * (2 * l + 1)
                M = Gd[gshifts[ib]]
                h = 0
                for k in range(4):
                    xv = blk[ib * 32 + k * 8 : ib * 32 + (k + 1) * 8]
                    if dl_actual > 0:
                        err = ((xv[None, :] - dl_actual * M) ** 2).sum(axis=1)
                        row = int(np.argmin(err))
                    else:
                        row = 0
                    rows[ib, k] = row
                    qs[4 * ib + k] = row & 0xFF
                    h |= ((row >> 8) & 7) << (3 * k)
                if gshifts[ib] < 0:
                    l |= 8
                qh[ib] = np.uint16((h | (l << 12)) & 0xFFFF)

            o = out[r, b * 50 : (b + 1) * 50]
            o[0:2] = np.frombuffer(_fp16_bytes(np.array([d_raw * 1.125])), dtype=np.uint8)
            o[2:34] = qs
            o[34:50] = qh.view(np.uint8)

    return out


def dequantize_iq1_s(packed: np.ndarray, n_rows: int) -> np.ndarray:
    """Dequantize IQ1_S block bytes (self-verification mirror)."""
    from autogsq.packer.iq1s_table import IQ1S_GRID, IQ1S_DELTA

    G = IQ1S_GRID.astype(np.float32)
    raw = np.asarray(packed, dtype=np.uint8).reshape(n_rows, -1)
    n_blocks = raw.shape[1] // 50
    out = np.zeros((n_rows, n_blocks * 256), dtype=np.float32)
    f16 = lambda b: np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)
    for r in range(n_rows):
        for b in range(n_blocks):
            o = raw[r, b * 50 : (b + 1) * 50]
            d = float(f16(o[0:2])[0])
            qs = o[2:34]
            qh = o[34:50].view("<u2").astype(np.int64)
            for ib in range(8):
                dl = d * (2 * ((qh[ib] >> 12) & 7) + 1)
                delta = -IQ1S_DELTA if (qh[ib] & 0x8000) else IQ1S_DELTA
                for l in range(4):
                    row = int(qs[4 * ib + l]) | ((int((qh[ib] >> (3 * l)) & 7)) << 8)
                    out[r, b * 256 + ib * 32 + l * 8 : b * 256 + ib * 32 + (l + 1) * 8] = (
                        dl * (G[row] + delta)
                    )
    return out


# Quant types implemented natively in this module (beyond gguf-py's set).
OUR_PACKERS = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ1_S")
_K_TYPES = set(OUR_PACKERS)


def repack_tensor_to_gguf(
    weight_tensor: Union[torch.Tensor, np.ndarray],
    gguf_type_name: str,
) -> Tuple[np.ndarray, int, str]:
    """Repack tensor into the closest GGUF layout this writer can emit.

    Args:
        weight_tensor: original or dequantized weight tensor
        gguf_type_name: requested GGUF quant type name (e.g. 'Q8_0', 'IQ2_XXS')

    Returns:
        (packed_numpy_array, ggml_type_code, actual_type_name). When the
        requested type has no numpy implementation here, the nearest writable
        type is used and reported via ``actual_type_name`` so manifests stay
        honest. True IQ block packing is a documented next step.
    """
    if isinstance(weight_tensor, torch.Tensor):
        arr = weight_tensor.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(weight_tensor, dtype=np.float32)

    requested = gguf_type_name.upper().strip()
    n_cols = int(arr.shape[-1]) if arr.ndim >= 1 else None
    actual = closest_writable_type(requested, n_cols=n_cols)
    type_code = get_ggml_type_code(actual)

    # Native K-quant packers from this module (exact ggml superblock format).
    if actual == "IQ1_S":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_iq1_s(arr), type_code, actual
    if actual == "IQ2_XXS":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_iq2_xxs(arr), type_code, actual
    if actual == "Q2_K":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_q2_K(arr), type_code, actual
    if actual == "Q3_K":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_q3_K(arr), type_code, actual
    if actual == "Q4_K":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_q4_K(arr), type_code, actual
    if actual == "Q5_K":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_q5_K(arr), type_code, actual
    if actual == "Q6_K":
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        return quantize_q6_K(arr), type_code, actual

    # Native gguf quantization when available (F16/F32 cast directly).
    if actual == "F32":
        return arr.astype(np.float32), type_code, actual
    if actual in ("F16", "BF16"):
        return arr.astype(np.float16), get_ggml_type_code(actual), actual
    if gguf is not None and hasattr(gguf, "quants"):
        try:
            quant_enum = GGMLQuantizationType(type_code)
            if hasattr(gguf.quants, "quantize"):
                quantized = gguf.quants.quantize(arr, quant_enum)
                return quantized, type_code, actual
        except Exception:
            pass

    # Custom superblock implementations for standard types
    if actual == "Q8_0":
        return quantize_q8_0(arr), NAME_TO_GGML_TYPE["Q8_0"], actual
    elif actual in ("Q4_0", "IQ4_NL"):
        return quantize_q4_0(arr), NAME_TO_GGML_TYPE["Q4_0"], actual
    elif actual == "F32":
        return arr.astype(np.float32), NAME_TO_GGML_TYPE["F32"], actual
    else:
        # Last resort: Q8_0 is always writable and easy to verify.
        return quantize_q8_0(arr), NAME_TO_GGML_TYPE["Q8_0"], "Q8_0"
