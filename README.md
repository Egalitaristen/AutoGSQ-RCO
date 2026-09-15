# AutoGSQ-RCO

> **⚠️ EARLY TESTING — NOT PRODUCTION READY.** This repo is active research code. Measured quality currently trails the official baseline (our probe PPL 42.67 vs official Q4_K_M 14.21), full-model results are still being built, APIs and file formats may change without notice, and generated checkpoints should be validated before any real use. See [Known limitations](#known-limitations-honest) for the honest state of play.

An end-to-end pipeline combining Gumbel-Softmax Quantization (GSQ) and Riemannian Constrained Optimization (RCO) to produce exact-budget, mixed-precision GGUF checkpoints for llama.cpp, Ollama, and LM Studio deployment.

## Pipeline

```
HF model → generate-db → allocate → assemble → .gguf (+ sidecar manifests)
                              ↘ benchmark (wikitext2 PPL, CUDA)
```

1. **generate-db** — builds a per-tensor candidate database over bit-width options (`2, 3, 4, "ternary"`) with GSQ-learned levels, MSE tables, and dequantized weights.
2. **allocate** — budgeted search (coarse static signal → peaked refinement → greedy polish → budget-first finalize; see `budgeted-allocation-search` skill) emitting `per_tensor: {bits, gguf_type}` at an exact `--target-bpw`.
3. **assemble** — repacks each tensor to its GGUF type natively and stitches the file with arch/tokenizer KV, SHA-pinned sidecar manifests (`.rco-allocation.json/.txt`).
4. **benchmark / gguf_ppl** — strided wikitext2 perplexity, including a same-harness scorer that dequantizes GGUF bytes and scores them in transformers.

## Native GGUF types (no substitution fallbacks)

All implemented from the ggml format spec and verified **bit-exact** against gguf-py's independent dequantizers (`tests/test_packer_regression.py`):

| Type | Block | Status |
|---|---|---|
| Q2_K / Q3_K / Q4_K / Q5_K / Q6_K | 84 / 110 / 144 / 176 / 210 B per 256 | native |
| IQ2_XXS (codebook) | 66 B per 256 | native |
| IQ1_S (ternary codebook) | 50 B per 256 | native |
| Q8_0 / Q4_0 / F16 / F32 | — | native |

Files load and generate in llama.cpp via Ollama (verified: Qwen3-4B with 19 IQ1_S + 7 Q3_K + 16 Q4_K tensors).

## Usage

```bash
.venv/Scripts/autogsq generate-db --config configs/examples/qwen3_8b_2p75bpw.yaml
.venv/Scripts/autogsq allocate --db-dir <db> --config <cfg> --target-bpw 2.75 --out alloc.json
.venv/Scripts/autogsq assemble --db-dir <db> --allocation alloc.json \
    --output model.gguf --config <cfg> -j 16
.venv/Scripts/python src/autogsq/eval/gguf_ppl.py --gguf model.gguf --source <hf-dir>
```

`assemble -j N` parallelizes repack over processes (byte-identical output for any N).

## Measured results (Qwen3-4B probe, wikitext2 strided, 20 windows, same harness)

| File | PPL | Size |
|---|---|---|
| ours (19 IQ1_S + 7 Q3_K + 16 Q4_K, rest F16) | 42.67 | 7.06 GB |
| official `Qwen3-4B-Q4_K_M` | 14.21 | ~2.5 GB |

Reproduce: `src/autogsq/eval/gguf_ppl.py --gguf <file> --source runtime/qwen3_4b_model`.

## Known limitations (honest)

- **PPL gap is allocation-side, not packer-side.** Per-type fidelity vs source: Q4_K cos 0.986, Q3_K ~0.95 (healthy); the 19 ternary picks at 1.56 bpw dominate the loss. Ternary DB candidates themselves score 0.66–0.85 cos — generate-db/RCO tuning is the next lever, not the packer.
- **Ternary picks pack from F16 source** (`packed_from: f16_source`), not from ternary candidates: double quantization (0.62 cos) loses to single quantization (0.78 cos). Shipped bytes beat DB-measured MSE (safe direction), noted in code and here.
- **IQ encoders are unweighted exhaustive search.** An importance-weighted IQ1_S variant scored worse on uniform fidelity (0.53 vs 0.78 cos) and exploded PPL when shipped — reverted. Quality tuning here needs PPL-arbitrated experiments, not proxy metrics.
- **K-quants need last-dim % 256**; other shapes fall back honestly (recorded in the manifest).
- **GGUF orientation trap** (for future harness work): reader dims are stored reversed — reshape dequantized flat arrays to `t.shape[::-1]`; flatten cos-sim cannot detect transposition, use matvec or block compare.
