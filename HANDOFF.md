# Handover — AutoGSQ-RCO (2026-09-15 ~16:45 UTC)

## What's running RIGHT NOW
- **Full 36-layer DB build** under Windows Task Scheduler (`schtasks /query /tn "AutoGSQ-DB-Build"`).
  Survives chat session cycles (agent-tracked background procs get SIGTERM'd on
  session migration — this happened twice; scheduler does not).
- Monitor: `.venv/Scripts/autogsq status --db-dir runtime/qwen3_4b_db --log runtime/qwen3_4b_build_sched.log`
- Current state: layers 0–5 done (168 candidates); task is in Hessian-capture preamble.
- The running task launched AFTER the resume fix was committed and the venv
  imports from `src/` directly, so it HAS the fix: expect "already complete,
  skipping" for layers 0–5 when capture phase starts printing. Verified
  pending — grep the sched log for "already complete, skipping".

## When the build finishes (notify/next check)
1. `autogsq allocate --db-dir runtime/qwen3_4b_db --config runtime/qwen3_4b_probe.yaml --target-bpw 2.75 --out runtime/qwen3_4b_alloc_full.json`
2. `autogsq assemble --db-dir runtime/qwen3_4b_db --allocation ... --output runtime/qwen3_4b_probe/model-full.gguf --config runtime/qwen3_4b_probe.yaml -j 16` (~15–20 min)
3. `.venv/Scripts/python src/autogsq/eval/gguf_ppl.py --gguf <file> --source runtime/qwen3_4b_model --max-windows 20`
4. Compare vs official baseline `runtime/baseline/Qwen3-4B-Q4_K_M.gguf` (PPL 14.21) and update README table.

## Roadmap state
- DONE: native Q2_K/Q3_K/Q4_K/Q5_K/Q6_K + IQ2_XXS + IQ1_S (all bit-exact vs gguf-py),
  `assemble -j N` (byte-identical), PPL harness, 36+3 tests green, git repo (5 commits),
  README with honest limitations, resume fix, progress tracking.
- OPEN future work: (a) ternary RCO quality — DB ternary cos 0.66–0.85, fresh-train 0.82,
  RCO picks ternary aggressively; needs full-DB numbers first; (b) IQ encoder tuning —
  importance-weighted IQ1_S variant EXPLODED (PPL 116k) and was reverted; needs
  PPL-arbitrated experiments, never proxy metrics alone.

## Hard-won lessons (do not relearn)
- GGUF reader dims are stored reversed: reshape dequantized flat arrays to `t.shape[::-1]`.
  Flatten cos-sim CANNOT detect transposition — verify orientation with matvec or block compare.
- safetensors returns mmap views: `.clone()` after `safe_open` closes, else segfaults (both loaders).
- Tokenizer: merge `added_tokens` (specials live outside `model.vocab`); merges may be
  `[a,b]` pairs (join them); `tokenizer.ggml.pre` must match family (`qwen2` for Qwen3);
  K-quants need last-dim % 256.
- Ternary picks pack from F16 source (`packed_from: f16_source` in cli/main.py) — single
  quant beats double quant (0.78 vs 0.62 cos). Manifest + README note this.
- Qwen3: use explicit `head_dim` (128, not hidden//heads=80) + key/value length KV.
- pytest chain gotcha: `| tail` masks failures — check `pytest ... ; echo $?` or rely on tool exit codes.
- Terminal runs git-bash; use `C:/`-style paths for native tools; `$LOCALAPPDATA/Temp` for scratch.

## Key files
- Packer: `src/autogsq/packer/calibration_repack.py` (+ `iq_tables.py`, `iq1s_table.py`)
- Harness: `src/autogsq/eval/gguf_ppl.py` | Tests: `tests/test_packer_regression.py`, `tests/test_resume.py`
- Sizes: probe file 7.06 GB (15% params quantized); full build will be bigger.
