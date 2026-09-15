"""Same-harness PPL for GGUF files: dequantize exact bytes via gguf-py, score in transformers.

Usage:
    autogsq-ppl --gguf <file> --source <hf model dir> [--max-windows N]
Compares fairly across files because every candidate goes through the
identical dequantize -> fp16 -> CUDA forward path.
"""

from pathlib import Path

import numpy as np
import torch
import typer
from rich.console import Console

console = Console()
app = typer.Typer()


def load_gguf_state_dict(gguf_path: Path, source_dir: Path) -> dict:
    import gguf
    from gguf import quants as _q
    from autogsq.packer.hf_gguf_map import map_tensor_names
    from autogsq.common.models import ModelWeightSource

    hf_names = ModelWeightSource(str(source_dir)).names()
    gguf_of = map_tensor_names(hf_names)
    hf_of = {g: h for h, g in gguf_of.items()}
    ref_shapes = {n: tuple(ModelWeightSource(str(source_dir)).get(n).shape) for n in hf_names}

    reader = gguf.GGUFReader(str(gguf_path))
    state: dict = {}
    for t in reader.tensors:
        arr = t.data
        try:
            from gguf import GGMLQuantizationType as QT

            dq = _q.dequantize(np.asarray(arr).reshape(-1), QT(t.tensor_type))
            # Reader dims are stored reversed (matmul order); un-reverse to
            # logical orientation (verified by matvec: cos 0.991 vs source).
            dq = np.asarray(dq, dtype=np.float32).reshape(tuple(int(x) for x in t.shape[::-1]))
        except Exception:
            dq = np.asarray(arr)
            if dq.ndim == 2:
                dq = np.ascontiguousarray(dq.T)
        name = t.name
        if name not in hf_of:
            # Embedding/output may be trimmed to the tokenizer; pad reserves.
            base = name
            if base in ("token_embd.weight", "output.weight"):
                key = (
                    "model.embed_tokens.weight"
                    if base == "token_embd.weight"
                    else "lm_head.weight"
                )
                ref = ModelWeightSource(str(source_dir)).get(key)
                full = np.zeros(tuple(ref.shape), dtype=np.float32)
                n = min(full.shape[0], dq.shape[0])
                full[:n] = dq.reshape(dq.shape[0], -1)[:n].reshape(n, *full.shape[1:])
                state[key] = torch.from_numpy(full)
                continue
            raise KeyError(f"GGUF tensor {name!r} has no HF counterpart")
        key = hf_of[name]
        want = ref_shapes[key]
        if tuple(dq.shape) != tuple(want):
            # Trimmed embeddings (tokenizer reserves): zero-pad rows.
            if (dq.ndim == 2 and len(want) == 2 and dq.shape[1] == want[1]
                    and dq.shape[0] < want[0]):
                full = np.zeros(tuple(want), dtype=np.float32)
                full[: dq.shape[0]] = dq
                dq = full
            else:
                raise RuntimeError(f"{key}: gguf {dq.shape} vs hf {want}")
        state[key] = torch.from_numpy(np.asarray(dq))
    # Sanity: every HF weight present.
    missing = [n for n in hf_names if n not in state]
    if missing:
        raise RuntimeError(f"{len(missing)} HF tensors missing from {gguf_path}: {missing[:5]}")
    return state


@app.command()
def main(
    gguf: Path = typer.Option(..., "--gguf", help="GGUF file to score"),
    source: Path = typer.Option(..., "--source", help="Source HF model dir (names + fallback)"),
    max_length: int = typer.Option(2048, "--max-length"),
    stride: int = typer.Option(512, "--stride"),
    max_windows: int = typer.Option(20, "--max-windows"),
) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print(f"[magenta]Loading + dequantizing {gguf}...[/magenta]")
    state = load_gguf_state_dict(gguf, source)
    # Pre-flight gate: a wrong orientation/mapping must fail loudly, never
    # silently score garbage (cos-sim vs source should be ~1 for F16 tensors).
    from autogsq.common.models import ModelWeightSource as _MWS

    probe = "model.layers.0.self_attn.q_proj.weight"
    if probe in state:
        # Orientation-sensitive gate: matching (64, 64) blocks correlate
        # ~1; a transposed view scores ~0. Flatten must NOT be used here —
        # it cannot detect transposition.
        a = state[probe].float()[:64, :64].flatten()
        b = _MWS(str(source)).get(probe).float()[:64, :64].flatten()
        cos = float((a @ b) / (a.norm() * b.norm()))
        console.print(f"[cyan]mapping check block cos-sim vs source: {cos:.4f}[/cyan]")
        if not cos > 0.9:
            raise RuntimeError(f"mapping check failed (cos-sim {cos:.4f}); refusing to score")
    tok = AutoTokenizer.from_pretrained(str(source), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.load_state_dict(state, strict=False)
    model.tie_weights()  # tied-embedding models ship no lm_head; re-tie after load
    model = model.cuda().eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt")["input_ids"][0]
    nll_sum, n_tokens = 0.0, 0
    start = 0
    windows = 0
    with torch.no_grad():
        import torch.nn.functional as F

        while start < enc.size(0) and windows < max_windows:
            end = min(start + max_length, enc.size(0))
            ids = enc[start:end].unsqueeze(0).cuda()
            out = model(ids)
            # Shifted next-token scoring; first window counts whole,
            # later windows only the fresh stride (standard strided eval).
            logits = out.logits[0, :-1].float().cpu()
            labels = ids[0, 1:].cpu()
            ntarget = len(labels) if start == 0 else min(stride, len(labels))
            nll_sum += float(F.cross_entropy(logits[-ntarget:], labels[-ntarget:], reduction="sum"))
            n_tokens += ntarget
            windows += 1
            if end >= enc.size(0):
                break
            start += stride
    ppl = float(torch.exp(torch.tensor(nll_sum / n_tokens)))
    console.print(f"[green]PPL: {ppl:.4f} over {n_tokens} tokens ({windows} windows)[/green]")
    print(f"PPL_RESULT {ppl:.4f} tokens={n_tokens} windows={windows}")


if __name__ == "__main__":
    app()
