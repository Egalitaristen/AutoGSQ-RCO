"""AutoGSQ-RCO Command Line Interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
from rich.console import Console
import torch

from autogsq.config import load_config
from autogsq.gsq.db_builder import build_candidate_database
from autogsq.common.store import CandidateStore
from autogsq.rco.solver import optimize_budget
from autogsq.packer.gguf_type_map import nearest_gguf_type
from autogsq.packer.gguf_writer import GGUFPacker

app = typer.Typer(
    name="autogsq",
    help="AutoGSQ-RCO: Mixed-precision GSQ candidate generation, RCO allocation, and GGUF assembly.",
    add_completion=False,
)
console = Console()


@app.command("generate-db")
def generate_db(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config YAML file"),
    out_dir: Path = typer.Option("./runtime/db", "--out-dir", "-o", help="Output database directory"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from progress.json"),
    max_layers: Optional[int] = typer.Option(None, "--max-layers", help="Limit number of layers for smoke testing"),
) -> None:
    """Stage 1: Build GSQ candidate database for all layers and bit-width options."""
    console.print(f"[bold green]Loading configuration from {config}...[/bold green]")
    cfg = load_config(config)

    build_candidate_database(
        model_id=cfg.model.name,
        calibration_data=cfg.data.dataset_name,
        bitwidth_options=cfg.gsq.bitwidth_options,
        out_dir=out_dir,
        group_size=cfg.gsq.group_size,
        init_method=cfg.gsq.init_method,
        num_epochs=cfg.gsq.num_epochs,
        device=cfg.model.device,
        max_layers=max_layers,
        resume=resume,
        calib_nsamples=cfg.gsq.gptq_nsamples,
        calib_max_length=cfg.gsq.calib_max_length,
        calib_device=cfg.gsq.calib_device,
        load_dtype=cfg.model.dtype,
        gptq_damping=cfg.gsq.gptq_damping,
        optimizer=cfg.gsq.optimizer,
        lr=cfg.gsq.learning_rate,
        lr_logits=cfg.gsq.lr_logits,
        lr_scales=cfg.gsq.lr_scales,
        weight_decay=cfg.gsq.gs_weight_decay,
        warmup_ratio=cfg.gsq.warmup_ratio,
        steps_per_epoch=cfg.gsq.steps_per_epoch,
        temp_range=tuple(cfg.gsq.temperature),
        scale_range=tuple(cfg.gsq.logit_scale),
        logits_dtype=cfg.gsq.logits_dtype,
    )
    console.print(f"[bold green]Candidate database ready at: {out_dir}[/bold green]")


# Effective bits-per-weight used for budget math. Non-integer grids map to
# their GGUF-nearest effective rate; the original labels are preserved
# verbatim in manifests because DB filenames use them (e.g. ".ternary.pth").
EFFECTIVE_BPW: Dict[str, float] = {"ternary": 1.6}
# Deterministic column order for the shared option set.
OPTION_PREFERENCE = ("2", "3", "4", "ternary")


def _effective_bpw(label: str) -> float:
    """Budget cost of an option label (numeric bits or a named grid)."""
    if label in EFFECTIVE_BPW:
        return EFFECTIVE_BPW[label]
    return float(label)


@app.command("backfill-mse")
def backfill_mse_cmd(
    db_dir: Path = typer.Option(..., "--db-dir", "-d", help="Candidate database directory"),
    model_source: Optional[Path] = typer.Option(
        None, "--model-source", "-m", help=".safetensors file or directory of shards with original weights"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="HuggingFace model ID (fallback weight source)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Recompute MSE even where already stored"),
) -> None:
    """Add per-candidate reconstruction MSE to an existing database (needed by allocate)."""
    from autogsq.gsq.backfill_mse import backfill_mse as _backfill

    console.print(f"[bold cyan]Backfilling reconstruction MSE for DB at {db_dir}...[/bold cyan]")
    summary = _backfill(db_dir, model_source=model_source, model_id=model, overwrite=overwrite)
    console.print(
        f"[bold green]Backfilled: {summary['backfilled']}, "
        f"skipped (already present): {summary['skipped']}[/bold green]"
    )
    for err in summary["errors"][:10]:
        console.print(f"[bold red]  error: {err}[/bold red]")
    if summary["errors"]:
        raise typer.Exit(code=1)


@app.command("allocate")
def allocate(
    db_dir: Path = typer.Option(..., "--db-dir", "-d", help="Candidate database directory"),
    config: Path = typer.Option(..., "--config", "-c", help="Path to config YAML file"),
    target_bpw: Optional[float] = typer.Option(None, "--target-bpw", help="Target BPW (overrides config)"),
    out: Path = typer.Option(Path("allocation.json"), "--out", "-o", help="Output allocation JSON path"),
) -> None:
    """Stage 2: RCO manifold optimization to find exact-budget bit-width allocation."""
    cfg = load_config(config)
    bpw = target_bpw if target_bpw is not None else cfg.rco.target_bpw
    console.print(f"[bold cyan]Running RCO budget optimization targeting {bpw:.3f} bpw...[/bold cyan]")

    store = CandidateStore(db_dir)
    progress = store.load_progress()
    candidates = progress.get("completed_candidates", [])

    # Discover tensors and bit-width options actually present in the database.
    tensor_options_map: Dict[str, List[str]] = {}
    for entry in candidates:
        if "::" in entry:
            tensor_name, bits = entry.split("::", 1)
            opts = tensor_options_map.setdefault(tensor_name, [])
            if bits not in opts:
                opts.append(bits)

    if not tensor_options_map:
        console.print(
            f"[bold red]No completed candidates found in DB at {db_dir}. "
            "Run `generate-db` first.[/bold red]"
        )
        raise typer.Exit(code=1)

    # The RCO solver works on a shared option set: use the options available
    # for every tensor (dense DBs built by generate-db satisfy this exactly).
    common_options = set.intersection(*(set(v) for v in tensor_options_map.values()))
    if not common_options:
        console.print("[bold red]Tensors share no common bit-width option; cannot allocate.[/bold red]")
        raise typer.Exit(code=1)
    partial = sorted(t for t, v in tensor_options_map.items() if set(v) != common_options)
    if partial:
        console.print(
            f"[bold yellow]{len(partial)} tensor(s) miss some options; "
            f"allocating over the common set {sorted(common_options)}.[/bold yellow]"
        )
    option_labels = [o for o in OPTION_PREFERENCE if o in common_options] + sorted(
        o for o in common_options if o not in OPTION_PREFERENCE
    )

    group_names = sorted(tensor_options_map.keys())
    costs = torch.tensor([_effective_bpw(o) for o in option_labels], dtype=torch.float64)

    # Parameter weights from qparams shapes; fidelity prefers the
    # activation-space error (mse_act) when every candidate carries it.
    weights_list: List[float] = []
    mse_rows: List[List[float]] = []
    mse_act_rows: List[List[float]] = []
    act_complete = True
    for name in group_names:
        row: List[float] = []
        act_row: List[float] = []
        params: Optional[int] = None
        for opt in option_labels:
            qp = store.load_qparams(name, opt)
            shape = qp.get("shape")
            n_params = int(shape[0] * shape[1]) if shape else None
            if params is None:
                params = n_params
            if "mse" not in qp:
                console.print(
                    f"[bold red]Candidate {name}::{opt} has no reconstruction MSE. "
                    "Run `backfill-mse` for this DB (or rebuild it) first.[/bold red]"
                )
                raise typer.Exit(code=1)
            row.append(float(qp["mse"]))
            if "mse_act" in qp:
                act_row.append(float(qp["mse_act"]))
            else:
                act_complete = False
        weights_list.append(float(params if params else 0))
        mse_rows.append(row)
        mse_act_rows.append(act_row)
    weights = torch.tensor(weights_list, dtype=torch.float64)
    fidelity_key = "mse_act" if act_complete else "mse"
    fid_rows = mse_act_rows if act_complete else mse_rows
    console.print(f"[bold cyan]Phase-A fidelity signal: {fidelity_key} ({len(group_names)} tensors).[/bold cyan]")

    min_bpw = float(torch.min(costs).item())
    if bpw < min_bpw - 1e-9:
        console.print(
            f"[bold yellow]Target {bpw:.3f} bpw is below the minimum achievable "
            f"({min_bpw:.3f}); allocation will use the cheapest option everywhere.[/bold yellow]"
        )

    # Real fidelity loss: expected reconstruction error under option probabilities.
    mse_mat = torch.tensor(fid_rows, dtype=torch.float32)

    def fidelity_loss(alpha: torch.Tensor):
        alpha_var = alpha.detach().clone().requires_grad_(True)
        p = torch.softmax(alpha_var, dim=-1)
        loss = torch.sum(p * mse_mat.to(device=alpha_var.device, dtype=alpha_var.dtype))
        grad = torch.autograd.grad(loss, alpha_var)[0]
        return loss.detach(), grad

    result = optimize_budget(
        weights=weights,
        costs=costs,
        target_bpw=bpw,
        loss_fn=fidelity_loss,
        group_names=group_names,
        num_steps=cfg.rco.num_steps,
        lr=cfg.rco.lr,
        temperature_start=cfg.rco.temperature,
        temperature_end=cfg.rco.temperature_min,
        option_labels=option_labels,
        temperature_schedule=cfg.rco.temperature_schedule,
    )
    eval_info: Dict[str, Any] = {"fidelity": fidelity_key, "phase_a_loss": result["best_loss"]}

    # Phase B: task-loss refinement on the soft-mixed model (reference RCO).
    if cfg.rco.refine_steps > 0:
        from autogsq.common.models import load_model_and_tokenizer
        from autogsq.rco.task_loss import TaskLossRefiner

        refine_dev = torch.device(
            cfg.model.device if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu"
        )
        refine_dtype = "bfloat16" if refine_dev.type == "cuda" else "float32"
        console.print(
            f"[bold cyan]Phase-B task-loss refinement: {cfg.rco.refine_steps} steps on {refine_dev}...[/bold cyan]"
        )
        r_model, r_tokenizer = load_model_and_tokenizer(
            cfg.model.name, device=str(refine_dev), dtype=refine_dtype
        )
        from autogsq.common.models import get_calibration_batches

        r_batches = get_calibration_batches(
            cfg.rco.eval_dataset,
            r_tokenizer,
            num_samples=max(cfg.rco.refine_batches * 4, 4),
            max_length=cfg.rco.refine_max_length,
            device=str(refine_dev),
        )
        refiner = TaskLossRefiner(
            r_model, store, group_names, option_labels, r_batches, refine_dev,
            batches_per_step=cfg.rco.refine_batches,
        )

        def task_loss_fn(alpha: torch.Tensor):
            loss_t, grad_t = refiner(alpha)
            return loss_t.detach(), grad_t.to(device=alpha.device, dtype=alpha.dtype)

        # Peaked warm start around the Phase-A answer: refinement nudges a
        # near-budget discrete point instead of re-exploring from scratch.
        # Polish is off here (each eval costs full forward passes).
        n_opts = len(option_labels)
        peaked = torch.full((len(group_names), n_opts), -2.0)
        for i, k in enumerate(result["assignment_indices"]):
            peaked[i, int(k)] = 2.0

        result = optimize_budget(
            weights=weights,
            costs=costs,
            target_bpw=bpw,
            loss_fn=task_loss_fn,
            group_names=group_names,
            num_steps=cfg.rco.refine_steps,
            lr=cfg.rco.lr,
            temperature_start=cfg.rco.temperature,
            temperature_end=cfg.rco.temperature_min,
            option_labels=option_labels,
            alpha_init=peaked,
            temperature_schedule=cfg.rco.temperature_schedule,
            polish=False,
        )
        eval_info["task_loss"] = result["best_loss"]
        del r_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    console.print(f"[bold green]Achieved BPW: {result['achieved_bpw']:.4f} (target: {bpw})[/bold green]")

    # Build JSON output, preserving DB labels so candidate files resolve exactly.
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "target_bpw": bpw,
        "achieved_bpw": result["achieved_bpw"],
        "model_id": cfg.model.name,
        "options": option_labels,
        "per_tensor": {},
        "eval": eval_info,
    }

    for name, info in result["per_group"].items():
        label = info.get("label", str(info["bits"]))
        gguf_type_val = nearest_gguf_type(
            label,
            tolerance=cfg.gguf.gguf_type_tolerance_bpw,
            fallback=cfg.gguf.fallback_on_no_match,
        )
        opt_idx = info["option_idx"]
        manifest["per_tensor"][name] = {
            "bits": label,
            "effective_bpw": info["bits"],
            "gguf_type": gguf_type_val,
            "params": info["params"],
            "mse": fid_rows[group_names.index(name)][opt_idx],
        }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[bold green]Allocation saved to {out}[/bold green]")
    for name in group_names:
        t = manifest["per_tensor"][name]
        bits_str = f"{t['bits']} bits" if str(t["bits"]).replace(".", "", 1).isdigit() else str(t["bits"])
        console.print(f"  {name}: {bits_str} -> {t['gguf_type']} (mse={t['mse']:.6f})")


def _normalize_db_label(bits: Any) -> str:
    """Map a manifest `bits` value back to the exact DB filename label."""
    if isinstance(bits, float) and bits.is_integer():
        return str(int(bits))
    return str(bits)


@app.command("assemble")
def assemble(
    db_dir: Path = typer.Option(..., "--db-dir", "-d", help="Candidate database directory"),
    allocation: Path = typer.Option(..., "--allocation", "-a", help="Path to allocation JSON file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output .gguf file path"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config YAML path"),
    model_src: Optional[str] = typer.Option(
        None, "--model-src", help="Model weight source (overrides config/model manifest; file, dir, or HF ID)"
    ),
    arch: str = typer.Option("llama", "--arch", help="GGUF architecture tag"),
    jobs: int = typer.Option(
        1, "--jobs", "-j", help="Parallel repack workers (processes over tensors; output bytes are identical for any value)"
    ),
) -> None:
    """Stage 3: Assemble GGUF file from RCO allocation and candidate database."""
    from autogsq.common.models import ModelWeightSource

    console.print(f"[bold magenta]Assembling GGUF checkpoint to {output}...[/bold magenta]")

    with open(allocation, "r", encoding="utf-8") as f:
        alloc_data = json.load(f)

    store = CandidateStore(db_dir)
    per_tensor_alloc = alloc_data.get("per_tensor", {})

    # Resolve the source model for tensors outside the allocation (embeddings,
    # norms, unquantized layers, ...). Without this the GGUF would be partial.
    src = model_src
    if src is None and config is not None:
        src = load_config(config).model.name
    if src is None:
        src = alloc_data.get("model_id", "autogsq_model")
    weights = ModelWeightSource(str(src))
    try:
        model_tensors = weights.names()
    except Exception as exc:
        console.print(f"[bold red]Cannot read source model weights from {src!r}: {exc}[/bold red]")
        raise typer.Exit(code=1)
    if not model_tensors:
        console.print(f"[bold red]Source model at {src!r} contains no tensors.[/bold red]")
        raise typer.Exit(code=1)

    tensors = {}
    packed_alloc: Dict[str, Dict[str, Any]] = {}
    gguf_cfg = load_config(config).gguf if config is not None else None
    tol = gguf_cfg.gguf_type_tolerance_bpw if gguf_cfg else 0.15
    fallback = gguf_cfg.fallback_on_no_match if gguf_cfg else "round_down"
    for name in model_tensors:
        if name in per_tensor_alloc:
            t_info = per_tensor_alloc[name]
            label = _normalize_db_label(t_info.get("bits", 4))
            try:
                tensors[name] = store.load_dequant_weight(name, label)
            except FileNotFoundError:
                console.print(
                    f"[bold red]Chosen candidate {name}::{label} is missing from {db_dir}. "
                    "Re-run `generate-db` (or fix the allocation file).[/bold red]"
                )
                raise typer.Exit(code=1)
            packed_alloc[name] = {
                "bits": label,
                "gguf_type": t_info.get("gguf_type") or nearest_gguf_type(label, tolerance=tol, fallback=fallback),
            }
            # Ternary DB candidates hold another format's optimum (ternary
            # levels), a provably worse starting point for IQ grids than the
            # source weights (measured 0.62 vs 0.78 cos on down_proj). Pack
            # ternary picks from F16 source: single quantization, and the
            # shipped bytes beat the DB-measured MSE (safe direction).
            if label == "ternary":
                tensors[name] = weights.get(name)
                packed_alloc[name]["packed_from"] = "f16_source"
        else:
            t = weights.get(name)
            # Spec convention (cf. LlamaFileType "except 1d tensors"):
            # 1-D tensors (norms) stay F32, 2-D carry-overs go F16.
            tensors[name] = t
            packed_alloc[name] = {"bits": "orig", "gguf_type": "F32" if t.ndim < 2 else "F16"}

    n_quant = len(per_tensor_alloc)
    console.print(
        f"[bold cyan]{n_quant}/{len(model_tensors)} tensors from RCO allocation; "
        f"{len(model_tensors) - n_quant} carried over in F16.[/bold cyan]"
    )

    packer = GGUFPacker(architecture=arch)
    quant_params = sum(int(t.numel()) for n, t in tensors.items() if n in per_tensor_alloc)
    total_params = sum(int(t.numel()) for t in tensors.values())
    metadata = {
        "target_bpw": alloc_data.get("target_bpw", 0.0),
        "achieved_bpw": alloc_data.get("achieved_bpw", 0.0),
        "eval": alloc_data.get("eval", {}),
    }

    # Canonical GGUF names + arch/tokenizer KV from the source model dir.
    from autogsq.packer.hf_gguf_map import load_hf_config, map_tensor_names

    src_path = Path(str(src))
    hf_config = load_hf_config(src_path) if src_path.is_dir() else None
    tokenizer_dir = src_path if (src_path.is_dir() and (src_path / "tokenizer.json").is_file()) else None
    try:
        name_map = map_tensor_names(model_tensors)
    except ValueError as exc:
        console.print(f"[bold red]Tensor name mapping failed: {exc}[/bold red]")
        raise typer.Exit(code=1)
    if hf_config is None:
        console.print("[bold yellow]No config.json at source; writing tensors without arch KV.[/bold yellow]")
    if tokenizer_dir is None:
        console.print("[bold yellow]No tokenizer.json at source; writing tensors without tokenizer KV.[/bold yellow]")

    vocab_size = None
    if tokenizer_dir is not None:
        try:
            from autogsq.packer.hf_gguf_map import _read_tokenizer_bundle

            # Merged length (base vocab + added tokens), matching the written
            # token list — NOT raw model.vocab, which omits specials.
            vocab_size = len(_read_tokenizer_bundle(tokenizer_dir)[0]) or None
        except Exception:
            vocab_size = None

    out_file = packer.stitch_gguf(
        tensors=tensors,
        allocation=packed_alloc,
        out_path=output,
        metadata=metadata,
        model_id=alloc_data.get("model_id", "autogsq_model"),
        tensor_name_map=name_map,
        hf_config=hf_config,
        tokenizer_dir=tokenizer_dir,
        vocab_size=vocab_size,
        jobs=jobs,
    )

    console.print(f"[bold green]Successfully assembled GGUF file: {out_file}[/bold green]")
    console.print(
        f"  tensors: {len(tensors)} | quantized params: {quant_params:,} / {total_params:,} "
        f"({quant_params / max(total_params, 1):.1%})"
    )


@app.command("run-all")
def run_all(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config YAML file"),
    target_bpw: Optional[float] = typer.Option(None, "--target-bpw", help="Target BPW override"),
    output: Path = typer.Option(Path("model-GSQ-RCO.gguf"), "--output", "-o", help="Output .gguf path"),
    max_layers: Optional[int] = typer.Option(None, "--max-layers", help="Limit layers for smoke test"),
) -> None:
    """Run all three stages: generate-db -> allocate -> assemble."""
    cfg = load_config(config)
    db_dir = Path(cfg.output.checkpoint_dir) / "candidate_db"
    alloc_json = Path(cfg.output.checkpoint_dir) / "allocation.json"

    console.print("[bold]=== Step 1: Candidate Database Generation ===[/bold]")
    generate_db(config=config, out_dir=db_dir, resume=True, max_layers=max_layers)

    console.print("[bold]=== Step 2: RCO Budget Allocation ===[/bold]")
    allocate(db_dir=db_dir, config=config, target_bpw=target_bpw, out=alloc_json)

    console.print("[bold]=== Step 3: GGUF Checkpoint Assembly ===[/bold]")
    assemble(db_dir=db_dir, allocation=alloc_json, output=output, config=config)

    console.print(f"[bold green]End-to-end AutoGSQ-RCO pipeline complete! Output: {output}[/bold green]")


@app.command("benchmark")
def benchmark(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML file"),
    model_id: Optional[str] = typer.Option(None, "--model", "-m", help="Model name or path"),
    db_dir: Optional[Path] = typer.Option(None, "--db-dir", "-d", help="Candidate database directory"),
    allocation: Optional[Path] = typer.Option(None, "--allocation", "-a", help="Path to allocation JSON file"),
    dataset: str = typer.Option("wikitext2", "--dataset", help="Validation dataset (wikitext2|c4|fineweb_edu)"),
    num_samples: int = typer.Option(8, "--samples", "-n", help="Number of validation samples"),
    max_length: int = typer.Option(512, "--max-length", "-l", help="Context length"),
    device: str = typer.Option("cpu", "--device", help="Device to evaluate on (cpu|cuda)"),
    eval_mode: str = typer.Option("strided", "--eval-mode", help="Perplexity mode (strided|windowed)"),
    stride: int = typer.Option(512, "--stride", help="Stride for strided eval"),
    split: str = typer.Option("test", "--split", help="Dataset split for strided eval"),
    max_windows: int = typer.Option(0, "--max-windows", help="Cap strided windows (0 = all)"),
    skip_uniform: bool = typer.Option(False, "--skip-uniform", help="Skip the uniform 2-bit baseline"),
) -> None:
    """Benchmark perplexity and compression of base model vs AutoGSQ-RCO allocation."""
    from autogsq.eval.benchmark import run_benchmark
    from autogsq.common.models import load_model_and_tokenizer

    target_model_id = model_id
    if config:
        cfg = load_config(config)
        if not target_model_id:
            target_model_id = cfg.model.name

    if not target_model_id:
        console.print("[bold red]Error: Either --config or --model must be provided.[/bold red]")
        raise typer.Exit(code=1)

    alloc_manifest = None
    if allocation and allocation.is_file():
        with open(allocation, "r", encoding="utf-8") as f:
            alloc_manifest = json.load(f)

    console.print(f"[bold green]Loading model '{target_model_id}' for benchmarking...[/bold green]")
    model, tokenizer = load_model_and_tokenizer(target_model_id, device=device)

    run_benchmark(
        model=model,
        tokenizer=tokenizer,
        candidate_db_dir=db_dir,
        allocation_manifest=alloc_manifest,
        dataset_name=dataset,
        num_samples=num_samples,
        max_length=max_length,
        device=device,
        console=console,
        eval_mode=eval_mode,
        stride=stride,
        split=split,
        max_windows=max_windows,
        skip_uniform=skip_uniform,
    )


if __name__ == "__main__":
    app()

