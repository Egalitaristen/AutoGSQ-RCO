"""SmolLM end-to-end smoke test: allocate -> assemble on the checked-in tiny DB.

Uses ``runtime/smollm_db`` (2 layers x 7 tensors x 4 options) and
``runtime/smollm_model`` so the full RCO allocation + GGUF assembly path is
exercised on CPU in seconds, for any change to the pipeline.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autogsq.cli.main import app
from autogsq.common.store import CandidateStore

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "runtime" / "smollm_db"
MODEL = REPO / "runtime" / "smollm_model"
CONFIG = REPO / "configs" / "examples" / "smollm_135m_2p75bpw.yaml"

runner = CliRunner()

requires_db = pytest.mark.skipif(
    not (DB / "progress.json").is_file() or not MODEL.is_dir(),
    reason="SmolLM smoke fixtures (runtime/smollm_db, runtime/smollm_model) not present",
)


@requires_db
def test_smollm_db_has_fidelity_signal():
    """Every completed candidate must carry reconstruction MSE for allocate."""
    store = CandidateStore(DB)
    completed = store.load_progress().get("completed_candidates", [])
    assert len(completed) == 56  # 14 tensors x 4 options
    for entry in completed:
        name, bits = entry.split("::", 1)
        qp = store.load_qparams(name, bits)
        assert "mse" in qp, f"{entry} missing mse"
        assert qp["mse"] >= 0.0


@requires_db
def test_smollm_allocate_hits_target(tmp_path, monkeypatch):
    """Real CLI allocate must hit 2.75 bpw with options that exist on disk."""
    monkeypatch.chdir(REPO)
    out = tmp_path / "smollm_alloc.json"
    result = runner.invoke(
        app,
        ["allocate", "--db-dir", str(DB), "--config", str(CONFIG), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["target_bpw"] == pytest.approx(2.75)
    assert manifest["achieved_bpw"] == pytest.approx(2.75, abs=0.15)
    assert len(manifest["per_tensor"]) == 14

    store = CandidateStore(DB)
    for name, info in manifest["per_tensor"].items():
        dequant_path, _ = store.get_candidate_paths(name, info["bits"])
        assert dequant_path.is_file(), f"chosen candidate missing: {name}::{info['bits']}"

    # Allocation must be mixed-precision, not degenerate single-bit.
    chosen = {info["bits"] for info in manifest["per_tensor"].values()}
    assert len(chosen) >= 2, f"degenerate allocation: {chosen}"


@requires_db
def test_smollm_assemble_complete_gguf(tmp_path, monkeypatch):
    """Assemble must pack every source tensor; file and sidecar must agree."""
    gguf = pytest.importorskip("gguf")
    monkeypatch.chdir(REPO)

    alloc = tmp_path / "smollm_alloc.json"
    r1 = runner.invoke(
        app,
        ["allocate", "--db-dir", str(DB), "--config", str(CONFIG), "--out", str(alloc)],
    )
    assert r1.exit_code == 0, r1.output

    gguf_path = tmp_path / "model-gsq-rco.gguf"
    r2 = runner.invoke(
        app,
        [
            "assemble",
            "--db-dir", str(DB),
            "--allocation", str(alloc),
            "--output", str(gguf_path),
            "--config", str(CONFIG),
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert gguf_path.is_file()

    from safetensors import safe_open

    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as f:
        src_names = set(f.keys())

    reader = gguf.GGUFReader(str(gguf_path))
    got_names = {t.name for t in reader.tensors}

    # Source HF names map to canonical GGUF names (plus tied output.weight).
    from autogsq.packer.hf_gguf_map import map_tensor_names

    name_map = map_tensor_names(sorted(src_names))
    expected = set(name_map.values())
    assert expected == got_names, (
        f"missing: {sorted(expected - got_names)[:5]}, "
        f"extra: {sorted(got_names - expected)[:5]}"
    )
    assert not [n for n in got_names if n.startswith("model.")], "HF names leaked into GGUF"

    # llama.cpp-loadability KV: arch scalars + tokenizer.
    assert reader.fields["general.architecture"].contents() == "llama"
    assert reader.fields["llama.block_count"].contents() == 30
    assert reader.fields["llama.embedding_length"].contents() == 576
    assert len(reader.fields["tokenizer.ggml.tokens"].contents()) == 49152
    assert reader.fields["tokenizer.ggml.bos_token_id"].contents() == 0
    # BPE vocabs must decode as gpt2 ("llama" breaks tokenization at runtime).
    assert reader.fields["tokenizer.ggml.model"].contents() == "gpt2"
    # Spec convention: 1-D tensors (norms) are F32, not F16.
    from gguf import GGMLQuantizationType as _QT

    for t in reader.tensors:
        if "norm" in t.name:
            assert t.tensor_type == _QT.F32, f"{t.name} is {t.tensor_type}, expected F32"

    # Sidecar manifest must reflect types actually written.
    sidecar = json.loads(
        (gguf_path.with_suffix("")).with_suffix(".rco-allocation.json").read_text(encoding="utf-8")
    )
    assert len(sidecar["per_tensor"]) == len(src_names)
