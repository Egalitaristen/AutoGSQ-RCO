"""Hessian streaming regression: generate-db must never hold >1 layer's Hessians.

Covers the Colab T4 OOM (and the coming local A4000 OOM): Phase 1 used to
accumulate every layer's H/Hinv up front, dying around layer 4-5 on <=16 GB
cards. Runs a tiny 3-block CPU model through the real
``build_candidate_database`` with real Hessian capture; weakrefs prove each
layer's H/Hinv are freed before the next layer is captured.
"""

import gc
import weakref
from types import SimpleNamespace

import torch
import torch.nn as nn

import autogsq.gsq.calibration as calibration_mod
import autogsq.gsq.db_builder as db_mod
from autogsq.gsq.db_builder import build_candidate_database


class _Block(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.k_proj(torch.relu(self.q_proj(x)))


class _Tiny(nn.Module):
    def __init__(self, n_layers=3, dim=16, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.model = SimpleNamespace(
            layers=nn.ModuleList([_Block(dim) for _ in range(n_layers)])
        )

    def forward(self, x):
        h = self.embed(x)
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_hessians_freed_per_layer(tmp_path, monkeypatch):
    tiny = _Tiny()
    monkeypatch.setattr(
        db_mod, "load_model_and_tokenizer", lambda *a, **k: (tiny, None)
    )
    monkeypatch.setattr(
        calibration_mod,
        "prepare_calibration_batches",
        lambda *a, **k: [torch.randint(0, 64, (1, 8)) for _ in range(2)],
    )

    real_collect = calibration_mod.collect_layer_hessians
    live = []  # per-layer lists of weakrefs to H/Hinv tensors

    def tracking_collect(model, layer, layer_name, batches, device, **kw):
        gc.collect()
        for i, refs in enumerate(live):
            still = [r for r in refs if r() is not None]
            assert not still, (
                f"layer {i} Hessians still alive during {layer_name} "
                "capture: streaming violated (OOM on <=16 GB cards)"
            )
        # Every layer must sit on one device: capture forwards run through
        # the whole model, so a layer parked elsewhere breaks the next
        # capture with a cuda:0/cpu mismatch. (Vacuous on CPU; bites on CUDA.)
        devices = {p.device for p in model.parameters()}
        assert len(devices) == 1, f"model split across devices: {devices}"
        out = real_collect(model, layer, layer_name, batches, device, **kw)
        assert len(out) == 2  # q_proj + k_proj
        assert all(set(d) == {"H", "Hinv"} for d in out.values())
        live.append([weakref.ref(v) for d in out.values() for v in d.values()])
        return out

    monkeypatch.setattr(
        calibration_mod, "collect_layer_hessians", tracking_collect
    )

    store = build_candidate_database(
        model_id="tiny",
        calibration_data="synthetic",
        bitwidth_options=[2, 3],
        out_dir=tmp_path / "db",
        num_epochs=1,
        steps_per_epoch=2,
        init_method="rtn",
        device="cpu",
        calib_device="cpu",
        logits_dtype="float32",
    )
    progress = store.load_progress()
    assert len(progress["completed_layers"]) == 3
    assert len(progress["completed_candidates"]) == 3 * 2 * 2

    gc.collect()
    for refs in live:
        assert all(r() is None for r in refs), "Hessians leaked after build"
