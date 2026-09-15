"""Regression test: layer-level resume must match tensor-keyed candidates.

Guards the bug where ``is_layer_complete`` checked ``"<layer>::<bits>"``
keys that never exist (candidates are keyed by full tensor name), so every
relaunch re-paid the full Hessian-capture preamble.
"""

import torch

from autogsq.common.store import CandidateStore

OPTS = [2, 3, 4, "ternary"]
TENSORS = ["model.layers.0.mlp.down_proj.weight", "model.layers.0.self_attn.q_proj.weight"]


def _qp():
    return {
        "qweight": torch.zeros(4, dtype=torch.uint8),
        "scales": torch.zeros(1),
        "zeros": torch.zeros(1),
        "bits": 2,
        "group_size": 128,
        "sym": False,
        "shape": (4,),
    }


def test_is_layer_complete_tensor_keyed(tmp_path):
    store = CandidateStore(tmp_path / "db")
    assert store.is_layer_complete("model.layers.0", OPTS, TENSORS) is False
    # Complete only one tensor x all options: still incomplete.
    for b in OPTS:
        store.save_candidate(TENSORS[0], b, torch.zeros(4), _qp())
    assert store.is_layer_complete("model.layers.0", OPTS, TENSORS) is False
    for b in OPTS:
        store.save_candidate(TENSORS[1], b, torch.zeros(4), _qp())
    assert store.is_layer_complete("model.layers.0", OPTS, TENSORS) is True
    # Layer-level key must never count, and other layers stay incomplete.
    assert store.is_layer_complete("model.layers.0", OPTS) is False
    other = [t.replace("layers.0", "layers.1") for t in TENSORS]
    assert store.is_layer_complete("model.layers.1", OPTS, other) is False
