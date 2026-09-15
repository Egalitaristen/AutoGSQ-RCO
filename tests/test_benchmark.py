"""Unit tests for perplexity evaluation and benchmark harness."""

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from autogsq.eval.benchmark import evaluate_perplexity, run_benchmark


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.q_proj(x)
        return self.down_proj(h) + x


class MockCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([MockDecoderLayer(hidden_dim) for _ in range(2)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits


def test_evaluate_perplexity_mock_lm():
    """Verify perplexity computation returns a finite positive value."""
    torch.manual_seed(42)
    model = MockCausalLM()

    # Synthetic batches
    batches = [torch.randint(0, 1000, (1, 32)) for _ in range(3)]
    ppl = evaluate_perplexity(model, batches=batches)

    assert not math.isnan(ppl)
    assert not math.isinf(ppl)
    assert ppl > 1.0


def test_run_benchmark_mock_pipeline():
    """Verify run_benchmark runs cleanly with mock model and produces valid metrics."""
    torch.manual_seed(42)
    model = MockCausalLM()

    results = run_benchmark(
        model=model,
        tokenizer=None,
        dataset_name="wikitext2",
        num_samples=2,
        max_length=32,
        device="cpu",
    )

    assert "baseline" in results
    assert results["baseline"]["bpw"] == 16.0
    assert results["baseline"]["ppl"] > 1.0
    assert results["baseline"]["compression"] == 1.0
