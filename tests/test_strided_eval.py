"""Unit tests for strided perplexity evaluation."""

import pytest
import torch

from autogsq.eval.benchmark import evaluate_perplexity_strided


class _TinyLM(torch.nn.Module):
    """Uniform-logit model: PPL must equal vocab size on any text."""

    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab)
        return (logits,)


class _ListTokenizer:
    """Whitespace tokenizer with a fixed small vocab."""

    def __init__(self) -> None:
        self._v = {"<pad>": 0}
        self._n = 1

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = []
        for tok in text.split():
            if tok not in self._v:
                self._v[tok] = self._n
                self._n += 1
            ids.append(self._v[tok])
        import torch as _t

        return type("Enc", (), {"input_ids": _t.tensor([ids])})()


def test_strided_uniform_model_equals_vocab():
    """Uniform logits => NLL = ln(V) per token => PPL == V (model vocab)."""
    model = _TinyLM(vocab=32)
    tok = _ListTokenizer()
    text = "the cat sat on the mat " * 200  # long enough for several windows
    ppl = evaluate_perplexity_strided(
        model, tok, text=text, max_length=64, stride=16, device="cpu"
    )
    assert ppl == pytest.approx(32.0)


def test_strided_scores_each_token_once():
    """Window accounting: first window full, then stride-sized tails."""
    seen = []

    class _SpyLM(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor):
            seen.append(input_ids.size(1))
            b, s = input_ids.shape
            return (torch.zeros(b, s, 32),)

    tok = _ListTokenizer()
    text = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 " * 30  # 300 tokens
    evaluate_perplexity_strided(
        _SpyLM(), tok, text=text, max_length=64, stride=16, device="cpu"
    )
    # begins 0,16,...,240 (end caps at 300 tokens) = 16 windows; after the
    # first, each scores a 16-token tail.
    assert seen[0] == 64
    assert all(s <= 64 for s in seen)
    assert len(seen) == 16
