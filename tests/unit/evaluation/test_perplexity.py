"""Verify perplexity against an analytically predictable model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evaluation.perplexity import compute_perplexity
from research.model.config import TransformerConfig
from research.model.transformer import HelixTransformer


class FixedTokenizer:
    """Return a fixed token sequence for an isolated perplexity fixture."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def encode(self, text: str) -> list[int]:
        assert text == "fixture text"
        return list(self.token_ids)


def _uniform_model() -> HelixTransformer:
    model = HelixTransformer(
        TransformerConfig(
            n_layer=1,
            n_head=1,
            n_embd=4,
            block_size=2,
            dropout=0.0,
            vocab_size=3,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def test_uniform_three_token_model_has_perplexity_three(tmp_path: Path) -> None:
    text_path = tmp_path / "eval.txt"
    text_path.write_text("fixture text", encoding="utf-8")
    model = _uniform_model()
    model.train()

    perplexity = compute_perplexity(
        model,
        FixedTokenizer([0, 1, 2, 1, 0]),
        text_path,
    )

    assert perplexity == pytest.approx(3.0)
    assert model.training


def test_perplexity_rejects_text_with_fewer_than_two_tokens(tmp_path: Path) -> None:
    text_path = tmp_path / "eval.txt"
    text_path.write_text("fixture text", encoding="utf-8")

    with pytest.raises(ValueError, match="at least two tokens"):
        compute_perplexity(_uniform_model(), FixedTokenizer([0]), text_path)
