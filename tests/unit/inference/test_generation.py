"""Test minimal greedy generation and tokenizer checkpoint identity validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from research.inference.generation import generate_token_ids, validate_tokenizer_identity
from research.model.config import TransformerConfig
from research.model.transformer import HelixTransformer


def test_greedy_generation_selects_argmax_and_crops_context() -> None:
    model = HelixTransformer(
        TransformerConfig(
            n_layer=1,
            n_head=1,
            n_embd=4,
            block_size=2,
            dropout=0.0,
            vocab_size=8,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    generated_ids = generate_token_ids(
        model,
        [1, 2, 3],
        max_new_tokens=2,
        device=torch.device("cpu"),
    )

    assert generated_ids == [1, 2, 3, 0, 0]


def test_tokenizer_identity_accepts_matching_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "tokenizer.json"
    artifact_content = b'{"tokenizer": "fixture"}\n'
    artifact_path.write_bytes(artifact_content)
    checkpoint_config: dict[str, object] = {
        "tokenizer_identity": {
            "artifact_sha256": hashlib.sha256(artifact_content).hexdigest(),
        }
    }

    validate_tokenizer_identity(checkpoint_config, artifact_path)


def test_tokenizer_identity_rejects_mismatched_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "tokenizer.json"
    artifact_path.write_text('{"tokenizer": "current"}\n', encoding="utf-8")
    checkpoint_config: dict[str, object] = {
        "tokenizer_identity": {"artifact_sha256": "checkpoint-sha256"}
    }

    with pytest.raises(ValueError, match="Tokenizer artifact is incompatible"):
        validate_tokenizer_identity(checkpoint_config, artifact_path)
