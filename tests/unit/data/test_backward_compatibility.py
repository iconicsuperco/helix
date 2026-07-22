"""Regression evidence for the unchanged Moby-Dick and tokenizer paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from helix.common.config import load_mapping
from research.tokenizer import tokenizer as tokenizer_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_prepare_corpus_source_and_outputs_are_byte_identical() -> None:
    assert _sha256(REPOSITORY_ROOT / "research/tokenizer/prepare_corpus.py") == (
        "0ddfe9914393ce79c2ae9bd002effb1aa12bd4ef27aaa23ff817ae1ce50e2383"
    )
    assert _sha256(REPOSITORY_ROOT / "datasets/processed/tokenizer-training/train.txt") == (
        "bc1a181e166a877841874cb8edf81b8b3af6e858b4c49ed276d1ef0d44c2772b"
    )
    assert _sha256(REPOSITORY_ROOT / "datasets/processed/tokenizer-training/heldout.txt") == (
        "6e5a61a28f1cf81606f0499bdd069a9bf5f1d59a8e5fa2ac3b89280a9a1ae4d9"
    )


def test_existing_training_config_and_tokenizer_artifact_still_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_mapping(
        REPOSITORY_ROOT / "config/model/transformer-smoke.yaml",
        description="model config",
        file_format="yaml",
    )
    assert config["tokenizer_config"] == "config/model/tokenizer.yaml"
    assert _sha256(REPOSITORY_ROOT / "research/tokenizer/artifacts/tokenizer.json") == (
        "3ededb0491b18d351cbbd9c03f4e9f9add452bb8b5c318b05c08bbadb0d996a4"
    )
    monkeypatch.delenv("HELIX_TOKENIZER_CONFIG", raising=False)
    monkeypatch.setattr(tokenizer_module, "_tokenizer", None)
    monkeypatch.setattr(tokenizer_module, "_loaded_artifact_path", None)
    text = "The old tokenizer remains the default."
    assert tokenizer_module.decode(tokenizer_module.encode(text)) == text


def test_composite_tokenizer_is_a_distinct_usable_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (
        REPOSITORY_ROOT / "research/data/tokenizers/helix-gutenberg-prose-bpe-v1/tokenizer.yaml"
    )
    monkeypatch.setenv("HELIX_TOKENIZER_CONFIG", str(config_path))
    monkeypatch.setattr(tokenizer_module, "_tokenizer", None)
    monkeypatch.setattr(tokenizer_module, "_loaded_artifact_path", None)

    artifact_path = tokenizer_module.tokenizer_artifact_path()
    assert artifact_path != REPOSITORY_ROOT / "research/tokenizer/artifacts/tokenizer.json"
    assert artifact_path.is_file()
    text = "Time and tide belong to the composite corpus."
    assert tokenizer_module.decode(tokenizer_module.encode(text)) == text
