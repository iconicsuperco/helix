"""Verify tokenizer configuration loading uses shared parsing without behavior changes."""

from __future__ import annotations

from pathlib import Path

import pytest

import research.tokenizer.prepare_corpus as prepare_corpus_module
import research.tokenizer.tokenizer as tokenizer_module
import research.tokenizer.train_tokenizer as train_tokenizer_module


def test_wrapper_config_loader_preserves_invalid_yaml_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("[invalid\n", encoding="utf-8")
    monkeypatch.setenv(tokenizer_module.CONFIG_ENVIRONMENT_VARIABLE, str(config_path))

    with pytest.raises(ValueError, match="Invalid YAML in tokenizer config"):
        tokenizer_module._artifact_path()


def test_corpus_config_loader_preserves_invalid_yaml_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("[invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML in tokenizer config"):
        prepare_corpus_module._load_config(config_path)


def test_training_config_loader_preserves_invalid_yaml_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("[invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid tokenizer config"):
        train_tokenizer_module._load_config(config_path)


def test_training_mapping_loader_preserves_json_for_nonstandard_extension(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tokenizer.config"
    config_path.write_text('{"answer": 42}\n', encoding="utf-8")

    assert train_tokenizer_module._load_mapping(config_path, "tokenizer config") == {"answer": 42}
