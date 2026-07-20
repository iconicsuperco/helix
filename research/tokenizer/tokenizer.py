"""Expose the trained Helix tokenizer through a small, typed Python API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from tokenizers import Tokenizer

from helix.common.config import load_mapping
from helix.common.exceptions import HelixConfigurationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ENVIRONMENT_VARIABLE = "HELIX_TOKENIZER_CONFIG"
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "model" / "tokenizer.yaml"
_tokenizer: Tokenizer | None = None
_loaded_artifact_path: Path | None = None


def _config_path() -> Path:
    configured_path = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    return Path(configured_path).resolve() if configured_path else DEFAULT_CONFIG_PATH


def _artifact_path() -> Path:
    config_path = _config_path()
    try:
        values = load_mapping(
            config_path,
            description="tokenizer config",
            file_format="yaml",
        )
    except HelixConfigurationError as error:
        cause = error.__cause__
        if isinstance(cause, OSError):
            raise OSError(str(error)) from cause
        if cause is not None:
            raise ValueError(f"Invalid YAML in tokenizer config at {config_path}") from cause
        raise ValueError(str(error)) from error
    artifacts_path = values.get("artifacts_path")
    tokenizer_file = values.get("tokenizer_file")
    if not isinstance(artifacts_path, str) or not artifacts_path:
        raise ValueError("Tokenizer config field 'artifacts_path' must be a non-empty string")
    if not isinstance(tokenizer_file, str) or not tokenizer_file:
        raise ValueError("Tokenizer config field 'tokenizer_file' must be a non-empty string")
    path = (REPOSITORY_ROOT / artifacts_path / tokenizer_file).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Configured tokenizer artifact escapes repository root: {path}")
    return path


def _get_tokenizer() -> Tokenizer:
    global _tokenizer, _loaded_artifact_path

    artifact_path = _artifact_path()
    if _tokenizer is None or _loaded_artifact_path != artifact_path:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Tokenizer artifact does not exist: {artifact_path}")
        _tokenizer = Tokenizer.from_file(str(artifact_path))
        _loaded_artifact_path = artifact_path
    return _tokenizer


def encode(text: str) -> list[int]:
    """Encode one string without automatically adding special tokens."""

    return cast(list[int], _get_tokenizer().encode(text, add_special_tokens=False).ids)


def decode(ids: list[int]) -> str:
    """Decode token IDs to clean text, excluding reserved special tokens."""

    return cast(str, _get_tokenizer().decode(ids, skip_special_tokens=True))


def encode_batch(texts: list[str]) -> list[list[int]]:
    """Encode strings in one tokenizer batch without adding special tokens."""

    encodings = _get_tokenizer().encode_batch(texts, add_special_tokens=False)
    return [encoding.ids for encoding in encodings]


def compute_compression_ratio(sample_texts: list[str]) -> float:
    """Return corpus-level average Unicode characters per emitted token."""

    character_count = sum(len(text) for text in sample_texts)
    token_count = sum(len(ids) for ids in encode_batch(sample_texts))
    if token_count == 0:
        return 0.0
    return character_count / token_count
