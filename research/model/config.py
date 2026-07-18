"""Load and validate configuration for the Helix transformer architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "model" / "transformer.yaml"


@dataclass(frozen=True)
class TransformerConfig:
    """Validated dimensions and dropout settings for a Helix transformer."""

    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    vocab_size: int


def _load_mapping(path: Path, description: str) -> dict[str, object]:
    """Read one YAML mapping and add context to parsing and I/O failures."""

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Unable to read {description} at {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {description} at {path}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{description.capitalize()} at {path} must be a mapping")
    return cast(dict[str, object], loaded)


def _required_positive_int(values: dict[str, object], key: str) -> int:
    """Return a required positive integer configuration field."""

    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Transformer config field '{key}' must be a positive integer")
    return value


def _required_dropout(values: dict[str, object]) -> float:
    """Return the required dropout probability after validating its range."""

    value = values.get("dropout")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Transformer config field 'dropout' must be numeric")
    dropout = float(value)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Transformer config field 'dropout' must be in [0.0, 1.0)")
    return dropout


def _required_path(values: dict[str, object], key: str) -> Path:
    """Resolve a required repository-relative path without allowing escapes."""

    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Transformer config field '{key}' must be a non-empty string")
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Configured path escapes repository root: {value}")
    return path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> TransformerConfig:
    """Load transformer dimensions and the tokenizer-owned vocabulary size."""

    values = _load_mapping(config_path.resolve(), "transformer config")
    if "vocab_size" in values:
        raise ValueError(
            "Transformer config must reference tokenizer_config instead of duplicating vocab_size"
        )

    n_layer = _required_positive_int(values, "n_layer")
    n_head = _required_positive_int(values, "n_head")
    n_embd = _required_positive_int(values, "n_embd")
    block_size = _required_positive_int(values, "block_size")
    dropout = _required_dropout(values)
    tokenizer_config_path = _required_path(values, "tokenizer_config")
    tokenizer_values = _load_mapping(tokenizer_config_path, "tokenizer config")
    vocab_size = _required_positive_int(tokenizer_values, "vocab_size")

    if n_embd % n_head != 0:
        raise ValueError("Transformer config field 'n_embd' must be divisible by 'n_head'")

    return TransformerConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        vocab_size=vocab_size,
    )
