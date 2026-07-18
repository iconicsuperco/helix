"""Train and document the configured Helix byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import tokenizers  # type: ignore[import-untyped]
import yaml
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "model" / "tokenizer.yaml"
CONFIG_ENVIRONMENT_VARIABLE = "HELIX_TOKENIZER_CONFIG"
LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Format research pipeline logs as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in LOG_RECORD_FIELDS and key not in {"message", "asctime"}
            }
        )
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class TrainingConfig:
    """Validated tokenizer training and artifact configuration."""

    config_path: Path
    vocab_size: int
    special_tokens: list[str]
    unk_token: str
    normalization: str
    lowercase: bool
    pre_tokenizer_type: str
    add_prefix_space: bool
    use_regex: bool
    min_frequency: int
    corpus_name: str
    corpus_path: Path
    training_file: str
    heldout_file: str
    manifest_path: Path
    artifacts_path: Path
    tokenizer_file: str
    tokenizer_card_file: str


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def _load_mapping(path: Path, description: str) -> dict[str, object]:
    try:
        if path.suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Unable to read {description} at {path}") from error
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Invalid {description} at {path}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{description.capitalize()} at {path} must be a mapping")
    return cast(dict[str, object], loaded)


def _required_str(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Tokenizer config field '{key}' must be a non-empty string")
    return value


def _required_bool(values: dict[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Tokenizer config field '{key}' must be a boolean")
    return value


def _required_int(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Tokenizer config field '{key}' must be a positive integer")
    return value


def _resolve_repository_path(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Configured path escapes repository root: {value}")
    return path


def _load_config(config_path: Path) -> TrainingConfig:
    resolved_config_path = config_path.resolve()
    values = _load_mapping(resolved_config_path, "tokenizer config")
    special_tokens_value = values.get("special_tokens")
    if not isinstance(special_tokens_value, list) or not all(
        isinstance(token, str) and token for token in special_tokens_value
    ):
        raise ValueError("Tokenizer config field 'special_tokens' must be a list of strings")
    special_tokens = cast(list[str], special_tokens_value)
    if len(special_tokens) != len(set(special_tokens)):
        raise ValueError("Tokenizer special tokens must be unique")

    pre_tokenizer_value = values.get("pre_tokenizer")
    if not isinstance(pre_tokenizer_value, dict):
        raise ValueError("Tokenizer config field 'pre_tokenizer' must be a mapping")
    pre_tokenizer_config = cast(dict[str, object], pre_tokenizer_value)
    unk_token = _required_str(values, "unk_token")
    if unk_token not in special_tokens:
        raise ValueError("Configured unk_token must appear in special_tokens")

    return TrainingConfig(
        config_path=resolved_config_path,
        vocab_size=_required_int(values, "vocab_size"),
        special_tokens=special_tokens,
        unk_token=unk_token,
        normalization=_required_str(values, "normalization"),
        lowercase=_required_bool(values, "lowercase"),
        pre_tokenizer_type=_required_str(pre_tokenizer_config, "type"),
        add_prefix_space=_required_bool(pre_tokenizer_config, "add_prefix_space"),
        use_regex=_required_bool(pre_tokenizer_config, "use_regex"),
        min_frequency=_required_int(values, "min_frequency"),
        corpus_name=_required_str(values, "corpus_name"),
        corpus_path=_resolve_repository_path(_required_str(values, "corpus_path")),
        training_file=_required_str(values, "training_file"),
        heldout_file=_required_str(values, "heldout_file"),
        manifest_path=_resolve_repository_path(_required_str(values, "manifest_path")),
        artifacts_path=_resolve_repository_path(_required_str(values, "artifacts_path")),
        tokenizer_file=_required_str(values, "tokenizer_file"),
        tokenizer_card_file=_required_str(values, "tokenizer_card_file"),
    )


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise OSError(f"Unable to hash corpus file: {path}") from error


def _compression_helper() -> Callable[[list[str]], float]:
    module_name = f"{__package__}.tokenizer" if __package__ else "tokenizer"
    wrapper_module = importlib.import_module(module_name)
    helper = getattr(wrapper_module, "compute_compression_ratio", None)
    if not callable(helper):
        raise RuntimeError("Tokenizer wrapper does not expose compute_compression_ratio")
    return cast(Callable[[list[str]], float], helper)


def _build_tokenizer(config: TrainingConfig) -> Tokenizer:
    if config.normalization != "NFC" or config.lowercase:
        raise ValueError("Milestone A1 requires NFC normalization without lowercasing")
    if config.pre_tokenizer_type != "ByteLevel":
        raise ValueError("Milestone A1 requires the ByteLevel pre-tokenizer")

    tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=config.add_prefix_space,
        use_regex=config.use_regex,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    training_path = config.corpus_path / config.training_file
    if not training_path.is_file():
        raise FileNotFoundError(f"Prepared training corpus does not exist: {training_path}")
    tokenizer.train([str(training_path)], trainer)
    return tokenizer


def _validate_tokenizer(tokenizer: Tokenizer, config: TrainingConfig) -> None:
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer produced vocabulary size {actual_vocab_size}; expected {config.vocab_size}"
        )
    for expected_id, special_token in enumerate(config.special_tokens):
        actual_id = tokenizer.token_to_id(special_token)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Special token {special_token!r} has ID {actual_id}; expected {expected_id}"
            )


def _write_tokenizer_card(
    config: TrainingConfig,
    manifest: dict[str, object],
    training_hash: str,
    heldout_hash: str,
    compression_ratio: float,
) -> Path:
    source = manifest.get("source")
    license_value = manifest.get("license")
    placeholder = manifest.get("placeholder_corpus")
    if not isinstance(source, str) or not isinstance(license_value, str):
        raise ValueError("Corpus manifest must contain string source and license fields")
    if not isinstance(placeholder, bool):
        raise ValueError("Corpus manifest must contain a boolean placeholder_corpus field")

    training_date = datetime.now(timezone.utc).date().isoformat()
    special_token_rows = "\n".join(
        f"| `{token}` | {token_id} |" for token_id, token in enumerate(config.special_tokens)
    )
    placeholder_note = (
        "This is a placeholder training corpus and must be replaced before the v0.1 model "
        "training run."
        if placeholder
        else "This corpus is approved for the v0.1 model training run."
    )
    card = f"""# Helix Tokenizer Card

## Model

- Algorithm: byte-level BPE
- Vocabulary size: {config.vocab_size}
- Normalization: {config.normalization}, no lowercasing
- Pre-tokenizer: {config.pre_tokenizer_type}
- `tokenizers` version: {tokenizers.__version__}
- Training date (UTC): {training_date}

## Special tokens

| Token | Reserved ID |
|---|---:|
{special_token_rows}

## Corpus

- Name: {config.corpus_name}
- Source: {source}
- License: {license_value}
- Training content SHA-256: `{training_hash}`
- Held-out content SHA-256: `{heldout_hash}`
- Corpus status: {placeholder_note}

The held-out split is assigned deterministically from each unique cleaned line's SHA-256
hash and is excluded from BPE training.

## Evaluation

- Held-out compression ratio: **{compression_ratio:.6f} characters/token**

The ratio is total Unicode characters divided by total emitted tokens over the held-out
sample. Empty samples would report `0.0`; this training run used a non-empty sample.

## Reproducibility

Corpus cleaning, exact-line deduplication, and held-out assignment are deterministic for
the same raw files and config. Hugging Face `tokenizers` does not expose a random seed for
`BpeTrainer`; this run uses its deterministic frequency-based training path with one
ordered training file.
"""
    card_path = config.artifacts_path / config.tokenizer_card_file
    card_path.write_text(card, encoding="utf-8")
    return card_path


def train_tokenizer(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Train, validate, serialize, and document the configured tokenizer."""

    config = _load_config(config_path)
    manifest = _load_mapping(config.manifest_path, "corpus manifest")
    training_path = config.corpus_path / config.training_file
    heldout_path = config.corpus_path / config.heldout_file
    if not heldout_path.is_file():
        raise FileNotFoundError(f"Prepared held-out corpus does not exist: {heldout_path}")

    training_hash = _sha256(training_path)
    heldout_hash = _sha256(heldout_path)
    manifest_hash = manifest.get("content_hash")
    if manifest_hash != training_hash:
        raise ValueError(
            "Prepared training corpus hash does not match the manifest; rerun prepare_corpus.py"
        )

    LOGGER.info("Training byte-level BPE tokenizer", extra={"corpus_name": config.corpus_name})
    tokenizer = _build_tokenizer(config)
    _validate_tokenizer(tokenizer, config)
    config.artifacts_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = config.artifacts_path / config.tokenizer_file
    tokenizer.save(str(tokenizer_path), pretty=True)

    try:
        heldout_texts = heldout_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise ValueError(f"Held-out corpus is not valid UTF-8: {heldout_path}") from error
    except OSError as error:
        raise OSError(f"Unable to read held-out corpus: {heldout_path}") from error
    if not heldout_texts:
        raise ValueError("Held-out corpus is empty")
    os.environ[CONFIG_ENVIRONMENT_VARIABLE] = str(config.config_path)
    compression_ratio = _compression_helper()(heldout_texts)
    card_path = _write_tokenizer_card(
        config,
        manifest,
        training_hash,
        heldout_hash,
        compression_ratio,
    )
    LOGGER.info(
        "Tokenizer training complete",
        extra={
            "tokenizer_path": str(tokenizer_path),
            "card_path": str(card_path),
            "compression_ratio": compression_ratio,
        },
    )
    return tokenizer_path


def main() -> None:
    """Run tokenizer training from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    _configure_logging()
    train_tokenizer(arguments.config)


if __name__ == "__main__":
    main()
