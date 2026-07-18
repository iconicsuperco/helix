"""Clean a raw text corpus and create reproducible tokenizer data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

from helix.common.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "model" / "tokenizer.yaml"


@dataclass(frozen=True)
class CorpusConfig:
    """Configuration required by the corpus preparation pipeline."""

    corpus_name: str
    source: str
    license: str
    placeholder: bool
    raw_corpus_path: Path
    raw_file_glob: str
    corpus_path: Path
    training_file: str
    heldout_file: str
    heldout_fraction: float
    manifest_path: Path


def _configure_logging() -> None:
    configure_logging()


def _load_mapping(config_path: Path) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Unable to read tokenizer config at {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in tokenizer config at {config_path}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"Tokenizer config at {config_path} must be a mapping")
    return cast(dict[str, object], loaded)


def _required_str(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Tokenizer config field '{key}' must be a non-empty string")
    return value


def _resolve_repository_path(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"Configured path escapes repository root: {value}")
    return path


def _load_config(config_path: Path) -> CorpusConfig:
    values = _load_mapping(config_path)
    placeholder = values.get("placeholder_corpus")
    heldout_fraction = values.get("heldout_fraction")
    if not isinstance(placeholder, bool):
        raise ValueError("Tokenizer config field 'placeholder_corpus' must be a boolean")
    if not isinstance(heldout_fraction, (int, float)):
        raise ValueError("Tokenizer config field 'heldout_fraction' must be numeric")
    fraction = float(heldout_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("Tokenizer config field 'heldout_fraction' must be between 0 and 1")

    return CorpusConfig(
        corpus_name=_required_str(values, "corpus_name"),
        source=_required_str(values, "corpus_source"),
        license=_required_str(values, "corpus_license"),
        placeholder=placeholder,
        raw_corpus_path=_resolve_repository_path(_required_str(values, "raw_corpus_path")),
        raw_file_glob=_required_str(values, "raw_file_glob"),
        corpus_path=_resolve_repository_path(_required_str(values, "corpus_path")),
        training_file=_required_str(values, "training_file"),
        heldout_file=_required_str(values, "heldout_file"),
        heldout_fraction=fraction,
        manifest_path=_resolve_repository_path(_required_str(values, "manifest_path")),
    )


def _strip_control_characters(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )


def _read_unique_lines(raw_files: list[Path]) -> list[str]:
    unique_lines: list[str] = []
    seen: set[str] = set()
    for raw_file in raw_files:
        try:
            text = raw_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Raw corpus file is not valid UTF-8: {raw_file}") from error
        except OSError as error:
            raise OSError(f"Unable to read raw corpus file: {raw_file}") from error
        for line in _strip_control_characters(text).split("\n"):
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
    return unique_lines


def _is_heldout(line: str, fraction: float) -> bool:
    digest = hashlib.sha256(line.encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    return score < fraction


def _split_lines(lines: list[str], heldout_fraction: float) -> tuple[list[str], list[str]]:
    training = [line for line in lines if not _is_heldout(line, heldout_fraction)]
    heldout = [line for line in lines if _is_heldout(line, heldout_fraction)]
    if not heldout and len(training) > 1:
        heldout.append(training.pop())
    if not training and len(heldout) > 1:
        training.append(heldout.pop(0))
    if not training or not heldout:
        raise ValueError("Corpus must contain at least two unique lines for a held-out split")
    return training, heldout


def _serialize_lines(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def prepare_corpus(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Clean configured raw inputs, write train/held-out files, and return the manifest path."""

    config = _load_config(config_path.resolve())
    raw_files = sorted(
        path for path in config.raw_corpus_path.glob(config.raw_file_glob) if path.is_file()
    )
    if not raw_files:
        raise FileNotFoundError(
            f"No raw corpus files matching '{config.raw_file_glob}' in {config.raw_corpus_path}"
        )

    LOGGER.info(
        "Preparing tokenizer corpus",
        extra={"corpus_name": config.corpus_name, "raw_file_count": len(raw_files)},
    )
    unique_lines = _read_unique_lines(raw_files)
    training_lines, heldout_lines = _split_lines(unique_lines, config.heldout_fraction)
    training_content = _serialize_lines(training_lines)
    heldout_content = _serialize_lines(heldout_lines)

    config.corpus_path.mkdir(parents=True, exist_ok=True)
    training_path = config.corpus_path / config.training_file
    heldout_path = config.corpus_path / config.heldout_file
    training_path.write_bytes(training_content)
    heldout_path.write_bytes(heldout_content)

    processed_date = datetime.now(UTC).date().isoformat()
    training_hash = _sha256(training_content)
    heldout_hash = _sha256(heldout_content)
    manifest: dict[str, object] = {
        "corpus_name": config.corpus_name,
        "source": config.source,
        "license": config.license,
        "placeholder_corpus": config.placeholder,
        "processed_date": processed_date,
        "raw_files": [str(path.relative_to(REPOSITORY_ROOT)) for path in raw_files],
        "content_hash": training_hash,
        "processed_outputs": {
            "training": {
                "path": str(training_path.relative_to(REPOSITORY_ROOT)),
                "sha256": training_hash,
                "unique_line_count": len(training_lines),
            },
            "heldout": {
                "path": str(heldout_path.relative_to(REPOSITORY_ROOT)),
                "sha256": heldout_hash,
                "unique_line_count": len(heldout_lines),
            },
        },
        "processing": {
            "control_characters_removed_except": ["newline", "tab"],
            "exact_duplicate_lines_removed": True,
            "heldout_fraction": config.heldout_fraction,
            "heldout_assignment": "sha256-line-hash",
        },
    }
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Tokenizer corpus prepared",
        extra={
            "training_hash": training_hash,
            "training_lines": len(training_lines),
            "heldout_lines": len(heldout_lines),
        },
    )
    return config.manifest_path


def main() -> None:
    """Run corpus preparation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    _configure_logging()
    prepare_corpus(arguments.config)


if __name__ == "__main__":
    main()
