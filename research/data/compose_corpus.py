"""Build a versioned composite corpus from approved, verified sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from helix.common.config import load_mapping
from research.data.cleaning import CLEANING_VERSION, clean_source_text
from research.data.dedup import (
    DedupExclusion,
    SourceLine,
    deduplicate_across_sources,
    deduplicate_within_source,
    find_partition_leakage,
)
from research.tokenizer.prepare_corpus import _is_heldout, _serialize_lines

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "research/data/config/composite-gutenberg-prose-v1.yaml"


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    manifest_path: Path
    adr_path: Path


@dataclass(frozen=True)
class CompositionConfig:
    config_path: Path
    composite_id: str
    declared_version: str
    heldout_fraction: float
    min_ngram_tokens: int
    sources: list[SourceConfig]
    output_directory: Path
    manifest_path: Path
    dedup_log_path: Path


@dataclass(frozen=True)
class VerifiedSource:
    source_id: str
    source_version: str
    raw_sha256: str
    manifest_path: Path
    records: list[SourceLine]
    exclusions: list[DedupExclusion]


def _required_str(values: dict[str, object], key: str, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} field '{key}' must be a non-empty string")
    return value


def _repository_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Configured path escapes repository root: {value}")
    return path


def _load_config(config_path: Path, repository_root: Path) -> CompositionConfig:
    values = load_mapping(config_path, description="composite corpus config", file_format="yaml")
    sources_value = values.get("sources")
    dedup_value = values.get("dedup")
    outputs_value = values.get("outputs")
    heldout_value = values.get("heldout_fraction")
    if not isinstance(sources_value, list) or not sources_value:
        raise ValueError("Composite corpus config field 'sources' must be a non-empty list")
    if not isinstance(dedup_value, dict) or not isinstance(outputs_value, dict):
        raise ValueError("Composite corpus config requires dedup and outputs mappings")
    if not isinstance(heldout_value, (int, float)) or isinstance(heldout_value, bool):
        raise ValueError("Composite corpus heldout_fraction must be numeric")
    heldout_fraction = float(heldout_value)
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("Composite corpus heldout_fraction must be between 0 and 1")
    threshold = dedup_value.get("cross_source_min_ngram_tokens")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError("Cross-source deduplication threshold must be a positive integer")

    sources: list[SourceConfig] = []
    for source_value in sources_value:
        if not isinstance(source_value, dict):
            raise ValueError("Each composite corpus source must be a mapping")
        source_mapping = cast(dict[str, object], source_value)
        sources.append(
            SourceConfig(
                source_id=_required_str(source_mapping, "source_id", "Source"),
                manifest_path=_repository_path(
                    repository_root,
                    _required_str(source_mapping, "manifest_path", "Source"),
                ),
                adr_path=_repository_path(
                    repository_root,
                    _required_str(source_mapping, "adr_path", "Source"),
                ),
            )
        )
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Composite corpus source IDs must be unique")

    return CompositionConfig(
        config_path=config_path.resolve(),
        composite_id=_required_str(values, "composite_id", "Composite corpus config"),
        declared_version=_required_str(values, "declared_version", "Composite corpus config"),
        heldout_fraction=heldout_fraction,
        min_ngram_tokens=threshold,
        sources=sources,
        output_directory=_repository_path(
            repository_root,
            _required_str(outputs_value, "directory", "Outputs"),
        ),
        manifest_path=_repository_path(
            repository_root,
            _required_str(outputs_value, "manifest_path", "Outputs"),
        ),
        dedup_log_path=_repository_path(
            repository_root,
            _required_str(outputs_value, "dedup_log_path", "Outputs"),
        ),
    )


def _adr_status(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError(f"Unable to read source ADR: {path}") from error
    for index, line in enumerate(lines):
        if line.strip() == "## Status":
            status = next((item.strip() for item in lines[index + 1 :] if item.strip()), "")
            if status:
                return status
    for line in lines:
        if line.startswith("- Status:"):
            return line.partition(":")[2].strip()
    raise ValueError(f"Source ADR has no Status field: {path}")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify_source(source: SourceConfig, repository_root: Path) -> VerifiedSource:
    status = _adr_status(source.adr_path)
    if status != "Accepted":
        raise ValueError(
            f"Source '{source.source_id}' is governed by ADR status {status!r}; expected 'Accepted'"
        )
    manifest = load_mapping(source.manifest_path, description="source manifest", file_format="json")
    dataset_id = _required_str(manifest, "dataset_id", "Source manifest")
    current_version = _required_str(manifest, "current_version", "Source manifest")
    raw_value = manifest.get("raw")
    if dataset_id != source.source_id:
        raise ValueError(
            f"Source ID mismatch: config has {source.source_id!r}, manifest has {dataset_id!r}"
        )
    if not isinstance(raw_value, dict):
        raise ValueError(f"Source manifest '{dataset_id}' must contain a raw mapping")
    raw = cast(dict[str, object], raw_value)
    raw_version = _required_str(raw, "version", "Source raw manifest")
    raw_sha256 = _required_str(raw, "sha256", "Source raw manifest")
    encoding = _required_str(raw, "encoding", "Source raw manifest")
    raw_path = _repository_path(
        repository_root,
        _required_str(raw, "expected_path", "Source raw manifest"),
    )
    if current_version != raw_version:
        raise ValueError(
            f"Source '{dataset_id}' version mismatch: "
            f"current={current_version!r}, raw={raw_version!r}"
        )
    if encoding.lower().replace("_", "-") != "utf-8":
        raise ValueError(f"Source '{dataset_id}' encoding must be verified UTF-8, got {encoding!r}")
    try:
        raw_content = raw_path.read_bytes()
    except OSError as error:
        raise OSError(f"Unable to read raw source '{dataset_id}' at {raw_path}") from error
    actual_sha256 = _sha256(raw_content)
    if actual_sha256 != raw_sha256:
        raise ValueError(
            f"Source '{dataset_id}' checksum mismatch: expected {raw_sha256}, got {actual_sha256}"
        )
    try:
        text = raw_content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"Source '{dataset_id}' is not valid UTF-8") from error

    cleaned_lines = clean_source_text(text)
    records, exclusions = deduplicate_within_source(source.source_id, cleaned_lines)
    return VerifiedSource(
        source_id=source.source_id,
        source_version=current_version,
        raw_sha256=raw_sha256,
        manifest_path=source.manifest_path,
        records=records,
        exclusions=exclusions,
    )


def _composition_fingerprint(config: CompositionConfig, sources: list[VerifiedSource]) -> str:
    version_input = {
        "cleaning_version": CLEANING_VERSION,
        "declared_version": config.declared_version,
        "heldout_fraction": config.heldout_fraction,
        "min_ngram_tokens": config.min_ngram_tokens,
        "sources": [
            {
                "raw_sha256": source.raw_sha256,
                "source_id": source.source_id,
                "source_version": source.source_version,
            }
            for source in sources
        ],
    }
    content = json.dumps(version_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(content)


def _split_records(
    records: list[SourceLine], heldout_fraction: float
) -> tuple[list[SourceLine], list[SourceLine]]:
    training = [record for record in records if not _is_heldout(record.text, heldout_fraction)]
    heldout = [record for record in records if _is_heldout(record.text, heldout_fraction)]
    if not heldout and len(training) > 1:
        heldout.append(training.pop())
    if not training and len(heldout) > 1:
        training.append(heldout.pop(0))
    if not training or not heldout:
        raise ValueError("Composite corpus must contain at least two unique lines")
    return training, heldout


def _validate_no_leakage(
    training: list[SourceLine], heldout: list[SourceLine], min_ngram_tokens: int
) -> None:
    findings = find_partition_leakage(training, heldout, min_ngram_tokens)
    if findings:
        detail = ", ".join(
            f"{training_id}->{heldout_id}:{reason}" for training_id, heldout_id, reason in findings
        )
        raise ValueError(f"Composite train/heldout leakage detected across sources: {detail}")


def _serialize_exclusions(exclusions: list[DedupExclusion]) -> bytes:
    return b"".join(
        (json.dumps(exclusion.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for exclusion in exclusions
    )


def _archive_previous_version(
    config: CompositionConfig,
    new_version: str,
) -> None:
    if not config.manifest_path.is_file():
        return
    prior = load_mapping(
        config.manifest_path,
        description="existing composite corpus manifest",
        file_format="json",
    )
    prior_version = _required_str(prior, "current_version", "Existing composite manifest")
    if prior_version == new_version:
        return
    manifest_archive = (
        config.manifest_path.parent / "history" / config.composite_id / f"{prior_version}.json"
    )
    output_archive = config.output_directory / "versions" / prior_version
    manifest_archive.parent.mkdir(parents=True, exist_ok=True)
    output_archive.mkdir(parents=True, exist_ok=True)
    if manifest_archive.exists():
        raise FileExistsError(f"Archived composite manifest already exists: {manifest_archive}")
    shutil.copy2(config.manifest_path, manifest_archive)
    for path in (
        config.output_directory / "train.txt",
        config.output_directory / "heldout.txt",
        config.dedup_log_path,
    ):
        if path.is_file():
            shutil.copy2(path, output_archive / path.name)


def compose_corpus(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Validate sources and write a deterministic composite corpus and manifest."""

    root = repository_root.resolve()
    config = _load_config(config_path.resolve(), root)
    verified_sources = [_verify_source(source, root) for source in config.sources]
    if config.composite_id in {source.source_id for source in verified_sources}:
        raise ValueError("Composite identity must differ from every source identity")

    all_exclusions: list[DedupExclusion] = []
    source_records: list[tuple[str, list[SourceLine]]] = []
    for verified_source in verified_sources:
        source_records.append((verified_source.source_id, verified_source.records))
        all_exclusions.extend(verified_source.exclusions)

    retained_by_source, cross_source_exclusions = deduplicate_across_sources(
        source_records,
        config.min_ngram_tokens,
    )
    all_exclusions.extend(cross_source_exclusions)
    combined = [
        record for source in config.sources for record in retained_by_source[source.source_id]
    ]
    training, heldout = _split_records(combined, config.heldout_fraction)
    _validate_no_leakage(training, heldout, config.min_ngram_tokens)

    training_content = _serialize_lines([record.text for record in training])
    heldout_content = _serialize_lines([record.text for record in heldout])
    dedup_content = _serialize_exclusions(all_exclusions)
    fingerprint = _composition_fingerprint(config, verified_sources)
    current_version = f"{config.declared_version}-{fingerprint[:12]}"
    _archive_previous_version(config, current_version)

    config.output_directory.mkdir(parents=True, exist_ok=True)
    config.dedup_log_path.parent.mkdir(parents=True, exist_ok=True)
    training_path = config.output_directory / "train.txt"
    heldout_path = config.output_directory / "heldout.txt"
    training_path.write_bytes(training_content)
    heldout_path.write_bytes(heldout_content)
    config.dedup_log_path.write_bytes(dedup_content)

    def relative(path: Path) -> str:
        return str(path.relative_to(root))

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": config.composite_id,
        "display_name": "Helix Gutenberg prose composite corpus",
        "current_version": current_version,
        "content_hash": _sha256(training_content),
        "source": "Composite of approved Project Gutenberg source datasets",
        "license": (
            "Public domain in the USA; distributed under the Project Gutenberg License. "
            "Verify status in the deployment jurisdiction."
        ),
        "placeholder_corpus": False,
        "sources": [
            {
                "manifest_path": relative(source.manifest_path),
                "raw_sha256": source.raw_sha256,
                "source_id": source.source_id,
                "source_version": source.source_version,
            }
            for source in verified_sources
        ],
        "composition": {
            "config_path": relative(config.config_path),
            "fingerprint_sha256": fingerprint,
            "cleaning_version": CLEANING_VERSION,
            "heldout_assignment": "sha256-line-hash-over-full-composite",
            "heldout_fraction": config.heldout_fraction,
            "within_source_exact_line_deduplication": True,
            "cross_source_min_ngram_tokens": config.min_ngram_tokens,
            "prior_versions_archived": True,
        },
        "processed_outputs": {
            "training": {
                "path": relative(training_path),
                "sha256": _sha256(training_content),
                "unique_line_count": len(training),
            },
            "heldout": {
                "path": relative(heldout_path),
                "sha256": _sha256(heldout_content),
                "unique_line_count": len(heldout),
            },
            "dedup_exclusions": {
                "path": relative(config.dedup_log_path),
                "sha256": _sha256(dedup_content),
                "exclusion_count": len(all_exclusions),
            },
        },
    }
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config.manifest_path


def main() -> None:
    """Compose the configured corpus from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    compose_corpus(arguments.config)


if __name__ == "__main__":
    main()
