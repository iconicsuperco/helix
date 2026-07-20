"""Validate manifest-backed evaluation data before model execution."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from evaluation.config import DatasetTargetConfig
from helix.common.config import load_mapping
from helix.common.exceptions import HelixConfigurationError
from helix.common.paths import resolve_repository_path

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidatedDataset:
    """A checksum-verified text artifact and its manifest identity."""

    label: str
    dataset_id: str
    version: str
    sha256: str
    path: Path
    encoding: str


def sha256_file(path: Path) -> str:
    """Compute SHA-256 from the bytes present at `path`."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(values: dict[str, object], key: str, *, description: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise HelixConfigurationError(f"{description.capitalize()} field '{key}' must be a mapping")
    return cast(dict[str, object], value)


def _require_str(values: dict[str, object], key: str, *, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a non-empty string"
        )
    return value


def _require_sha256(values: dict[str, object], key: str, *, description: str) -> str:
    value = _require_str(values, key, description=description).lower()
    if not SHA256_PATTERN.fullmatch(value):
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a lowercase SHA-256 digest"
        )
    return value


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else resolve_repository_path(path)


def _validate_manifest_provenance(
    manifest: dict[str, object], *, description: str
) -> tuple[str, str, dict[str, object]]:
    dataset_id = _require_str(manifest, "dataset_id", description=description)
    current_version = _require_str(manifest, "current_version", description=description)
    _require_str(manifest, "display_name", description=description)
    _require_str(manifest, "license", description=description)
    canonical = _require_mapping(manifest, "canonical_source", description=description)
    for key in ("url", "selection_rationale", "retrieved_at", "retrieval_method"):
        _require_str(canonical, key, description=f"{description} canonical_source")
    fallback_sources = manifest.get("fallback_sources")
    if not isinstance(fallback_sources, list):
        raise HelixConfigurationError(
            f"{description.capitalize()} field 'fallback_sources' must be a list"
        )
    raw = _require_mapping(manifest, "raw", description=description)
    raw_version = _require_str(raw, "version", description=f"{description} raw")
    raw_sha256 = _require_sha256(raw, "sha256", description=f"{description} raw")
    _require_str(raw, "expected_path", description=f"{description} raw")
    _require_str(raw, "encoding", description=f"{description} raw")
    if not isinstance(raw.get("gitignored"), bool):
        raise HelixConfigurationError(
            f"{description.capitalize()} raw field 'gitignored' must be a boolean"
        )
    if dataset_id in {current_version, raw_sha256} or current_version == raw_sha256:
        raise HelixConfigurationError(
            f"{description.capitalize()} must keep identity, version, and checksum distinct"
        )
    if raw_version != current_version:
        raise HelixConfigurationError(
            f"{description.capitalize()} raw version {raw_version!r} does not match "
            f"current_version {current_version!r}"
        )
    return dataset_id, current_version, raw


def _processed_record(
    manifest: dict[str, object], *, name: str, description: str
) -> dict[str, object]:
    outputs = manifest.get("processed_outputs")
    if isinstance(outputs, dict):
        value = outputs.get(name)
        if isinstance(value, dict):
            return cast(dict[str, object], value)
    processed = manifest.get("processed")
    if isinstance(processed, list):
        for value in processed:
            if isinstance(value, dict) and value.get("name") == name:
                return cast(dict[str, object], value)
    raise HelixConfigurationError(
        f"{description.capitalize()} does not define processed artifact {name!r}"
    )


def validate_dataset(target: DatasetTargetConfig) -> ValidatedDataset:
    """Hard-fail unless the configured text matches its complete provenance record."""

    description = f"{target.label} dataset manifest"
    manifest = load_mapping(target.manifest_path, description=description)
    dataset_id, current_version, raw = _validate_manifest_provenance(
        manifest,
        description=description,
    )
    encoding = _require_str(raw, "encoding", description=f"{description} raw").lower()
    if encoding not in {"utf-8", "utf8"}:
        raise HelixConfigurationError(
            f"{description.capitalize()} encoding {encoding!r} is unsupported; expected UTF-8"
        )

    if target.artifact == "raw":
        record = raw
        expected_path_value = _require_str(
            record,
            "expected_path",
            description=f"{description} raw",
        )
        expected_sha256 = _require_sha256(
            record,
            "sha256",
            description=f"{description} raw",
        )
        artifact_version = _require_str(
            record,
            "version",
            description=f"{description} raw",
        )
    else:
        if target.artifact_name is None:
            raise HelixConfigurationError(
                f"{description.capitalize()} requires a processed artifact name"
            )
        record = _processed_record(
            manifest,
            name=target.artifact_name,
            description=description,
        )
        expected_path_value = _require_str(
            record,
            "path",
            description=f"{description} processed artifact",
        )
        expected_sha256 = _require_sha256(
            record,
            "sha256",
            description=f"{description} processed artifact",
        )
        artifact_version = _require_str(
            record,
            "derived_from_raw_version",
            description=f"{description} processed artifact",
        )
        derived_from_raw_sha256 = _require_sha256(
            record,
            "derived_from_raw_sha256",
            description=f"{description} processed artifact",
        )
        raw_sha256 = _require_sha256(raw, "sha256", description=f"{description} raw")
        if derived_from_raw_sha256 != raw_sha256:
            raise HelixConfigurationError(
                f"{description.capitalize()} processed artifact raw checksum does not match "
                "the current raw provenance record"
            )

    expected_path = _resolve_recorded_path(expected_path_value)
    if expected_path != target.text_path.resolve():
        raise HelixConfigurationError(
            f"{description.capitalize()} records {expected_path}, but evaluation configured "
            f"{target.text_path.resolve()}"
        )
    if artifact_version != current_version:
        raise HelixConfigurationError(
            f"{description.capitalize()} artifact version {artifact_version!r} does not match "
            f"current_version {current_version!r}"
        )
    if not target.text_path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset does not exist: {target.text_path} "
            f"(dataset_id={dataset_id}, version={current_version})"
        )
    actual_sha256 = sha256_file(target.text_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Evaluation dataset checksum mismatch for {target.text_path}: "
            f"expected_sha256={expected_sha256}; actual_sha256={actual_sha256}"
        )
    return ValidatedDataset(
        label=target.label,
        dataset_id=dataset_id,
        version=current_version,
        sha256=actual_sha256,
        path=target.text_path,
        encoding="utf-8",
    )
