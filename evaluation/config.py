"""Load and validate evaluation configuration through the shared Helix loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from helix.common.config import load_mapping, require_positive_int, require_str
from helix.common.exceptions import HelixConfigurationError
from helix.common.paths import resolve_repository_path

DatasetArtifactKind = Literal["raw", "processed"]


@dataclass(frozen=True)
class DatasetTargetConfig:
    """Identify one manifest-backed text artifact used by evaluation."""

    label: str
    manifest_path: Path
    text_path: Path
    artifact: DatasetArtifactKind
    artifact_name: str | None


@dataclass(frozen=True)
class EvaluationConfig:
    """Resolved inputs and metric settings for one evaluation run."""

    in_corpus: DatasetTargetConfig
    out_of_corpus: DatasetTargetConfig
    memorization_corpus: DatasetTargetConfig
    prompt_suite_path: Path
    repetition_n: int
    distinct_n: int
    min_memorization_overlap_tokens: int
    rubric_id: str
    rubric_version: str
    rubric_path: Path
    reports_directory: Path


def _require_mapping(values: dict[str, object], key: str, *, description: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise HelixConfigurationError(f"{description.capitalize()} field '{key}' must be a mapping")
    return cast(dict[str, object], value)


def _repository_path(values: dict[str, object], key: str, *, description: str) -> Path:
    return resolve_repository_path(require_str(values, key, description=description))


def _load_dataset_target(values: dict[str, object], *, label: str) -> DatasetTargetConfig:
    description = f"evaluation dataset '{label}'"
    artifact_value = require_str(values, "artifact", description=description)
    if artifact_value not in {"raw", "processed"}:
        raise HelixConfigurationError(
            f"{description.capitalize()} field 'artifact' must be 'raw' or 'processed'"
        )
    artifact = cast(DatasetArtifactKind, artifact_value)
    artifact_name_value = values.get("artifact_name")
    artifact_name: str | None = None
    if artifact == "processed":
        artifact_name = require_str(values, "artifact_name", description=description)
    elif artifact_name_value is not None:
        raise HelixConfigurationError(
            f"{description.capitalize()} field 'artifact_name' is valid only for processed data"
        )
    return DatasetTargetConfig(
        label=label,
        manifest_path=_repository_path(values, "manifest_path", description=description),
        text_path=_repository_path(values, "text_path", description=description),
        artifact=artifact,
        artifact_name=artifact_name,
    )


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load one evaluation YAML/JSON file through `helix.common.config`."""

    values = load_mapping(path, description="evaluation config")
    datasets = _require_mapping(values, "datasets", description="evaluation config")
    metrics = _require_mapping(values, "metrics", description="evaluation config")
    prompts = _require_mapping(values, "prompts", description="evaluation config")
    rubric = _require_mapping(values, "rubric", description="evaluation config")
    reports = _require_mapping(values, "reports", description="evaluation config")
    return EvaluationConfig(
        in_corpus=_load_dataset_target(
            _require_mapping(datasets, "in_corpus", description="evaluation datasets"),
            label="in_corpus",
        ),
        out_of_corpus=_load_dataset_target(
            _require_mapping(datasets, "out_of_corpus", description="evaluation datasets"),
            label="out_of_corpus",
        ),
        memorization_corpus=_load_dataset_target(
            _require_mapping(
                datasets,
                "memorization_corpus",
                description="evaluation datasets",
            ),
            label="memorization_corpus",
        ),
        prompt_suite_path=_repository_path(
            prompts,
            "path",
            description="evaluation prompts",
        ),
        repetition_n=require_positive_int(
            metrics, "repetition_n", description="evaluation metrics"
        ),
        distinct_n=require_positive_int(metrics, "distinct_n", description="evaluation metrics"),
        min_memorization_overlap_tokens=require_positive_int(
            metrics,
            "min_memorization_overlap_tokens",
            description="evaluation metrics",
        ),
        rubric_id=require_str(rubric, "id", description="evaluation rubric"),
        rubric_version=require_str(rubric, "version", description="evaluation rubric"),
        rubric_path=_repository_path(rubric, "path", description="evaluation rubric"),
        reports_directory=_repository_path(
            reports,
            "directory",
            description="evaluation reports",
        ),
    )
