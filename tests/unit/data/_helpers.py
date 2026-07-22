"""Fixture repository builder for composite-corpus tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CompositeFixture:
    """Paths used by a complete two-source composite fixture."""

    root: Path
    config_path: Path
    manifest_path: Path
    output_directory: Path
    source_paths: dict[str, Path]
    adr_paths: dict[str, Path]


def _gutenberg_text(lines: list[str]) -> str:
    return (
        "Project Gutenberg fixture header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
        + "\n".join(lines)
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
        "Project Gutenberg fixture footer\n"
    )


def _source_lines(prefix: str) -> list[str]:
    return [
        f"{prefix} passage {index} contains deliberate unique fixture words number {index}."
        for index in range(20)
    ]


def _write_source(
    root: Path,
    source_id: str,
    lines: list[str],
    status: str,
) -> tuple[Path, Path, Path]:
    raw_path = root / "raw" / source_id / "source.txt"
    manifest_path = root / "manifests" / f"{source_id}.json"
    adr_path = root / "docs" / f"{source_id}.md"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    adr_path.parent.mkdir(parents=True, exist_ok=True)
    raw_content = _gutenberg_text(lines).encode("utf-8")
    raw_path.write_bytes(raw_content)
    manifest = {
        "current_version": f"{source_id}-v1",
        "dataset_id": source_id,
        "raw": {
            "encoding": "utf-8",
            "expected_path": str(raw_path.relative_to(root)),
            "sha256": hashlib.sha256(raw_content).hexdigest(),
            "version": f"{source_id}-v1",
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    adr_path.write_text(f"# Fixture ADR\n\n## Status\n{status}\n", encoding="utf-8")
    return raw_path, manifest_path, adr_path


def build_composite_fixture(
    root: Path,
    *,
    second_status: str = "Accepted",
    threshold: int = 8,
) -> CompositeFixture:
    """Create a complete local fixture without network access."""

    source_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    adr_paths: dict[str, Path] = {}
    for source_id, lines, status in (
        ("fixture-alpha", _source_lines("alpha"), "Accepted"),
        ("fixture-beta", _source_lines("beta"), second_status),
    ):
        raw_path, manifest_path, adr_path = _write_source(root, source_id, lines, status)
        source_paths[source_id] = raw_path
        manifest_paths[source_id] = manifest_path
        adr_paths[source_id] = adr_path

    output_directory = root / "processed" / "fixture-composite"
    manifest_path = root / "composite-manifests" / "fixture-composite.json"
    config_path = root / "config" / "fixture-composite.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "composite_id": "fixture-composite",
        "declared_version": "fixture-v1",
        "heldout_fraction": 0.2,
        "dedup": {"cross_source_min_ngram_tokens": threshold},
        "sources": [
            {
                "source_id": source_id,
                "manifest_path": str(manifest_paths[source_id].relative_to(root)),
                "adr_path": str(adr_paths[source_id].relative_to(root)),
            }
            for source_id in ("fixture-alpha", "fixture-beta")
        ],
        "outputs": {
            "directory": str(output_directory.relative_to(root)),
            "manifest_path": str(manifest_path.relative_to(root)),
            "dedup_log_path": str((output_directory / "dedup-exclusions.jsonl").relative_to(root)),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return CompositeFixture(
        root=root,
        config_path=config_path,
        manifest_path=manifest_path,
        output_directory=output_directory,
        source_paths=source_paths,
        adr_paths=adr_paths,
    )


def set_threshold(fixture: CompositeFixture, threshold: int) -> None:
    """Update only the fixture's configured dedup threshold."""

    config = yaml.safe_load(fixture.config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise AssertionError("Fixture config must be a mapping")
    dedup = config.get("dedup")
    if not isinstance(dedup, dict):
        raise AssertionError("Fixture dedup config must be a mapping")
    dedup["cross_source_min_ngram_tokens"] = threshold
    fixture.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
