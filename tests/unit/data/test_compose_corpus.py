"""Tests for source validation, manifest generation, and corpus versioning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.data.compose_corpus import _validate_no_leakage, compose_corpus
from research.data.dedup import SourceLine
from tests.unit.data._helpers import build_composite_fixture, set_threshold


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("Expected JSON mapping")
    return value


def test_compose_corpus_generates_manifest_with_distinct_source_lineage(tmp_path: Path) -> None:
    fixture = build_composite_fixture(tmp_path)

    result = compose_corpus(fixture.config_path, repository_root=fixture.root)

    manifest = _load_json(result)
    assert manifest["dataset_id"] == "fixture-composite"
    assert str(manifest["current_version"]).startswith("fixture-v1-")
    sources = manifest["sources"]
    assert isinstance(sources, list)
    assert [(source["source_id"], source["source_version"]) for source in sources] == [
        ("fixture-alpha", "fixture-alpha-v1"),
        ("fixture-beta", "fixture-beta-v1"),
    ]
    outputs = manifest["processed_outputs"]
    assert isinstance(outputs, dict)
    for output_name in ("training", "heldout", "dedup_exclusions"):
        output = outputs[output_name]
        assert isinstance(output, dict)
        output_path = fixture.root / str(output["path"])
        assert hashlib.sha256(output_path.read_bytes()).hexdigest() == output["sha256"]


def test_compose_corpus_refuses_non_accepted_source_adr(tmp_path: Path) -> None:
    fixture = build_composite_fixture(tmp_path, second_status="Proposed")

    with pytest.raises(ValueError, match="status 'Proposed'; expected 'Accepted'"):
        compose_corpus(fixture.config_path, repository_root=fixture.root)


def test_compose_corpus_hard_fails_source_checksum_mismatch(tmp_path: Path) -> None:
    fixture = build_composite_fixture(tmp_path)
    fixture.source_paths["fixture-beta"].write_text("changed bytes", encoding="utf-8")

    with pytest.raises(ValueError, match="fixture-beta.*checksum mismatch"):
        compose_corpus(fixture.config_path, repository_root=fixture.root)


def test_dedup_policy_change_produces_new_version_and_archives_prior(tmp_path: Path) -> None:
    fixture = build_composite_fixture(tmp_path, threshold=8)
    compose_corpus(fixture.config_path, repository_root=fixture.root)
    first_manifest = _load_json(fixture.manifest_path)
    first_version = str(first_manifest["current_version"])

    set_threshold(fixture, 9)
    compose_corpus(fixture.config_path, repository_root=fixture.root)
    second_manifest = _load_json(fixture.manifest_path)
    second_version = str(second_manifest["current_version"])

    assert second_version != first_version
    assert (
        fixture.manifest_path.parent / "history" / "fixture-composite" / f"{first_version}.json"
    ).is_file()
    assert (fixture.output_directory / "versions" / first_version / "train.txt").is_file()


def test_aggressive_cross_source_dedup_is_written_to_reviewable_log(tmp_path: Path) -> None:
    fixture = build_composite_fixture(tmp_path, threshold=2)

    compose_corpus(fixture.config_path, repository_root=fixture.root)

    log_path = fixture.output_directory / "dedup-exclusions.jsonl"
    exclusions = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    cross_source = [
        exclusion for exclusion in exclusions if exclusion["reason"] == "cross_source_long_ngram"
    ]
    assert cross_source
    assert all(exclusion["compared_source_id"] == "fixture-alpha" for exclusion in cross_source)
    assert all(exclusion["overlap_token_count"] == 2 for exclusion in cross_source)


def test_cross_source_partition_leakage_hard_fails_finalization() -> None:
    training = [SourceLine("alpha", 0, "one two three four five")]
    heldout = [SourceLine("beta", 0, "zero one two three four five six")]

    with pytest.raises(ValueError, match="train/heldout leakage detected"):
        _validate_no_leakage(training, heldout, min_ngram_tokens=5)
