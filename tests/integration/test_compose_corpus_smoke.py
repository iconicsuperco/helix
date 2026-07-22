"""Exercise the complete composite-corpus pipeline reproducibly."""

from __future__ import annotations

from pathlib import Path

from research.data.compose_corpus import compose_corpus
from tests.unit.data._helpers import build_composite_fixture


def test_composite_pipeline_is_byte_reproducible(tmp_path: Path) -> None:
    fixture = build_composite_fixture(tmp_path)
    manifest_path = compose_corpus(fixture.config_path, repository_root=fixture.root)
    produced_paths = (
        manifest_path,
        fixture.output_directory / "train.txt",
        fixture.output_directory / "heldout.txt",
        fixture.output_directory / "dedup-exclusions.jsonl",
    )
    first_outputs = {path: path.read_bytes() for path in produced_paths}

    compose_corpus(fixture.config_path, repository_root=fixture.root)

    assert {path: path.read_bytes() for path in produced_paths} == first_outputs
