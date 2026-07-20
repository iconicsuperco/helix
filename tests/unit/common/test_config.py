"""Test shared configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helix.common.config import (
    config_path,
    load_config,
    load_mapping,
    load_named_config,
    require_positive_int,
)
from helix.common.exceptions import HelixConfigurationError, HelixPathError
from helix.common.paths import repository_root, resolve_repository_path


def test_load_mapping_reads_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump({"answer": 42}), encoding="utf-8")

    assert load_mapping(path)["answer"] == 42


def test_load_mapping_honors_explicit_format_for_nonstandard_extension(tmp_path: Path) -> None:
    path = tmp_path / "sample.config"
    path.write_text('{"answer": 42}\n', encoding="utf-8")

    assert load_mapping(path, file_format="json")["answer"] == 42


def test_load_mapping_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(HelixConfigurationError):
        load_mapping(path)


def test_load_named_config_uses_standard_sections(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tiny.yaml").write_text("n_layer: 1\n", encoding="utf-8")

    assert load_named_config("model", "tiny", root=tmp_path) == {"n_layer": 1}


def test_load_named_config_rejects_unknown_section() -> None:
    with pytest.raises(HelixConfigurationError):
        load_named_config("unknown", "tiny")


def test_load_config_resolves_repository_relative_paths() -> None:
    values = load_config("config/model/transformer.yaml")

    assert require_positive_int(values, "n_layer") == 6


def test_repository_path_rejects_escapes() -> None:
    with pytest.raises(HelixPathError):
        resolve_repository_path("../outside")


def test_config_path_defaults_to_repo_config() -> None:
    assert (
        config_path("model", "transformer.yaml")
        == repository_root() / "config" / "model" / "transformer.yaml"
    )
