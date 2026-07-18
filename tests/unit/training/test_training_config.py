"""Test Forge training configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from helix.common.exceptions import HelixConfigurationError
from research.training.config import DEFAULT_CONFIG_PATH, load_training_config


def test_default_forge_config_loads() -> None:
    config = load_training_config()

    assert config.model_config_path.is_file()
    assert config.data.train_paths[0].is_file()
    assert config.data.validation_paths[0].is_file()
    assert config.data.context_length == 128
    assert config.loop.max_steps == 1000


def test_config_rejects_ambiguous_validation_sources(tmp_path: Path) -> None:
    loaded = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    values = cast(dict[str, object], loaded)
    data_values = cast(dict[str, object], values["data"])
    data_values["validation_fraction"] = 0.1
    path = tmp_path / "ambiguous.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(HelixConfigurationError, match="either validation_paths"):
        load_training_config(path)
