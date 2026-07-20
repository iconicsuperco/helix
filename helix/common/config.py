"""Centralized YAML/JSON configuration loading for Helix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import yaml

from helix.common.exceptions import HelixConfigurationError
from helix.common.paths import PathInput, config_root, resolve_repository_path

CONFIG_SECTIONS = frozenset({"model", "training", "tokenizer", "inference", "evaluation"})
ConfigFileFormat = Literal["json", "yaml"]


def load_mapping(
    path: PathInput,
    *,
    description: str = "config",
    file_format: ConfigFileFormat | None = None,
) -> dict[str, object]:
    """Load a YAML or JSON mapping from disk with contextual errors."""

    resolved_path = Path(path).expanduser().resolve()
    selected_format = file_format
    if selected_format is None:
        if resolved_path.suffix in {".yaml", ".yml"}:
            selected_format = "yaml"
        elif resolved_path.suffix == ".json":
            selected_format = "json"
        else:
            raise HelixConfigurationError(
                f"Unsupported {description} extension for {resolved_path}; expected YAML or JSON"
            )
    try:
        if selected_format == "yaml":
            loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        else:
            loaded = json.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HelixConfigurationError(f"Unable to read {description} at {resolved_path}") from error
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise HelixConfigurationError(f"Invalid {description} at {resolved_path}") from error

    if not isinstance(loaded, dict):
        raise HelixConfigurationError(
            f"{description.capitalize()} at {resolved_path} must be a mapping"
        )
    return cast(dict[str, object], loaded)


def config_path(*parts: str, root: PathInput | None = None) -> Path:
    """Resolve a path beneath the configured Helix config root."""

    base = Path(root).expanduser().resolve() if root is not None else config_root()
    return base.joinpath(*parts).resolve()


def load_config(path: PathInput, *, description: str = "config") -> dict[str, object]:
    """Load a config by repository-relative or absolute path."""

    raw_path = Path(path)
    resolved_path = (
        raw_path.expanduser().resolve()
        if raw_path.is_absolute()
        else resolve_repository_path(raw_path)
    )
    return load_mapping(resolved_path, description=description)


def load_named_config(
    section: str, name: str, *, root: PathInput | None = None
) -> dict[str, object]:
    """Load `config/<section>/<name>.yaml` using standard Helix config sections."""

    if section not in CONFIG_SECTIONS:
        expected = ", ".join(sorted(CONFIG_SECTIONS))
        raise HelixConfigurationError(
            f"Unknown config section {section!r}; expected one of: {expected}"
        )

    filename = name if name.endswith((".yaml", ".yml", ".json")) else f"{name}.yaml"
    path = config_path(section, filename, root=root)
    return load_mapping(path, description=f"{section} config")


def require_bool(values: dict[str, object], key: str, *, description: str = "config") -> bool:
    """Return a required boolean field from a loaded config mapping."""

    value = values.get(key)
    if not isinstance(value, bool):
        raise HelixConfigurationError(f"{description.capitalize()} field '{key}' must be a boolean")
    return value


def require_positive_int(
    values: dict[str, object], key: str, *, description: str = "config"
) -> int:
    """Return a required positive integer field from a loaded config mapping."""

    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a positive integer"
        )
    return value


def require_str(values: dict[str, object], key: str, *, description: str = "config") -> str:
    """Return a required non-empty string field from a loaded config mapping."""

    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a non-empty string"
        )
    return value
