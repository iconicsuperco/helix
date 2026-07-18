"""Repository and configuration path helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, TypeAlias

from helix.common.exceptions import HelixPathError

PathInput: TypeAlias = str | Path

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_DIR_ENV: Final[str] = "HELIX_CONFIG_DIR"


def repository_root() -> Path:
    """Return the resolved repository root."""

    return REPOSITORY_ROOT


def config_root() -> Path:
    """Return the active config root, defaulting to `<repo>/config`."""

    configured = os.environ.get(CONFIG_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return REPOSITORY_ROOT / "config"


def resolve_repository_path(path: PathInput) -> Path:
    """Resolve a repository-relative path and reject paths that escape the repository."""

    raw_path = Path(path).expanduser()
    resolved_path = (
        raw_path.resolve() if raw_path.is_absolute() else (REPOSITORY_ROOT / raw_path).resolve()
    )
    if not resolved_path.is_relative_to(REPOSITORY_ROOT):
        raise HelixPathError(f"Path escapes repository root: {path}")
    return resolved_path


def relative_to_repository(path: PathInput) -> Path:
    """Return a path relative to the repository root."""

    resolved_path = resolve_repository_path(path)
    return resolved_path.relative_to(REPOSITORY_ROOT)
