"""Shared type aliases for Helix infrastructure."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | Path
ScalarValue: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = ScalarValue | list["JSONValue"] | dict[str, "JSONValue"]
ConfigMapping: TypeAlias = dict[str, object]
