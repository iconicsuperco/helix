"""Configurable structured logging helpers for Helix research code."""

from __future__ import annotations

import json
import logging as stdlib_logging
import os
import sys
from typing import Final, TextIO

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_LEVEL_ENV: Final[str] = "HELIX_LOG_LEVEL"
LOG_FORMAT_ENV: Final[str] = "HELIX_LOG_FORMAT"
_STANDARD_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    stdlib_logging.makeLogRecord({}).__dict__
)


class JsonFormatter(stdlib_logging.Formatter):
    """Format log records as one compact JSON object per line."""

    def format(self, record: stdlib_logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD_RECORD_FIELDS and key not in {"message", "asctime"}
            }
        )
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _coerce_level(level: str | int | None) -> int:
    raw_level = level if level is not None else os.environ.get(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL)
    if isinstance(raw_level, int):
        return raw_level
    numeric_level = stdlib_logging.getLevelName(raw_level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {raw_level}")
    return numeric_level


def configure_logging(
    *,
    level: str | int | None = None,
    json_logs: bool | None = None,
    stream: TextIO | None = None,
    force: bool = True,
) -> None:
    """Configure root logging for scripts and research jobs."""

    use_json = json_logs
    if use_json is None:
        use_json = os.environ.get(LOG_FORMAT_ENV, "json").lower() != "text"

    handler = stdlib_logging.StreamHandler(stream if stream is not None else sys.stderr)
    formatter: stdlib_logging.Formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = stdlib_logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    stdlib_logging.basicConfig(level=_coerce_level(level), handlers=[handler], force=force)


def get_logger(name: str) -> stdlib_logging.Logger:
    """Return a standard library logger for the given module name."""

    return stdlib_logging.getLogger(name)
