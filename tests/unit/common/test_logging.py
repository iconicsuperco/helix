"""Test structured logging helpers."""

from __future__ import annotations

import io
import json
import logging

from helix.common.logging import configure_logging, get_logger


def test_configure_logging_writes_json_records() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)
    logger = get_logger("helix.test")

    logger.info("hello", extra={"step": 1})

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "info"
    assert payload["logger"] == "helix.test"
    assert payload["message"] == "hello"
    assert payload["step"] == 1


def test_configure_logging_can_write_text_records() -> None:
    stream = io.StringIO()
    configure_logging(json_logs=False, stream=stream)

    logging.getLogger("helix.test").warning("careful")

    assert stream.getvalue() == "WARNING:helix.test:careful\n"
