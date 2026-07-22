"""Apply the shared deterministic cleaning policy to source text."""

from __future__ import annotations

import re
import unicodedata

from research.tokenizer.prepare_corpus import _strip_control_characters

CLEANING_VERSION = "gutenberg-text-v1"
_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


def clean_source_text(text: str) -> list[str]:
    """Return normalized content lines between Project Gutenberg markers."""

    normalized = unicodedata.normalize(
        "NFC",
        _strip_control_characters(text.replace("\r\n", "\n").replace("\r", "\n")),
    )
    lines = normalized.split("\n")
    start_index = next(
        (index for index, line in enumerate(lines) if _START_MARKER in line.upper()),
        None,
    )
    end_index = next(
        (
            index
            for index, line in enumerate(lines)
            if start_index is not None and index > start_index and _END_MARKER in line.upper()
        ),
        None,
    )
    if start_index is None or end_index is None:
        raise ValueError("Source text must contain Project Gutenberg START and END markers")

    return [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in lines[start_index + 1 : end_index]
    ]
