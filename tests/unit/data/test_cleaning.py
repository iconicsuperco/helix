"""Tests for the shared source cleaning policy."""

from __future__ import annotations

import pytest

from research.data.cleaning import clean_source_text


def test_clean_source_text_strips_boilerplate_and_normalizes_text() -> None:
    text = (
        "Header\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\r\n"
        "  Cafe\u0301\t\tstory\x00  \r\n"
        "Second   line\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\r\n"
        "Footer\r\n"
    )

    assert clean_source_text(text) == ["Café story", "Second line"]


def test_clean_source_text_requires_gutenberg_markers() -> None:
    with pytest.raises(ValueError, match="START and END markers"):
        clean_source_text("Unmarked source text")
