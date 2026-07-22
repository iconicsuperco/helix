"""Tests for R4 per-source contribution measurements."""

from __future__ import annotations

import pytest

from research.data.balance_check import summarize_contributions
from research.data.dedup import SourceLine


def test_summarize_contributions_reports_line_and_word_shares() -> None:
    report = summarize_contributions(
        [
            (
                "alpha",
                [SourceLine("alpha", 0, "one two"), SourceLine("alpha", 1, "three")],
            ),
            ("beta", [SourceLine("beta", 0, "four")]),
        ]
    )

    assert report.total_retained_lines == 3
    assert report.total_words == 4
    assert report.dominant_source_ids == ("alpha",)
    assert report.sources[0].line_share == pytest.approx(2 / 3)
    assert report.sources[0].word_share == pytest.approx(3 / 4)


def test_summarize_contributions_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        summarize_contributions(
            [("alpha", [SourceLine("alpha", 0, "word")])], dominance_threshold=1.0
        )

    with pytest.raises(ValueError, match="non-empty and have unique IDs"):
        summarize_contributions([])
