"""Tests for deterministic within- and cross-source deduplication."""

from __future__ import annotations

from research.data.dedup import (
    SourceLine,
    deduplicate_across_sources,
    deduplicate_within_source,
)


def test_within_source_dedup_keeps_first_line_and_logs_every_duplicate() -> None:
    retained, exclusions = deduplicate_within_source("alpha", ["first", "same", "same", "first"])

    assert [record.text for record in retained] == ["first", "same"]
    assert [exclusion.line_index for exclusion in exclusions] == [2, 3]
    assert {exclusion.reason for exclusion in exclusions} == {"within_source_exact_line"}


def test_cross_source_dedup_removes_planted_long_overlap() -> None:
    alpha = [SourceLine("alpha", 0, "one two three four five six")]
    beta = [SourceLine("beta", 0, "prefix one two three four five suffix")]

    retained, exclusions = deduplicate_across_sources(
        [("alpha", alpha), ("beta", beta)], min_ngram_tokens=5
    )

    assert retained["alpha"] == alpha
    assert retained["beta"] == []
    assert len(exclusions) == 1
    assert exclusions[0].compared_source_id == "alpha"
    assert exclusions[0].overlap_token_count == 5
    assert exclusions[0].overlap_sha256 is not None


def test_cross_source_dedup_preserves_non_overlapping_sources() -> None:
    alpha = [SourceLine("alpha", 0, "one two three four five")]
    beta = [SourceLine("beta", 0, "six seven eight nine ten")]

    retained, exclusions = deduplicate_across_sources(
        [("alpha", alpha), ("beta", beta)], min_ngram_tokens=3
    )

    assert retained == {"alpha": alpha, "beta": beta}
    assert exclusions == []


def test_aggressive_threshold_exclusions_remain_reviewable() -> None:
    alpha = [SourceLine("alpha", 0, "shared phrase alpha")]
    beta = [SourceLine("beta", 4, "shared phrase beta")]

    _, exclusions = deduplicate_across_sources(
        [("alpha", alpha), ("beta", beta)], min_ngram_tokens=2
    )

    assert [exclusion.to_mapping() for exclusion in exclusions] == [
        {
            "line_index": 4,
            "reason": "cross_source_long_ngram",
            "source_id": "beta",
            "text": "shared phrase beta",
            "compared_source_id": "alpha",
            "overlap_token_count": 2,
            "overlap_sha256": exclusions[0].overlap_sha256,
        }
    ]
