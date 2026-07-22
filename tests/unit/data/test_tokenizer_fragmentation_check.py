"""Tests for existing-tokenizer fragmentation measurements."""

from __future__ import annotations

import pytest

from research.data.tokenizer_fragmentation_check import (
    FragmentationMetrics,
    aggregate_fragmentation,
    compare_fragmentation,
    measure_fragmentation,
)


def test_measure_fragmentation_reports_known_word_splits() -> None:
    metrics = measure_fragmentation(
        "fixture",
        "one three seven",
        lambda word: 1 if len(word) == 3 else 2,
    )

    assert metrics.word_count == 3
    assert metrics.token_count == 5
    assert metrics.split_word_count == 2
    assert metrics.average_tokens_per_word == pytest.approx(5 / 3)
    assert metrics.split_word_fraction == pytest.approx(2 / 3)


def test_aggregate_fragmentation_is_weighted_by_word_count() -> None:
    aggregate = aggregate_fragmentation(
        "combined",
        [
            FragmentationMetrics("alpha", 2, 2, 0, 1.0, 0.0),
            FragmentationMetrics("beta", 1, 3, 1, 3.0, 1.0),
        ],
    )

    assert aggregate.word_count == 3
    assert aggregate.average_tokens_per_word == pytest.approx(5 / 3)
    assert aggregate.split_word_fraction == pytest.approx(1 / 3)


def test_compare_fragmentation_applies_predeclared_threshold() -> None:
    report = compare_fragmentation(
        [
            FragmentationMetrics("baseline", 10, 10, 0, 1.0, 0.0),
            FragmentationMetrics("new", 10, 12, 2, 1.2, 0.2),
        ],
        baseline_source_ids=["baseline"],
        new_source_ids=["new"],
        meaningful_increase_threshold=0.10,
    )

    assert report.relative_average_increase == pytest.approx(0.2)
    assert report.new_sources_meaningfully_worse is True
