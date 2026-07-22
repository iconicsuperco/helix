"""Tests for the narrow R2-to-R4 corpus diversity measurements."""

from __future__ import annotations

import pytest

from research.data.diversity_check import analyze_corpus, compare_corpora


def test_analyze_corpus_reports_known_sentence_distribution() -> None:
    metrics = analyze_corpus("Red fox runs. Blue fox rests. Green whale swims slowly.")

    assert metrics.word_count == 10
    assert metrics.unique_word_count == 9
    assert metrics.sentence_count == 3
    assert metrics.mean_sentence_words == pytest.approx(10 / 3)
    assert metrics.median_sentence_words == 3.0
    assert metrics.p90_sentence_words == 4


def test_compare_corpora_measures_added_vocabulary() -> None:
    comparison = compare_corpora(
        "Red fox runs. Blue fox rests.",
        "Red fox runs. Blue fox rests. Green whale swims slowly.",
    )

    assert comparison.added_unique_word_count == 4
    assert comparison.unique_word_growth_fraction == pytest.approx(4 / 5)
    assert comparison.vocabulary_jaccard_distance == pytest.approx(4 / 9)
    assert comparison.word_count_growth_fraction == pytest.approx(4 / 6)


def test_compare_corpora_rejects_empty_baseline() -> None:
    with pytest.raises(ValueError, match="Baseline corpus"):
        compare_corpora("", "Some expanded text.")
