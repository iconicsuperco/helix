"""Verify pure generation metrics against hand-constructed text."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import distinct_n, memorization_overlap, repetition_rate


def test_repetition_rate_counts_repeated_ngram_occurrences() -> None:
    text = "a b c a b c"

    assert repetition_rate(text, n=2) == pytest.approx(2 / 5)


def test_distinct_n_counts_unique_ngrams() -> None:
    text = "a b c a b c"

    assert distinct_n(text, n=2) == pytest.approx(3 / 5)


def test_degenerate_length_returns_explicit_nan_sentinel() -> None:
    assert math.isnan(repetition_rate("too short", n=4))
    assert math.isnan(distinct_n("one", n=2))


def test_memorization_overlap_finds_planted_span_and_rejects_nonmatch() -> None:
    corpus = "zero one planted exact overlap of five tokens here seven eight"
    generated = "prefix planted exact overlap of five tokens suffix"

    findings = memorization_overlap(generated, corpus, min_overlap_tokens=5)

    assert len(findings) == 1
    assert findings[0].generated_start == 1
    assert findings[0].corpus_start == 2
    assert findings[0].token_count == 6
    assert findings[0].text == "planted exact overlap of five tokens"
    assert (
        memorization_overlap(
            "entirely unrelated generated phrase",
            corpus,
            min_overlap_tokens=3,
        )
        == []
    )
