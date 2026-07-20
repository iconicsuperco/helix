"""Pure text metrics for fixed-suite generation evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapFinding:
    """One maximal exact token overlap between generated and corpus text."""

    generated_start: int
    corpus_start: int
    token_count: int
    text: str


def _tokens(text: str) -> list[str]:
    return text.split()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def repetition_rate(text: str, n: int = 4) -> float:
    """Return repeated n-gram occurrences divided by all n-gram occurrences.

    NaN is the explicit sentinel when the text is too short to form one n-gram.
    """

    ngrams = _ngrams(_tokens(text), n)
    if not ngrams:
        return math.nan
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def distinct_n(text: str, n: int = 2) -> float:
    """Return unique n-grams divided by all n-gram occurrences.

    NaN is the explicit sentinel when the text is too short to form one n-gram.
    """

    ngrams = _ngrams(_tokens(text), n)
    if not ngrams:
        return math.nan
    return len(set(ngrams)) / len(ngrams)


def memorization_overlap(
    generated: str,
    corpus_text: str,
    min_overlap_tokens: int,
) -> list[OverlapFinding]:
    """Return maximal exact token spans meeting the configured minimum length."""

    if min_overlap_tokens <= 0:
        raise ValueError("min_overlap_tokens must be a positive integer")
    generated_tokens = _tokens(generated)
    corpus_tokens = _tokens(corpus_text)
    if len(generated_tokens) < min_overlap_tokens or len(corpus_tokens) < min_overlap_tokens:
        return []

    corpus_index: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for corpus_start in range(len(corpus_tokens) - min_overlap_tokens + 1):
        key = tuple(corpus_tokens[corpus_start : corpus_start + min_overlap_tokens])
        corpus_index[key].append(corpus_start)

    findings: list[OverlapFinding] = []
    for generated_start in range(len(generated_tokens) - min_overlap_tokens + 1):
        key = tuple(generated_tokens[generated_start : generated_start + min_overlap_tokens])
        for corpus_start in corpus_index.get(key, []):
            if (
                generated_start > 0
                and corpus_start > 0
                and generated_tokens[generated_start - 1] == corpus_tokens[corpus_start - 1]
            ):
                continue
            token_count = min_overlap_tokens
            while (
                generated_start + token_count < len(generated_tokens)
                and corpus_start + token_count < len(corpus_tokens)
                and generated_tokens[generated_start + token_count]
                == corpus_tokens[corpus_start + token_count]
            ):
                token_count += 1
            findings.append(
                OverlapFinding(
                    generated_start=generated_start,
                    corpus_start=corpus_start,
                    token_count=token_count,
                    text=" ".join(
                        generated_tokens[generated_start : generated_start + token_count]
                    ),
                )
            )
    return sorted(
        findings,
        key=lambda finding: (
            finding.generated_start,
            -finding.token_count,
            finding.corpus_start,
        ),
    )
