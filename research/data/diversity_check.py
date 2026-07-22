"""Compare objective lexical and sentence statistics for the R2 and R4 corpora."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median

_WORD_PATTERN = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}][^\W_]+)*")
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:[\"'\N{RIGHT DOUBLE QUOTATION MARK}]+)?(?:\s+|$)")


@dataclass(frozen=True)
class CorpusDiversityMetrics:
    """Objective lexical and sentence-level measurements for one corpus."""

    nonempty_line_count: int
    word_count: int
    unique_word_count: int
    type_token_ratio: float
    sentence_count: int
    mean_sentence_words: float
    median_sentence_words: float
    p90_sentence_words: int


@dataclass(frozen=True)
class DiversityComparison:
    """Measured change from a baseline corpus to an expanded corpus."""

    baseline: CorpusDiversityMetrics
    expanded: CorpusDiversityMetrics
    added_unique_word_count: int
    unique_word_growth_fraction: float
    vocabulary_jaccard_distance: float
    word_count_growth_fraction: float


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def _sentence_lengths(text: str) -> list[int]:
    return [
        len(words) for fragment in _SENTENCE_BOUNDARY.split(text) if (words := _words(fragment))
    ]


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    rank = max(1, int(len(values) * percentile + 0.999999999))
    return sorted(values)[rank - 1]


def analyze_corpus(text: str) -> CorpusDiversityMetrics:
    """Measure lexical breadth and sentence-length distribution."""

    words = _words(text)
    unique_words = set(words)
    sentence_lengths = _sentence_lengths(text)
    return CorpusDiversityMetrics(
        nonempty_line_count=sum(bool(line.strip()) for line in text.splitlines()),
        word_count=len(words),
        unique_word_count=len(unique_words),
        type_token_ratio=len(unique_words) / len(words) if words else 0.0,
        sentence_count=len(sentence_lengths),
        mean_sentence_words=fmean(sentence_lengths) if sentence_lengths else 0.0,
        median_sentence_words=float(median(sentence_lengths)) if sentence_lengths else 0.0,
        p90_sentence_words=_nearest_rank(sentence_lengths, 0.90),
    )


def compare_corpora(baseline_text: str, expanded_text: str) -> DiversityComparison:
    """Compare an expanded corpus with its baseline using fixed objective metrics."""

    baseline_words = _words(baseline_text)
    expanded_words = _words(expanded_text)
    baseline_vocabulary = set(baseline_words)
    expanded_vocabulary = set(expanded_words)
    if not baseline_words or not baseline_vocabulary:
        raise ValueError("Baseline corpus must contain at least one word")
    vocabulary_union = baseline_vocabulary | expanded_vocabulary
    vocabulary_intersection = baseline_vocabulary & expanded_vocabulary
    return DiversityComparison(
        baseline=analyze_corpus(baseline_text),
        expanded=analyze_corpus(expanded_text),
        added_unique_word_count=len(expanded_vocabulary - baseline_vocabulary),
        unique_word_growth_fraction=(
            len(expanded_vocabulary - baseline_vocabulary) / len(baseline_vocabulary)
        ),
        vocabulary_jaccard_distance=(
            1.0 - len(vocabulary_intersection) / len(vocabulary_union) if vocabulary_union else 0.0
        ),
        word_count_growth_fraction=(len(expanded_words) - len(baseline_words))
        / len(baseline_words),
    )


def main() -> None:
    """Print the R2-to-R4 diversity comparison as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--expanded", required=True, type=Path)
    arguments = parser.parse_args()
    baseline_text = arguments.baseline.read_text(encoding="utf-8")
    expanded_text = arguments.expanded.read_text(encoding="utf-8")
    comparison = compare_corpora(baseline_text, expanded_text)
    print(json.dumps(asdict(comparison), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
