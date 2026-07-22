"""Measure the existing composite tokenizer's word fragmentation by source."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenizers import Tokenizer

from research.data.cleaning import clean_source_text
from research.data.dedup import deduplicate_within_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_PATH = (
    REPOSITORY_ROOT / "research/data/tokenizers/helix-gutenberg-prose-bpe-v1/tokenizer.json"
)
MEANINGFUL_INCREASE_THRESHOLD = 0.10
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}][^\W_]+)*")


@dataclass(frozen=True)
class SourceText:
    """One source identity and its verified local raw-text path."""

    source_id: str
    path: Path


@dataclass(frozen=True)
class FragmentationMetrics:
    """Tokenizer fragmentation measurements for one source group."""

    label: str
    word_count: int
    token_count: int
    split_word_count: int
    average_tokens_per_word: float
    split_word_fraction: float


@dataclass(frozen=True)
class FragmentationReport:
    """Existing-source baseline and new-source tokenizer comparison."""

    sources: tuple[FragmentationMetrics, ...]
    existing_source_baseline: FragmentationMetrics
    new_source_aggregate: FragmentationMetrics
    relative_average_increase: float
    meaningful_increase_threshold: float
    new_sources_meaningfully_worse: bool


DEFAULT_SOURCES = (
    SourceText(
        "gutenberg-moby-dick",
        REPOSITORY_ROOT / "datasets/raw/gutenberg-moby-dick/moby-dick.txt",
    ),
    SourceText(
        "gutenberg-time-machine",
        REPOSITORY_ROOT / "research/data/sources/raw/gutenberg-time-machine/pg35.txt",
    ),
    SourceText(
        "gutenberg-pride-and-prejudice",
        REPOSITORY_ROOT / "research/data/sources/raw/gutenberg-pride-and-prejudice/pg1342.txt",
    ),
)
DEFAULT_BASELINE_SOURCE_IDS = ("gutenberg-moby-dick", "gutenberg-time-machine")
DEFAULT_NEW_SOURCE_IDS = ("gutenberg-pride-and-prejudice",)


def measure_fragmentation(
    label: str,
    text: str,
    token_count_for_word: Callable[[str], int],
) -> FragmentationMetrics:
    """Measure isolated-word token counts with one tokenizer."""

    words = [match.group(0) for match in _WORD_PATTERN.finditer(text)]
    if not words:
        raise ValueError(f"Fragmentation input {label!r} must contain at least one word")
    token_counts = [token_count_for_word(word) for word in words]
    if any(count <= 0 for count in token_counts):
        raise ValueError("Tokenizer must emit at least one token for every measured word")
    token_count = sum(token_counts)
    split_word_count = sum(count > 1 for count in token_counts)
    return FragmentationMetrics(
        label=label,
        word_count=len(words),
        token_count=token_count,
        split_word_count=split_word_count,
        average_tokens_per_word=token_count / len(words),
        split_word_fraction=split_word_count / len(words),
    )


def aggregate_fragmentation(
    label: str,
    metrics: Sequence[FragmentationMetrics],
) -> FragmentationMetrics:
    """Combine source measurements using their measured word counts as weights."""

    if not metrics:
        raise ValueError(f"Fragmentation aggregate {label!r} requires at least one source")
    word_count = sum(metric.word_count for metric in metrics)
    token_count = sum(metric.token_count for metric in metrics)
    split_word_count = sum(metric.split_word_count for metric in metrics)
    return FragmentationMetrics(
        label=label,
        word_count=word_count,
        token_count=token_count,
        split_word_count=split_word_count,
        average_tokens_per_word=token_count / word_count,
        split_word_fraction=split_word_count / word_count,
    )


def compare_fragmentation(
    source_metrics: Sequence[FragmentationMetrics],
    *,
    baseline_source_ids: Sequence[str],
    new_source_ids: Sequence[str],
    meaningful_increase_threshold: float = MEANINGFUL_INCREASE_THRESHOLD,
) -> FragmentationReport:
    """Compare weighted new-source fragmentation with the existing-source baseline."""

    if meaningful_increase_threshold < 0.0:
        raise ValueError("Meaningful increase threshold cannot be negative")
    by_label = {metric.label: metric for metric in source_metrics}
    if len(by_label) != len(source_metrics):
        raise ValueError("Fragmentation source labels must be unique")
    try:
        baseline = aggregate_fragmentation(
            "existing-source-baseline", [by_label[source_id] for source_id in baseline_source_ids]
        )
        new_sources = aggregate_fragmentation(
            "new-source-aggregate", [by_label[source_id] for source_id in new_source_ids]
        )
    except KeyError as error:
        raise ValueError(f"Missing fragmentation source: {error.args[0]}") from error
    relative_increase = new_sources.average_tokens_per_word / baseline.average_tokens_per_word - 1.0
    return FragmentationReport(
        sources=tuple(source_metrics),
        existing_source_baseline=baseline,
        new_source_aggregate=new_sources,
        relative_average_increase=relative_increase,
        meaningful_increase_threshold=meaningful_increase_threshold,
        new_sources_meaningfully_worse=relative_increase > meaningful_increase_threshold,
    )


def analyze_sources(
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    sources: Sequence[SourceText] = DEFAULT_SOURCES,
) -> FragmentationReport:
    """Load verified source text and measure it with the existing tokenizer artifact."""

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def token_count_for_word(word: str) -> int:
        return len(tokenizer.encode(word, add_special_tokens=False).ids)

    metrics: list[FragmentationMetrics] = []
    for source in sources:
        text = source.path.read_text(encoding="utf-8")
        cleaned_lines = clean_source_text(text)
        retained, _ = deduplicate_within_source(source.source_id, cleaned_lines)
        metrics.append(
            measure_fragmentation(
                source.source_id,
                "\n".join(record.text for record in retained),
                token_count_for_word,
            )
        )
    return compare_fragmentation(
        metrics,
        baseline_source_ids=DEFAULT_BASELINE_SOURCE_IDS,
        new_source_ids=DEFAULT_NEW_SOURCE_IDS,
    )


def main() -> None:
    """Print the R4 tokenizer-fragmentation report as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    arguments = parser.parse_args()
    report = analyze_sources(arguments.tokenizer)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
