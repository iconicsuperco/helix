"""Measure retained source contributions for the R4 composite corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from research.data.compose_corpus import REPOSITORY_ROOT, _load_config, _verify_source
from research.data.dedup import SourceLine, deduplicate_across_sources

DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "research/data/config/composite-gutenberg-prose-r4-v1.yaml"


@dataclass(frozen=True)
class SourceContribution:
    """One source's retained share of a composite corpus."""

    source_id: str
    retained_line_count: int
    word_count: int
    line_share: float
    word_share: float


@dataclass(frozen=True)
class BalanceReport:
    """Per-source contribution measurements and dominance finding."""

    total_retained_lines: int
    total_words: int
    dominance_threshold: float
    dominant_source_ids: tuple[str, ...]
    sources: tuple[SourceContribution, ...]


def summarize_contributions(
    sources: Sequence[tuple[str, Sequence[SourceLine]]],
    *,
    dominance_threshold: float = 0.5,
) -> BalanceReport:
    """Summarize retained line and whitespace-token contributions by source."""

    if not 0.0 < dominance_threshold < 1.0:
        raise ValueError("Dominance threshold must be between zero and one")
    if not sources or len({source_id for source_id, _ in sources}) != len(sources):
        raise ValueError("Sources must be non-empty and have unique IDs")
    line_counts = {source_id: len(records) for source_id, records in sources}
    word_counts = {
        source_id: sum(len(record.text.split()) for record in records)
        for source_id, records in sources
    }
    total_lines = sum(line_counts.values())
    total_words = sum(word_counts.values())
    if total_lines == 0 or total_words == 0:
        raise ValueError("Retained sources must contain lines and words")
    contributions = tuple(
        SourceContribution(
            source_id=source_id,
            retained_line_count=line_counts[source_id],
            word_count=word_counts[source_id],
            line_share=line_counts[source_id] / total_lines,
            word_share=word_counts[source_id] / total_words,
        )
        for source_id, _ in sources
    )
    return BalanceReport(
        total_retained_lines=total_lines,
        total_words=total_words,
        dominance_threshold=dominance_threshold,
        dominant_source_ids=tuple(
            contribution.source_id
            for contribution in contributions
            if contribution.line_share > dominance_threshold
        ),
        sources=contributions,
    )


def analyze_composition(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    dominance_threshold: float = 0.5,
) -> BalanceReport:
    """Apply the existing composition policy and measure retained source shares."""

    root = repository_root.resolve()
    config = _load_config(config_path.resolve(), root)
    verified_sources = [_verify_source(source, root) for source in config.sources]
    retained_by_source, _ = deduplicate_across_sources(
        [(source.source_id, source.records) for source in verified_sources],
        config.min_ngram_tokens,
    )
    return summarize_contributions(
        [(source.source_id, retained_by_source[source.source_id]) for source in verified_sources],
        dominance_threshold=dominance_threshold,
    )


def main() -> None:
    """Print the R4 source-balance report as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dominance-threshold", type=float, default=0.5)
    arguments = parser.parse_args()
    report = analyze_composition(
        arguments.config,
        dominance_threshold=arguments.dominance_threshold,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
