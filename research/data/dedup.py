"""Deterministically remove exact and long cross-source duplicate text."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLine:
    """One cleaned source line with stable provenance coordinates."""

    source_id: str
    line_index: int
    text: str


@dataclass(frozen=True)
class DedupExclusion:
    """A reviewable record describing one removed source line."""

    source_id: str
    line_index: int
    text: str
    reason: str
    compared_source_id: str | None = None
    overlap_token_count: int | None = None
    overlap_sha256: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        values: dict[str, object] = {
            "line_index": self.line_index,
            "reason": self.reason,
            "source_id": self.source_id,
            "text": self.text,
        }
        if self.compared_source_id is not None:
            values["compared_source_id"] = self.compared_source_id
        if self.overlap_token_count is not None:
            values["overlap_token_count"] = self.overlap_token_count
        if self.overlap_sha256 is not None:
            values["overlap_sha256"] = self.overlap_sha256
        return values


@dataclass(frozen=True)
class _NgramOwner:
    source_id: str
    tokens: tuple[str, ...]


def _digest_tokens(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def deduplicate_within_source(
    source_id: str,
    lines: Sequence[str],
) -> tuple[list[SourceLine], list[DedupExclusion]]:
    """Keep each source's first exact line and log later duplicates."""

    retained: list[SourceLine] = []
    exclusions: list[DedupExclusion] = []
    seen: set[str] = set()
    for line_index, text in enumerate(lines):
        if text in seen:
            exclusions.append(
                DedupExclusion(
                    source_id=source_id,
                    line_index=line_index,
                    text=text,
                    reason="within_source_exact_line",
                )
            )
            continue
        seen.add(text)
        retained.append(SourceLine(source_id=source_id, line_index=line_index, text=text))
    return retained, exclusions


def _flatten_records(records: Sequence[SourceLine]) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    record_positions: list[int] = []
    for record_position, record in enumerate(records):
        record_tokens = record.text.split()
        tokens.extend(record_tokens)
        record_positions.extend([record_position] * len(record_tokens))
    return tokens, record_positions


def _index_ngrams(
    index: dict[str, _NgramOwner],
    source_id: str,
    records: Sequence[SourceLine],
    threshold: int,
) -> None:
    tokens, _ = _flatten_records(records)
    for start in range(len(tokens) - threshold + 1):
        ngram = tuple(tokens[start : start + threshold])
        index.setdefault(_digest_tokens(ngram), _NgramOwner(source_id=source_id, tokens=ngram))


def deduplicate_across_sources(
    sources: Sequence[tuple[str, Sequence[SourceLine]]],
    min_ngram_tokens: int,
) -> tuple[dict[str, list[SourceLine]], list[DedupExclusion]]:
    """Drop later-source lines participating in an earlier source's matching n-gram."""

    if min_ngram_tokens <= 0:
        raise ValueError("Cross-source deduplication threshold must be positive")

    ngram_index: dict[str, _NgramOwner] = {}
    retained_by_source: dict[str, list[SourceLine]] = {}
    exclusions: list[DedupExclusion] = []
    for source_id, source_records in sources:
        records = list(source_records)
        tokens, record_positions = _flatten_records(records)
        excluded_positions: dict[int, _NgramOwner] = {}
        for start in range(len(tokens) - min_ngram_tokens + 1):
            ngram = tuple(tokens[start : start + min_ngram_tokens])
            owner = ngram_index.get(_digest_tokens(ngram))
            if owner is None or owner.tokens != ngram:
                continue
            for position in set(record_positions[start : start + min_ngram_tokens]):
                excluded_positions.setdefault(position, owner)

        retained = [
            record for position, record in enumerate(records) if position not in excluded_positions
        ]
        for position in sorted(excluded_positions):
            record = records[position]
            owner = excluded_positions[position]
            exclusions.append(
                DedupExclusion(
                    source_id=source_id,
                    line_index=record.line_index,
                    text=record.text,
                    reason="cross_source_long_ngram",
                    compared_source_id=owner.source_id,
                    overlap_token_count=min_ngram_tokens,
                    overlap_sha256=_digest_tokens(owner.tokens),
                )
            )
        retained_by_source[source_id] = retained
        _index_ngrams(ngram_index, source_id, retained, min_ngram_tokens)
    return retained_by_source, exclusions


def find_partition_leakage(
    training: Sequence[SourceLine],
    heldout: Sequence[SourceLine],
    min_ngram_tokens: int,
) -> list[tuple[str, str, str]]:
    """Return exact or long n-gram leakage evidence across source boundaries."""

    if min_ngram_tokens <= 0:
        raise ValueError("Cross-source leakage threshold must be positive")
    findings: list[tuple[str, str, str]] = []
    heldout_exact = {(line.source_id, line.text) for line in heldout}
    for line in training:
        for heldout_source_id, heldout_text in heldout_exact:
            if line.source_id != heldout_source_id and line.text == heldout_text:
                findings.append((line.source_id, heldout_source_id, "exact_line"))

    heldout_by_source: dict[str, list[SourceLine]] = {}
    for line in heldout:
        heldout_by_source.setdefault(line.source_id, []).append(line)
    heldout_ngrams: dict[str, set[str]] = {}
    for source_id, records in heldout_by_source.items():
        tokens, _ = _flatten_records(records)
        heldout_ngrams[source_id] = {
            _digest_tokens(tokens[start : start + min_ngram_tokens])
            for start in range(len(tokens) - min_ngram_tokens + 1)
        }

    training_by_source: dict[str, list[SourceLine]] = {}
    for line in training:
        training_by_source.setdefault(line.source_id, []).append(line)
    for training_source_id, records in training_by_source.items():
        tokens, _ = _flatten_records(records)
        digests = {
            _digest_tokens(tokens[start : start + min_ngram_tokens])
            for start in range(len(tokens) - min_ngram_tokens + 1)
        }
        for heldout_source_id, heldout_digests in heldout_ngrams.items():
            if training_source_id == heldout_source_id:
                continue
            if digests.intersection(heldout_digests):
                findings.append((training_source_id, heldout_source_id, "long_ngram"))
    return sorted(set(findings))
