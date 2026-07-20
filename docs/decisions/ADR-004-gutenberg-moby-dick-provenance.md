# ADR-004: Corpus Provenance — Gutenberg Moby-Dick Dataset

## Status
Accepted

## Context
This ADR instantiates the dataset provenance architecture defined in ADR-003 for the Gutenberg Moby-Dick corpus. An earlier draft of this ADR recommended Project Gutenberg eBook #2701 based on catalog-metadata research alone. That recommendation was superseded during milestone M3a: live retrieval and regeneration demonstrated that the repository's existing committed processed outputs are reproducible from Project Gutenberg eBook #15, not #2701. Per ADR-003 §5 (Trust Model), this empirical result takes precedence over the earlier metadata-based recommendation.

The repository manifest (`datasets/manifests/gutenberg-moby-dick.json`) now contains fully populated, verified provenance for this dataset. This ADR records the resulting decision and defers all implementation-specific values to that manifest, per ADR-003 §2.

## Decision

**Canonical source:** Project Gutenberg eBook #15 — "Moby-Dick; or, The Whale" by Herman Melville.

**Why #2701 was not adopted:** #2701 was the original candidate based only on metadata. Empirical regeneration showed it is not the source from which the repository's committed processed artifacts were produced.

**Dataset identity:** `dataset_id = gutenberg-moby-dick` — a stable logical dataset identity per ADR-003 §3, independent of the specific Project Gutenberg edition.

**Implementation values:** The authoritative URL, SHA-256 values, encoding, retrieval metadata, and other provenance details are recorded in `datasets/manifests/gutenberg-moby-dick.json`. The manifest is the single source of truth for these implementation values.

## Consequences

- Repository provenance now matches the actual source used to generate the committed processed corpus.
- Dataset identity remains stable even if future corpus versions change.
- The architecture worked as intended: a proposal remained unaccepted until supported by empirical verification.

## Date

2026-07-20
