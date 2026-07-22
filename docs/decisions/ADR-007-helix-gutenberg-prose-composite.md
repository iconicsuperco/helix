# ADR-007: Helix Gutenberg Prose Composite Corpus

## Status
Accepted

## Context
R2 replaces the single-source training-corpus assumption with a versioned composition layer while
preserving the existing Moby-Dick pipeline and artifacts. ADR-003 requires source identity,
version, and checksum to remain independent and requires mixed-corpus relationships to be
expressed outside individual source provenance records.

Moby-Dick provides nineteenth-century maritime literary prose. *The Time Machine* adds a different
author and a late-Victorian scientific/speculative register while remaining small enough for the
current eager, laptop-scale pipeline. The R1 Alice corpus remains evaluation-only so that Helix
retains a genuinely out-of-corpus measurement.

## Decision

**Composite identity:** `dataset_id = helix-gutenberg-prose`, distinct from every constituent
source identity.

**Constituent sources:** `gutenberg-moby-dick` and `gutenberg-time-machine`, in that deterministic
composition order. Their authoritative source versions and raw checksums remain in their own
manifests and are referenced by the composite manifest.

**Composition policy:** The versioned configuration at
`research/data/config/composite-gutenberg-prose-v1.yaml` is loaded through
`helix.common.config`. One shared cleaning procedure removes Project Gutenberg boilerplate and
normalizes Unicode, line endings, control characters, and horizontal whitespace. Exact duplicate
lines are removed within each source. Long shared token sequences are removed from later sources
using the configured cross-source n-gram threshold. Every exclusion is written to a reviewable,
deterministic log.

The train/held-out assignment is computed over the full retained corpus using each line's SHA-256,
not independently per source. Finalization hard-fails if exact or configured long n-gram leakage
is detected across source boundaries. Composite versions include a semantic configuration
fingerprint, and a policy change archives the previous outputs rather than silently replacing an
indistinguishable version.

**Tokenizer migration:** Retrain the existing byte-level BPE algorithm against the composite
training split because the added source changes token-frequency evidence. The result is stored as
the new, distinct `helix-gutenberg-prose-bpe` tokenizer artifact and records the composite version
and training checksum in its own manifest. The existing Moby-Dick-only tokenizer config and
artifact remain unchanged and continue to be the default for existing checkpoints and training
configs.

The generated composite version, source lineage, output checksums, and dedup log checksum are
authoritative in `datasets/manifests/composite/helix-gutenberg-prose.json`; implementation values
are not duplicated here.

## Consequences

- Helix gains a reproducible multi-source training corpus without replacing the A1 corpus.
- Every source remains independently traceable to an Accepted provenance ADR and verified raw
  checksum.
- Deduplication and partition decisions are deterministic and reviewable.
- Any source checksum mismatch, non-Accepted ADR, or cross-source partition leakage blocks
  finalization.
- Existing checkpoints continue to resolve the original tokenizer artifact unless explicitly
  configured for a future composite-corpus run.

## Date

2026-07-22
