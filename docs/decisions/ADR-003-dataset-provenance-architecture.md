# ADR-003: Dataset Provenance Architecture for Helix

- Status: Proposed
- Date: 2026-07-20

## Context
A technical audit of the repository found that the data pipeline could not regenerate
its raw corpus from a clean checkout: the existing dataset manifest recorded SHA-256
checksums for processed datasets, but not for the raw source, and no canonical download
URL or verified checksum existed for the raw corpus. An initial attempt to patch this
dataset-by-dataset (starting with the Gutenberg Moby-Dick corpus) surfaced a deeper gap:
the project had no explicit, project-wide definition of what "provenance" means for a
dataset, and an early draft of this ADR itself conflated three concepts — dataset
identity, dataset version, and dataset checksum — that must be treated as distinct.

This ADR defines the required architecture once, at the project level, so that every
current and future dataset inherits a consistent provenance standard rather than each
dataset inventing its own ad hoc conventions.

This ADR defines architecture only. It makes no claims about any specific corpus.
The Gutenberg Moby-Dick corpus is referenced here only as an illustrative example; the
binding provenance decision for that dataset is recorded separately in ADR-004.

## Decision

### 1. Required provenance information
Every dataset onboarded into Helix must have a provenance record containing, at minimum:
- **Identity** — a stable, unique dataset identity that persists across all versions
  of that dataset (see below).
- **Version** — an explicit marker of which evolution point of the dataset is in use,
  distinct from identity and distinct from checksum.
- **Checksum** — a cryptographic hash (SHA-256) proving the integrity of the specific
  bytes retrieved for a specific version. A checksum proves "these are the correct
  bytes" and nothing more; it is never used as a stand-in for version or identity.
- **Canonical source** — an explicit, resolvable location the raw data was obtained from.
- **Source selection rationale** — why this source was chosen over any plausible
  alternative, especially when the same underlying work exists under multiple
  catalog entries, editions, or mirrors.
- **Encoding** — explicitly recorded, never assumed.
- **License / usage terms.**
- **Retrieval method and timestamp** — how and when the raw data was obtained, so
  retrieval is auditable and repeatable.
- **Processed-dataset lineage** — a recorded link from any processed/derived artifact
  back to the specific raw version it was derived from, plus a checksum confirming
  the integrity of that raw data at the moment of derivation.

A dataset missing any of the above is, by definition, not reproducible, and must be
documented as such rather than silently assumed complete.

### 2. Identity, Version, and Checksum are three distinct concepts
These must never be treated as interchangeable:
- **Identity** answers: *which logical dataset is this, regardless of time or content
  changes.* It is permanent — assigned once, never reassigned, independent of
  filename, URL, checksum, or version.
- **Version** answers: *which evolution point of that dataset is currently in use.*
  It is an explicit, deliberately assigned marker (a sequential tag, a date-based
  release marker, or an upstream-provided version). It is not derived from, and is
  never equal to, a checksum.
- **Checksum** answers: *do these specific bytes match what this version is supposed
  to contain.* It is evidence of integrity for a given version; it does not itself
  define what a version is.

A dataset has exactly one identity for its entire lifetime in the project, but may
have many versions over time, each with its own checksum.

### 3. Manifest / schema requirements
Every dataset must expose the concepts above in some structured, machine-readable
form. The following is an **example reference schema only** — it illustrates one way
to satisfy this architecture, not the only permitted implementation. Future
implementations may use a different file format, field layout, or storage mechanism,
provided identity, version, and checksum remain three distinct, independently
recorded values.

```json
{
  "dataset_id": "<stable logical identity — never changes across versions>",
  "display_name": "<human-readable name>",
  "current_version": "<explicit version marker>",
  "canonical_source": {
    "url": "<resolvable download location>",
    "selection_rationale": "<why this source, vs alternatives considered>",
    "retrieved_at": "<ISO-8601 timestamp of actual retrieval>",
    "retrieval_method": "<how it was obtained>"
  },
  "raw": {
    "expected_path": "<local path convention>",
    "encoding": "<confirmed encoding>",
    "version": "<version this raw file represents>",
    "sha256": "<checksum proving integrity of this version's bytes>",
    "gitignored": true
  },
  "license": "<terms>",
  "processed": [
    {
      "name": "<e.g. train, heldout>",
      "path": "<local path>",
      "derived_from_raw_version": "<which raw version this was produced from>",
      "derived_from_raw_sha256": "<integrity snapshot at derivation time>",
      "sha256": "<checksum of this processed file's own bytes>"
    }
  ],
  "fallback_sources": []
}
```

### 4. Versioning strategy
- If a dataset is re-fetched and the checksum matches the recorded value for the
  current version, this confirms the same version was retrieved correctly — no
  version change, no review required.
- If a re-fetch produces a checksum that does not match the recorded value, this is
  **not automatically a new version.** It is either an integrity failure or an
  unreviewed upstream change, and must hard-fail pending human review. Only after
  review results in a deliberate decision to adopt new content does a new version
  marker get assigned, with its own new checksum recorded against it. The prior
  version and checksum remain in history rather than being overwritten.

### 5. Trust model
- No provenance value (checksum, URL resolution, encoding, or version assignment)
  may be asserted by an LLM from memory, inference, or pattern-matching against
  similar known sources. Every such value must come from a real, reproducible action
  taken against the live source, by a process with actual network access.
- If a retrieval cannot be performed in a given execution environment, the correct
  behavior is an explicit, reported blocker — never a plausible-looking placeholder.
- A checksum mismatch against a claimed version is treated as untrusted until
  reviewed; the pipeline must hard-fail rather than silently proceed.

### 6. Checksum policy
- SHA-256 for all raw and processed files, no exceptions, no weaker algorithms.
- Computed only from bytes actually retrieved or produced locally — never
  transcribed from a third party's own claimed checksum.
- Recorded against a specific version marker, never floating free of one.

### 7. Update policy if an upstream source changes
- A checksum mismatch on re-fetch is a hard failure, never silently reinterpreted
  as an automatic version bump.
- Resolving it requires: diffing old vs. new content, an explicit recorded decision
  (ADR or ADR amendment) on whether to adopt the new content, and only then
  assigning a new version marker with its own new checksum.
- Silent auto-adoption of new upstream content — via automatic version bump or
  in-place checksum overwrite — is never acceptable.

### 8. Canonical source selection policy
When a work exists under multiple plausible sources, editions, or catalog numbers:
- Enumerate the alternatives actually found.
- Document why each rejected alternative was rejected.
- Record the rationale in the dataset's own dedicated ADR, not in code comments or
  commit messages.
- Prefer sources with independent, checkable identity over sources identified only
  by a generic title.
- Source selection determines identity and initial version, not checksum — the
  checksum is computed only after a source and version are decided and the
  corresponding bytes retrieved.

### 9. Multi-dataset support
- Each dataset's provenance record is independently addressable by its `dataset_id`.
- No dataset's record may reference another dataset's fields directly; cross-dataset
  relationships (e.g. mixed-corpus training) are expressed at the training-config
  level, not the provenance-record level.

### 10. Processed datasets in relation to raw datasets
- Every processed artifact's provenance record must reference both the raw
  dataset's **version** (lineage: which evolution point it came from) and a
  **checksum** snapshot (integrity: confirming the raw bytes were intact at
  derivation time). These serve different purposes and must not be collapsed into
  a single field.
- If the raw dataset's version is updated, every processed artifact still
  referencing the prior version must be structurally identifiable as stale by
  comparing version references, not by manual tracking.
- Regenerating processed data must be possible from a given raw version plus a
  recorded, versioned processing procedure.

### 11. Tokenizer artifacts in relation to processed datasets
- A trained tokenizer artifact must record which processed dataset version it was
  trained against, plus a checksum confirming the integrity of that processed data
  at training time — the same identity/version/checksum separation, one level
  further down the lineage chain (raw → processed → tokenizer).
- This lineage link is not currently confirmed to exist in the repository and is
  flagged here as an open gap, not resolved by this ADR.
- Any future retraining of the tokenizer against updated data produces a new
  tokenizer artifact identity with its own version, never an in-place overwrite.

### 12. How future datasets inherit this architecture
- Every new dataset requires its own dedicated ADR covering: canonical source
  selection and rationale, identity assignment, initial version marker, expected
  URL/filename/encoding, checksum, gitignore posture, and dataset-specific risks.
- No fetch/verification tooling may be built against a dataset until its dedicated
  ADR is approved — provenance is decided first, automation follows.
- A dataset is considered onboarded correctly only when identity, version, and
  checksum are each independently and correctly recorded — not merely when some
  hash exists somewhere in its record.

## Consequences

**Positive:**
- Every dataset in the project has a consistent, auditable provenance trail.
- Silent, unreviewed corpus drift becomes structurally detectable rather than
  dependent on manual vigilance.
- Future datasets inherit a settled standard instead of requiring a fresh policy
  debate each time.

**Negative / accepted tradeoffs:**
- Requiring three distinct fields (identity, version, checksum) where a single hash
  is often used casually elsewhere adds conceptual overhead and requires discipline
  from every implementer.
- Making the manifest schema explicitly non-mandatory (any structure is acceptable
  provided the concepts are preserved) means no single mechanical validator can
  enforce this architecture across all future implementations; enforcement depends
  on review discipline until/unless a shared validation library is built.
- The raw → processed → tokenizer lineage chain is only as strong as its weakest
  recorded link; if any stage skips recording its parent's version and checksum
  separately, the chain breaks silently.
