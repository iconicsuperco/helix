# ADR-006: Training Corpus Provenance — Gutenberg The Time Machine

## Status
Accepted

## Context
R2 expands Helix from a single-source Moby-Dick training corpus to a deliberately varied
multi-source prose corpus. ADR-003 requires each new dataset to have an independent provenance
decision and requires identity, version, and checksum to remain separate.

The source was verified by live retrieval on 2026-07-22. Project Gutenberg's catalog identifies
eBook #35 as *The Time Machine* by H. G. Wells and offers a UTF-8 plain-text representation. The
requested text URL resolved to the canonical cached text, returned HTTP 200 with
`text/plain; charset=utf-8`, decoded successfully as UTF-8, and produced the same SHA-256 on two
independent retrievals.

## Decision

**Canonical source:** Project Gutenberg eBook #35, plain-text UTF-8 edition.

**Selection rationale:** This work adds a different author and late-Victorian
scientific/speculative register while remaining compact public-domain English narrative prose
suitable for laptop-scale training. It broadens the corpus without removing Moby-Dick.

**Alternatives rejected:** Project Gutenberg eBook #11, *Alice's Adventures in Wonderland*, was
kept out of training so it remains a genuinely out-of-corpus R1 evaluation dataset. Project
Gutenberg eBook #1342, *Pride and Prejudice*, was considered but its cataloged plain-text file is
substantially larger for this milestone. The HTML and EPUB representations of eBook #35 were
rejected because markup, images, and container structure would add format-specific training
noise.

**Dataset identity:** `dataset_id = gutenberg-time-machine`.

**Version:** `pg35-2026-06-16`, based on the "Most recently updated" value embedded in the
retrieved Project Gutenberg text.

**Canonical URL:** `https://www.gutenberg.org/cache/epub/35/pg35.txt`.

**SHA-256:** `2892e919000e17c83e1dac51b30f4675db50536b644d7579fe8a89bb399a9bdc`,
computed locally from the 204,384 bytes actually retrieved, per ADR-003 §6. It is an integrity
value for the recorded version and is not used as either dataset identity or version.

**Encoding:** UTF-8, confirmed by the live response content type and successful strict decoding.

**Expected local file:** `research/data/sources/raw/gutenberg-time-machine/pg35.txt`, kept
gitignored.

**Checksum policy:** A future retrieval that differs from the recorded SHA-256 hard-fails and
requires human review under ADR-003 §§4, 6, and 7. It is never adopted as an automatic version
bump.

**License:** Public domain in the USA and distributed under the Project Gutenberg License;
deployment jurisdictions must verify their own status.

The authoritative retrieval command, timestamp, byte count, URL, checksum, encoding, and
upstream metadata are recorded in
`research/data/sources/manifests/gutenberg-time-machine.json`.

## Consequences

- R2 gains a verified training source with a different author and register from Moby-Dick.
- The raw text remains gitignored and must match the manifest checksum before composition.
- Any upstream byte change hard-fails validation pending review under ADR-003.
- This source is independently addressable and can evolve without changing composite identity.

## Date

2026-07-22
