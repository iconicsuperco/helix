# ADR-008: Training Corpus Provenance - Gutenberg Pride and Prejudice

## Status
Accepted

## Context
R4 expands the Helix composite corpus beyond Moby-Dick and *The Time Machine* with a
substantially larger source and a different prose register. ADR-003 requires each new dataset to
have an independent provenance decision and requires identity, version, and checksum to remain
separate.

The source was verified by live retrieval on 2026-07-22. Project Gutenberg's retrieved text
identifies eBook #1342 as *Pride and Prejudice* by Jane Austen and records English as its language.
The requested text URL resolved to the canonical cached text, returned HTTP 200 with
`text/plain; charset=utf-8`, decoded successfully as UTF-8, and produced the same SHA-256 on two
independent retrievals.

## Decision

**Canonical source:** Project Gutenberg eBook #1342, plain-text UTF-8 edition.

**Selection rationale:** This work adds Regency-era social and domestic narrative prose from a
different author, era, and register than Moby-Dick's maritime prose and *The Time Machine*'s
late-Victorian scientific prose. Its size materially expands the training corpus while retaining
an unambiguous Project Gutenberg catalog identity.

**Alternatives rejected:** The HTML and EPUB representations of eBook #1342 were rejected because
markup, illustrations, and container structure would add format-specific training noise. No other
work was attempted because this source alone produced the material R4 corpus expansion required.

**Dataset identity:** `dataset_id = gutenberg-pride-and-prejudice`.

**Version:** `pg1342-2026-02-10`, based on the "Most recently updated" value embedded in the
retrieved Project Gutenberg text.

**Canonical URL:** `https://www.gutenberg.org/cache/epub/1342/pg1342.txt`.

**SHA-256:** `74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806`,
computed locally from the 772,386 bytes actually retrieved, per ADR-003 section 6. It is an
integrity value for the recorded version and is not used as either dataset identity or version.

**Encoding:** UTF-8, confirmed by the live response content type and successful strict decoding.

**Expected local file:**
`research/data/sources/raw/gutenberg-pride-and-prejudice/pg1342.txt`, kept gitignored.

**Checksum policy:** A future retrieval that differs from the recorded SHA-256 hard-fails and
requires human review under ADR-003 sections 4, 6, and 7. It is never adopted as an automatic
version bump.

**License:** Public domain in the USA and distributed under the Project Gutenberg License;
deployment jurisdictions must verify their own status.

The authoritative retrieval command, timestamp, byte count, URL, checksum, encoding, and
upstream metadata are recorded in
`research/data/sources/manifests/gutenberg-pride-and-prejudice.json`.

## Consequences

- R4 gains verified prose from a different author, era, and narrative register.
- The raw text remains gitignored and must match the manifest checksum before composition.
- Any upstream byte change hard-fails validation pending review under ADR-003.
- This source is independently addressable and can evolve without changing composite identity.

## Date

2026-07-22
