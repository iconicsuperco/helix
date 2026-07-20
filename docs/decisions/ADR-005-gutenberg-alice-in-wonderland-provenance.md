# ADR-005: Evaluation Corpus Provenance — Gutenberg Alice's Adventures in Wonderland

## Status
Accepted

## Context
R1 requires an out-of-corpus dataset for measuring whether Helix behavior extends beyond the
Moby-Dick and whaling material used by the current tokenizer and training pipeline. ADR-003
requires a dedicated provenance decision before dataset-specific tooling and requires identity,
version, and checksum to remain separate.

The source was verified by live retrieval on 2026-07-20. Project Gutenberg's catalog identifies
eBook #11 as *Alice's Adventures in Wonderland* by Lewis Carroll and offers a UTF-8 plain-text
representation. The requested text URL resolved to the canonical cached text, returned HTTP 200
with `text/plain; charset=utf-8`, decoded successfully as UTF-8, and produced the same SHA-256 on
two independent retrievals.

## Decision

**Canonical source:** Project Gutenberg eBook #11, plain-text UTF-8 edition.

**Selection rationale:** This corpus is public-domain English prose, is outside the current
Moby-Dick/whaling domain, and is compact enough for repeatable CPU evaluation. Project Gutenberg
eBook #1342, *Pride and Prejudice*, was also found in the live catalog but its listed plain-text
file is substantially larger. The HTML and EPUB forms of eBook #11 were rejected because markup,
images, and container structure would introduce format-specific evaluation noise.

**Dataset identity:** `dataset_id = gutenberg-alice-in-wonderland`.

**Version:** `pg11-2025-06-26`, based on the "Most recently updated" value embedded in the
retrieved Project Gutenberg text.

**Canonical URL:** `https://www.gutenberg.org/cache/epub/11/pg11.txt`.

**SHA-256:** `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`,
computed locally from the 174,311 bytes actually retrieved, per ADR-003 §6. It is an integrity
value for the recorded version and is not used as either dataset identity or version.

**Encoding:** UTF-8, confirmed by the live response content type and successful strict decoding.

**Expected local file:**
`evaluation/datasets/raw/gutenberg-alice-in-wonderland/pg11.txt`, kept gitignored.

**License:** Public domain in the USA and distributed under the Project Gutenberg License;
deployment jurisdictions must verify their own status.

The authoritative retrieval command, timestamp, byte count, URL, checksum, encoding, and
upstream metadata are recorded in
`evaluation/datasets/manifests/gutenberg-alice-in-wonderland.json`.

## Consequences

- R1 can compare in-corpus Moby-Dick perplexity with a separately versioned literary corpus.
- The raw Alice text remains gitignored and must match the manifest checksum before evaluation.
- Any upstream byte change hard-fails validation and requires review under ADR-003 before a new
  version can be adopted.
- This dataset is evaluation-only in R1; adopting it for training would be separate R2 work.

## Date

2026-07-20
