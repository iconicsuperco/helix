# ADR-009: Helix Gutenberg Prose R4 Composite Corpus

## Status
Accepted

## Context
R4 prepares a materially larger and more varied corpus for a future scaling experiment without
changing the generic R2 composition machinery. The R2 composite contains Moby-Dick and *The Time
Machine*, with 19,729 training lines and 2,236 held-out lines. ADR-003 requires each source to
retain independent identity, version, and checksum lineage.

ADR-008 accepts Project Gutenberg eBook #1342, *Pride and Prejudice*, as a verified source. Its
Regency-era social and domestic narrative register differs from Moby-Dick's maritime prose and
*The Time Machine*'s late-Victorian scientific prose.

## Decision

**Composite identity:** Retain `dataset_id = helix-gutenberg-prose` and create declared version
`r4-v1` through `research/data/config/composite-gutenberg-prose-r4-v1.yaml`.

**Constituent sources:** Compose `gutenberg-moby-dick`, `gutenberg-time-machine`, and
`gutenberg-pride-and-prejudice`, in that deterministic order. Each source has an Accepted
provenance ADR and a raw checksum verified by the existing composition pipeline.

**Generated version:** The unchanged composition machinery produced
`r4-v1-688ecae1740e`, with composition fingerprint
`688ecae1740ead55a24738c015c5c6db743411aacb9376971c8be13da5ebabec`. It contains
29,927 training lines and 3,371 held-out lines, increases of 10,198 and 1,135 respectively over
R2. A second run with the identical configuration reproduced the manifest, training split,
held-out split, and deduplication log byte-for-byte.

**Diversity evidence:** Against the R2 training split, the R4 training split increases measured
word count from 224,244 to 339,116 (51.23%) and unique case-folded word types from 17,839 to
19,957, adding 2,118 types (11.87%). Vocabulary Jaccard distance is 0.1061. Mean, median, and
90th-percentile sentence lengths shift from 21.22, 16, and 46 words to 19.87, 15, and 43 words.
These lexical additions and distributional changes demonstrate measurable diversity rather than
source-count-only diversity.

**Balance decision:** After existing within- and cross-source deduplication, Moby-Dick contributes
19,187 lines (57.62%) and 212,502 whitespace-delimited words (57.13%); *The Time Machine*
contributes 2,778 lines (8.34%) and 32,453 words (8.73%); *Pride and Prejudice* contributes 11,333
lines (34.04%) and 126,985 words (34.14%). The mild Moby-Dick majority is accepted. No source
approaches a two-thirds share, the new source supplies more than one-third of the corpus, and the
existing composition schema deliberately retains complete verified works without a capping or
subsampling control. Adding such a control would change the generic composition policy outside
R4's scope.

**Tokenizer decision:** Reuse the existing `helix-gutenberg-prose-bpe-v1` artifact unchanged.
Measured in isolation by word, the existing-source weighted baseline averages 1.2905 tokens per
word with 21.85% of words split, while *Pride and Prejudice* averages 1.3527 tokens per word with
23.70% split. The 4.83% average-token increase is below the predeclared 10% threshold for a
meaningful fragmentation regression, so retraining is not justified.

The authoritative current output checksums and source lineage remain in
`datasets/manifests/composite/helix-gutenberg-prose.json`. The byte-identical R2 outputs and
manifest are retained in the version-history paths produced by the existing composition policy.

## Consequences

- The current composite training split is 51.7% larger by line count than R2 and contains a
  materially different author, era, and narrative register.
- The existing tokenizer remains the selected R4 tokenizer, avoiding unjustified artifact churn.
- R2's config, source records, corpus bytes, and tokenizer artifact remain preserved unchanged.
- The R4 corpus and selected tokenizer are ready to serve as input to a separately approved R5
  scaling experiment.

## Date

2026-07-22
