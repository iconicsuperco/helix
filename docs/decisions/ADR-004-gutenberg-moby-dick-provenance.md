# ADR-004: Corpus Provenance — Gutenberg Moby-Dick Dataset


- Status: Proposed — pending real (not simulated) retrieval verification
- Date: 2026-07-20


## Context
This ADR instantiates the architecture defined in ADR-003 for exactly one dataset:
the Gutenberg Moby-Dick corpus currently referenced by
`datasets/manifests/gutenberg-moby-dick.json`. A prior implementation attempt (Codex,
milestone M3a) correctly refused to fabricate a canonical download URL or checksum
when neither existed in a trustworthy form in the repository. This ADR records the
provenance decision that attempt was blocked on. It does not itself perform any
retrieval — no value below marked UNVERIFIED may be treated as fact until confirmed
by an actual network-connected retrieval process.


**Labeling used throughout:**
- **VERIFIED** — confirmed via live web search, independently checkable.
- **UNVERIFIED** — not confirmed by any real retrieval; must be confirmed before
 this ADR can move from Proposed to Accepted.
- **RECOMMENDATION** — a judgment call, open for amendment.


## Decision


### 1. Canonical source
RECOMMENDATION: Project Gutenberg eBook **#2701** — "Moby Dick; Or, The Whale," by
Herman Melville.


### 2. Selection rationale
VERIFIED: At least three distinct Project Gutenberg catalog entries exist for
"Moby Dick": #2701, #15, and #9147.
VERIFIED: #15 appears in search results to be an early (1991) file that is a
chapter-heading reference list rather than the complete novel.
VERIFIED: #9147 is associated with a separate, copyrighted audio-performance
production of the text.
RECOMMENDATION: #2701 is the standard, full-text, modern-transcription entry and the
one most likely to align with common usage of "Gutenberg Moby Dick" as a corpus
elsewhere. This is a judgment call, not an independently verified claim of textual
superiority.


### 3. Alternatives considered and rejected


| Alternative | Status | Disposition |
|---|---|---|
| PG #15 | VERIFIED to exist; appears incomplete/fragmentary | Rejected — unsuitable as a full-text training corpus |
| PG #9147 | VERIFIED to exist, distinct edition | Rejected — no clear advantage over #2701, adds ambiguity |
| Standard Ebooks edition | VERIFIED to exist, actively maintained project | Rejected for now (RECOMMENDATION) — different transcription/formatting pipeline than raw Project Gutenberg text; revisit if PG formatting proves inconvenient |
| Internet Archive mirror | VERIFIED to exist as alternate host | Not canonical; recorded as fallback per ADR-003 §7, contingent on independent verification against the primary checksum |


### 4. Dataset identity
RECOMMENDATION: `dataset_id = gutenberg-moby-dick-pg2701`, per ADR-003 §3 (identity
independent of filename, URL, checksum, or version; namespaced to disambiguate from
other PG editions of the same title).


### 5. Expected download URL
UNVERIFIED — action required: candidate `https://www.gutenberg.org/cache/epub/2701/pg2701.txt`,
following Project Gutenberg's standard plain-text URL convention. This has not been
fetched or confirmed to resolve. Must be verified by a process with real network
access before this ADR moves to Accepted.


### 6. Expected filename
RECOMMENDATION: `datasets/raw/gutenberg-moby-dick/moby-dick.txt`, consistent with
the path already present in the existing manifest.


### 7. Expected encoding
UNVERIFIED: not to be assumed as UTF-8 by default. Must be confirmed at actual
retrieval time and recorded explicitly, per ADR-003 §1's blanket requirement — no
dataset-specific exception.


### 8. Raw corpus gitignore posture
RECOMMENDATION: gitignored, per ADR-003's general policy — no dataset-specific
exception needed.


### 9. Checksum policy
Inherits ADR-003 §6 directly — SHA-256, computed only from an actually-retrieved
file. No dataset-specific variance.


### 10. Update policy
Inherits ADR-003 §7 directly — no dataset-specific variance. Any future re-fetch
mismatch triggers the hard-fail-plus-review process defined there, recorded as an
amendment to this ADR rather than a silent manifest edit.


### 11. Open item carried forward from the original audit
UNVERIFIED, flagged not resolved: it is not confirmed whether the tokenizer artifact
and processed datasets already committed to the repository (from milestones A1–A3)
were derived from PG #2701 specifically, or from an unknown/undocumented source.
This ADR proposes #2701 as the forward-looking canonical source; it does not
retroactively verify what the existing committed artifacts were built from.
Reconciling this is a distinct follow-up action item, not resolved by adopting this
ADR.


## Consequences


**Positive:**
- Establishes a defensible, documented choice among three ambiguous PG catalog
 entries rather than leaving "Moby Dick from Gutenberg" undefined.
- Gives any future fetch/verification tooling a real target to implement against.


**Negative / accepted tradeoffs:**
- This ADR alone does not prove the existing repository's already-committed
 artifacts match the chosen source (see §11) — that reconciliation remains open.
- The candidate URL and encoding are unverified; this ADR cannot move to Accepted
 status until a real retrieval confirms them.
- Public-domain texts can be re-transcribed by their host over time even under an
 unchanged catalog number, which is why ADR-003's update policy (not this ADR)
 governs how future drift is handled.




