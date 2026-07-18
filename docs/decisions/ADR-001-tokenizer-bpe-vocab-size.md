# ADR-001: Byte-level BPE with a 16,000-token vocabulary

- Status: Accepted
- Date: 2026-07-18
- Milestone: A1 (Tokenizer)

## Context

Helix needs an owned tokenizer trained from scratch for its v0.1 model lineage. It must
round-trip arbitrary Unicode text, avoid unknown-token failures, preserve case, remain
small enough for a 10–125M parameter model, and be understandable and reproducible.

Vocabulary size directly trades model parameters against sequence length. A 4,000–8,000
token vocabulary reduces the embedding table but lengthens common text. At 32,000 tokens
or more, compression improves, but embeddings consume a disproportionate share of a tiny
model's parameter budget.

## Decision

Use Hugging Face `tokenizers` to train a byte-level BPE tokenizer with a target vocabulary
of 16,000 tokens. Begin from the complete 256-byte alphabet, apply NFC normalization
without lowercasing, and use the byte-level pre-tokenizer. Reserve `<pad>`, `<bos>`,
`<eos>`, `<unk>`, `<user>`, and `<assistant>` in that order.

Byte-level fallback represents every UTF-8 input without emitting `<unk>`. BPE merges
frequent byte sequences into useful words and subwords, giving materially shorter
sequences than character-level tokenization while retaining universal input coverage.
The 16,000-token vocabulary balances compression with an embedding table suitable for
Helix's planned small models.

## Alternatives considered

- **WordPiece:** Similar subword behavior, but uses a likelihood-oriented selection
  criterion. It offers no decisive v0.1 advantage and is less direct to reason about than
  frequency-based BPE.
- **Unigram:** Selects a probabilistic vocabulary by pruning a large candidate set. It can
  be valuable for larger multilingual production systems, but adds unnecessary conceptual
  and implementation complexity at this milestone.
- **Character-level:** Naturally avoids unknown characters and minimizes vocabulary size,
  but produces much longer sequences. That would consume the limited effective context
  window and compute budget of Helix's tiny model.
- **Smaller BPE vocabulary (4,000–8,000):** Saves embedding parameters at the cost of
  longer token sequences and weaker compression.
- **Larger BPE vocabulary (32,000+):** Improves compression, but makes embeddings too large
  a fraction of the v0.1 parameter budget.

## Consequences

- Any Unicode string can be represented from bytes; `<unk>` remains reserved only for
  downstream interface consistency.
- Case information is preserved, while canonically equivalent Unicode input is normalized
  to NFC.
- The tokenizer uses more embedding parameters than a 4,000–8,000-token design but offers
  better effective context through shorter sequences.
- The corpus must still be broad enough for useful merges; compression ratio is recorded
  on held-out text for every training run.
- Changing vocabulary size or reserved-token order is a compatibility-breaking decision
  and requires a superseding ADR and tokenizer retraining.
