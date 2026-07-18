# Helix — Milestone A1: Tokenizer — Engineering Specification

Track: A (Core AI Research) · Release: v0.1 · Status: Ready for implementation
Parent doc: HELIX_TDD.md (§2.5 Training Flow, §3 Repository Structure, §4 Stack)

---

## 1. Objectives

Produce a from-scratch-trained, custom BPE tokenizer for Helix — trained on our own corpus, owned by us, with no dependency on a pretrained vocabulary (e.g. GPT-2/GPT-4's tokenizer). This is the first artifact in the Helix model lineage and the input format every later Track A component depends on.

## 2. Success Criteria

- Tokenizer trains end-to-end on a defined corpus without manual intervention.
- Round-trip property holds: `decode(encode(text)) == text` for the full test corpus (byte-level BPE guarantees this — see §9).
- Vocabulary size, special tokens, and training config are all defined in a checked-in config file, not hardcoded.
- Tokenizer artifact + vocab + merges are reproducible from raw data + committed script (deterministic given the same corpus and seed).
- A fixed sample of test strings (including edge cases: emoji, non-English text, code snippets, repeated whitespace) encode/decode correctly.

## 3. Scope

- Training a byte-level BPE tokenizer using Hugging Face `tokenizers` (per TDD §4).
- Defining vocabulary size, special tokens, normalization rules.
- Producing a loadable tokenizer artifact + a thin Python wrapper module with `encode()`/`decode()`/`encode_batch()`.
- Unit tests covering round-trip correctness and edge cases.
- A short evaluation report: compression ratio (chars/token) on a held-out sample.

## 4. Out of Scope (explicitly deferred)

- Model architecture (A2).
- Any training-loop integration (A3) — this milestone produces the tokenizer only, used *by* A3 later.
- Dynamic/adaptive vocabulary, multilingual-specific tokenization tuning, or subword regularization (BPE-dropout) — these are future extensibility items (§16), not v0.1.
- A tokenizer-serving API/service — this is a library artifact consumed by other research code, not a running service.

## 5. Research References

- Sennrich et al., 2016 — "Neural Machine Translation of Rare Words with Subword Units" (the original BPE-for-NLP paper).
- Radford et al., 2019 (GPT-2) — byte-level BPE approach, which avoids an explicit unknown-token problem by operating over raw UTF-8 bytes rather than Unicode code points.
- Hugging Face `tokenizers` library documentation — the implementation we're using, not reinventing.

These are references to understand *why* the design below works, per TDD §1.3 ("understood, not copy-pasted").

## 6. Architecture

```
Raw corpus (datasets/raw/)
        │
        ▼
Corpus loader + basic cleaning (research/tokenizer/prepare_corpus.py)
        │
        ▼
BPE trainer (research/tokenizer/train_tokenizer.py)
   — byte-level pre-tokenizer
   — BPE model
   — special tokens injected
        │
        ▼
Tokenizer artifact (research/tokenizer/artifacts/tokenizer.json)
        │
        ▼
Thin wrapper module (research/tokenizer/tokenizer.py)
   exposes: encode(text) -> ids, decode(ids) -> text, encode_batch()
        │
        ▼
Consumed later by: A3 training pipeline, A5 instruction tuning, eventually LocalHelixProvider
```

## 7. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Algorithm | Byte-level BPE | See §9 below. |
| Library | Hugging Face `tokenizers` (Rust-backed) | Per TDD §4 — this is infrastructure tooling, not the research subject; use the well-tested library rather than reimplementing BPE merge logic by hand. |
| Vocabulary size | 16,000 | See §10 trade-off analysis. |
| Normalization | NFC Unicode normalization, no lowercasing | Preserves case information (matters even at small scale for names/acronyms); NFC avoids multiple representations of visually-identical characters. |
| Pre-tokenization | Byte-level (GPT-2 style regex split on whitespace/punctuation boundaries before BPE merge) | Standard, well-understood, avoids merges spanning unrelated word boundaries. |

## 8. Data Flow

1. Raw text files land in `datasets/raw/<corpus-name>/` with a manifest entry (source, license, date, hash) per TDD §3.
2. `prepare_corpus.py` concatenates/cleans (strip control characters except newline/tab, dedupe exact-duplicate lines) → writes to `datasets/processed/tokenizer-training/`.
3. `train_tokenizer.py` reads the processed corpus, trains the BPE model with the config in `config/model/tokenizer.yaml`, writes `tokenizer.json` + a `tokenizer_card.md` (vocab size, corpus stats, training date, corpus hash it was trained on).
4. `tokenizer.py` wrapper loads `tokenizer.json` and exposes the Python API other components import.

## 9. Algorithm Explanation — Why Byte-Level BPE

BPE starts from individual units (bytes, in our case) and iteratively merges the most frequent adjacent pair into a new token, repeating until the target vocabulary size is reached. This gives a compact vocabulary that captures common subwords/words as single tokens while still being able to represent any input, because rare/unseen sequences fall back to shorter, more granular merges (down to individual bytes in the worst case).

**Why byte-level specifically, instead of Unicode-character-level BPE:** operating over raw UTF-8 bytes means the base vocabulary is fixed at 256 possible byte values, and *any* valid Unicode string — any language, emoji, or malformed text — can always be represented, because it's built from bytes, not from a fixed set of "known characters." This eliminates the unknown-token problem structurally (§12) rather than requiring a fallback strategy for characters that weren't in the training corpus.

**Why BPE over alternatives:**
- *WordPiece* (BERT-style): very similar in practice, merge criterion differs slightly (likelihood-based vs frequency-based). BPE is simpler to understand and implement correctly, and is what GPT-family models use — better learning value given TDD §1.3.
- *Unigram* (SentencePiece-style): probabilistically drops tokens from a large candidate set rather than building up from merges. More complex to reason about; better suited to multilingual production systems at larger scale than v0.1 needs.
- *Character-level (no subword merging)*: no unknown-token problem either, but produces very long sequences for a given text, which directly hurts a tiny model's effective context window — bad trade-off at our compute budget (TDD §4).

BPE is the standard, well-documented, "understand it deeply" choice appropriate for a first tokenizer — matches §1.3's philosophy directly.

## 10. Vocabulary Size Trade-offs

| Vocab size | Pros | Cons |
|---|---|---|
| 4,000–8,000 | Smaller embedding table (matters more at 10–30M param scale); faster to train | Longer token sequences per text (worse compression), more merges needed for common words |
| 16,000 (chosen) | Reasonable compression for a small/medium English-heavy corpus; embedding table stays small relative to a 10–125M param model | Still small enough that a large embedding table doesn't dominate total parameter count |
| 32,000+ (GPT-2 scale) | Better compression, closer to production-grade tokenizers | Embedding table (vocab × hidden_dim) becomes a disproportionate fraction of total parameters at our model sizes — wasteful at v0.1's scale |

**Decision: 16,000.** Revisit as an ADR if the corpus grows substantially or becomes multilingual (§16).

## 11. Unicode Handling

- Input normalized to NFC before tokenization (composed form — e.g. é as one code point, not e + combining accent) for consistency.
- Because tokenization operates at the byte level (§9), any Unicode string is representable regardless of the training corpus's language mix — there is no separate "Unicode strategy" needed beyond normalization, which is precisely the point of the byte-level approach.

## 12. Unknown Token Strategy

Byte-level BPE has no true "unknown token" case: any input decomposes to bytes, and every byte value is in the base vocabulary by construction. An `<unk>` special token is still defined (§13) for interface consistency with downstream code that may expect one, but it should never actually be emitted during normal encoding — this is a correctness property to test for (§15), not a fallback path to rely on.

## 13. Special Tokens

| Token | Purpose |
|---|---|
| `<pad>` | Padding for batched training sequences (A3). |
| `<bos>` | Beginning of sequence marker. |
| `<eos>` | End of sequence marker. |
| `<unk>` | Reserved for interface consistency (§12) — should not appear in practice. |
| `<user>`, `<assistant>` | Reserved now, unused until A5 (instruction tuning chat template) — defining them at A1 avoids a vocabulary-breaking change later. |

## 14. File Formats

- `tokenizer.json` — Hugging Face `tokenizers` native serialization (vocab + merges + normalization + pre-tokenization rules in one file). Single source of truth for the trained tokenizer.
- `tokenizer_card.md` — human-readable metadata: vocab size, special tokens, corpus name + hash, training date, library version.
- `config/model/tokenizer.yaml` — training configuration (vocab size, special tokens list, corpus path) consumed by `train_tokenizer.py`.

## 15. Training Pipeline

1. `prepare_corpus.py` — cleans raw corpus, writes processed text + a manifest entry (per TDD §3 datasets folder).
2. `train_tokenizer.py` — loads `config/model/tokenizer.yaml`, trains via `tokenizers` `BpeTrainer`, writes `tokenizer.json` + `tokenizer_card.md` to `research/tokenizer/artifacts/`.
3. Both scripts are deterministic given the same corpus + config (fixed seed where the library exposes one; document if any non-determinism exists).

## 16. Evaluation Strategy

- **Compression ratio:** average characters-per-token on a held-out sample not used in training — this is the primary tokenizer-quality metric, tracked in `tokenizer_card.md` and compared across any future retraining.
- **Round-trip test:** `decode(encode(text)) == text` across a fixed test set (§17).
- No perplexity/model-quality evaluation here — that belongs to A6, applied to the *model*, not the tokenizer.

## 17. Test Plan

Unit tests in `tests/unit/tokenizer/`:
1. Round-trip on plain English sentences.
2. Round-trip on text with emoji, accented characters, and non-English (e.g. a Hindi or Japanese sample) — proves byte-level handling per §11.
3. Round-trip on code snippets (mixed indentation, special characters).
4. Round-trip on edge cases: empty string, very long repeated-character string, string of only whitespace.
5. Confirm `<unk>` is never emitted for any test-set input (§12).
6. Confirm special tokens (§13) encode to their reserved, fixed IDs and are excluded correctly when decoding "clean" text.
7. Confirm `encode_batch()` produces the same results as calling `encode()` individually per item.

## 18. Expected Outputs

- `research/tokenizer/artifacts/tokenizer.json`
- `research/tokenizer/artifacts/tokenizer_card.md`
- `research/tokenizer/prepare_corpus.py`
- `research/tokenizer/train_tokenizer.py`
- `research/tokenizer/tokenizer.py` (wrapper module)
- `config/model/tokenizer.yaml`
- `tests/unit/tokenizer/test_tokenizer.py`
- `docs/decisions/ADR-001-tokenizer-bpe-vocab-size.md` (captures §7/§9/§10 as a formal ADR per TDD §9.3)

## 19. Repository Locations

All paths above are relative to repo root, matching HELIX_TDD.md §3 exactly — no new top-level folders introduced by this milestone.

## 20. Risks

- **Corpus too small/narrow for a meaningful vocabulary:** mitigate by checking the compression ratio (§16) against a sane baseline (roughly 3.5–4.5 chars/token is typical for English BPE at this vocab size) before proceeding to A2 — if far off, the corpus or vocab size needs revisiting, not the model.
- **Corpus licensing ambiguity:** mitigate via the manifest requirement (TDD §10) — no raw data enters `datasets/raw/` without a documented source/license.
- **Non-determinism silently creeping into training:** mitigate by pinning `tokenizers` library version and documenting any observed non-determinism in `tokenizer_card.md` rather than assuming reproducibility.

## 21. Future Extensibility

- Vocabulary size is revisitable via ADR as the corpus grows (§10).
- Multilingual corpus expansion is possible without an architecture change, since byte-level BPE already handles arbitrary Unicode (§9, §11) — only the *training corpus* changes, not the approach.
- BPE-dropout (subword regularization) could be added later as a training-time augmentation without changing the tokenizer artifact format.
- The `<user>`/`<assistant>` tokens reserved now (§13) mean A5's chat template work won't require a vocabulary-breaking retrain.
