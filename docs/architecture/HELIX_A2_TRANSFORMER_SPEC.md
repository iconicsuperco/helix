# Helix — Milestone A2: Tiny Transformer — Engineering Specification

Track: A (Core AI Research) · Release: v0.1 · Status: Ready for implementation
Parent docs: HELIX_TDD.md (§2.5 Training Flow, §3 Repository Structure, §4 Stack) · HELIX_A1_TOKENIZER_SPEC.md (upstream dependency)
Depends on: A1 (Tokenizer) — merged, PR #1.

---

## 1. Objectives

Implement, from scratch in PyTorch, a decoder-only transformer language model — the second artifact in the Helix model lineage. This milestone produces the **architecture only**: a model class that takes token IDs and returns logits. No training loop, no optimizer, no data pipeline — that is A3.

## 2. Success Criteria

- A config-driven `HelixTransformer` model class exists: layer count, head count, embedding dimension, context length, and dropout are all parameters, not hardcoded.
- A forward pass runs end-to-end on randomly initialized weights: `(batch, sequence)` token IDs in → `(batch, sequence, vocab_size)` logits out, with correct shapes for every configured size.
- Causal masking is verifiably correct: changing a future token never changes an earlier position's output.
- Parameter count is computed and reported, and matches a hand-calculated expectation for the default config within a documented tolerance.
- A "sanity overfit" test passes: the model can drive the loss on a single tiny fixed batch down to near-zero within a bounded number of gradient steps — this is the standard, minimal proof that gradients flow correctly through the whole architecture, independent of any real training run (which is A3's job).
- The model consumes A1's tokenizer vocabulary (16,000 + reserved tokens) without modification to either component.

## 3. Scope

- Token embedding + learned positional embedding.
- N stacked transformer blocks: pre-norm, causal multi-head self-attention, MLP with GELU, residual connections.
- Final layer norm + linear output head, with output head weights **tied** to the input token embedding.
- Config-driven sizing (`config/model/transformer.yaml`).
- Parameter initialization scheme (documented, not arbitrary).
- Unit tests: shape correctness, causal-mask correctness, parameter count, determinism, single-batch overfit sanity check.

## 4. Out of Scope (explicitly deferred)

- Training loop, optimizer, learning-rate schedule, data loader, checkpointing (A3).
- Evaluation harness / perplexity tracking on real data (A6) — this milestone's "overfit sanity check" is an architecture-correctness test, not a model-quality evaluation.
- Rotary or relative positional encodings, KV-caching for fast inference, mixed precision, gradient checkpointing — all future extensibility (§16), not needed for v0.1's tiny scale.
- Any integration with the platform / `LocalHelixProvider` (that's B7, after A3 and A5).

## 5. Research References

- Vaswani et al., 2017 — "Attention Is All You Need" (the original transformer; we use the decoder-only half).
- Radford et al., 2019 (GPT-2) — pre-norm decoder-only architecture, learned positional embeddings, weight tying between input embedding and output head. This is the architecture family we're implementing, for the same "understood, standard, well-documented" reasons the tokenizer chose byte-level BPE (A1 §5).
- Karpathy, nanoGPT — a widely-used minimal reference implementation of exactly this architecture. Read to *understand*, not to copy verbatim, per TDD §1.3.

## 6. Architecture

```
Token IDs (batch, seq_len)
        │
        ▼
Token Embedding (vocab_size × n_embd)  +  Learned Positional Embedding (block_size × n_embd)
        │
        ▼
┌─────────────────────────── Transformer Block × n_layer ───────────────────────────┐
│  LayerNorm → Causal Multi-Head Self-Attention → residual add                       │
│  LayerNorm → MLP (Linear → GELU → Linear) → residual add                           │
└──────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
Final LayerNorm
        │
        ▼
Linear Output Head (tied to Token Embedding weights)
        │
        ▼
Logits (batch, seq_len, vocab_size)
```

## 7. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture family | Decoder-only, pre-norm (GPT-2 style) | Standard, best-documented, matches the reference material in §5; post-norm (original transformer) is harder to train stably at small scale without careful warmup tuning. |
| Positional encoding | Learned absolute positional embeddings | Simplest to implement and understand correctly; matches "build simple before complex" (TDD §9.5). Rotary (RoPE) is deferred to §16 — better length-extrapolation, but adds conceptual complexity not needed for v0.1's short, fixed context length. |
| Attention computation | `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` | This is the textbook attention formula (softmax(QKᵀ/√d)V) with a fused, correct, well-documented PyTorch implementation — using it is the same category of decision as A1 using the `tokenizers` library: standard infrastructure, not the research subject itself. The manual formula is documented in §11 so it's understood, not treated as a black box. |
| Weight tying | Output head shares weights with the token embedding | Standard practice (GPT-2, most small LMs); reduces parameter count meaningfully at small `n_embd`/`vocab_size` ratios, since the embedding table otherwise dominates a tiny model's parameter budget (as already noted in A1 §10). |
| Normalization placement | Pre-norm (LayerNorm before attention/MLP, not after) | More stable training at small scale without needing a learning-rate warmup as delicate as post-norm requires — directly relevant since A3 will have a modest compute budget for tuning. |
| Default model size | `n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.1` | See §12 sizing rationale. |
| Initialization | Normal(mean=0, std=0.02) for embeddings/linear layers; residual-path output projections scaled by `1/√(2·n_layer)` | Matches GPT-2's published initialization scheme, which exists specifically to keep activation variance stable as depth increases — a documented, non-arbitrary choice. |

## 8. Data Flow

```
A1 tokenizer.encode() → token ID sequence (list[int])
        │
        ▼
Batching/padding to block_size (A3's responsibility to build; A2 only requires the model accept
a (batch, seq_len) LongTensor with seq_len ≤ block_size)
        │
        ▼
HelixTransformer.forward(idx) → logits (batch, seq_len, vocab_size)
```

A2 does not implement the batching/padding step — it only defines the contract (`forward` accepts a `(batch, seq_len)` LongTensor with values in `[0, vocab_size)`) that A3's data pipeline must satisfy.

## 9. Algorithm Explanation

A decoder-only transformer predicts each token from only the tokens before it (causal/left-to-right), which is what makes it usable for autoregressive text generation. Each transformer block refines every position's representation by:
1. **Self-attention** — every position looks at all *earlier* positions (causal mask prevents looking ahead) and computes a weighted combination of their representations, where the weights come from a learned similarity (query·key) between positions.
2. **MLP** — an independent, per-position nonlinear transformation (two linear layers with a GELU nonlinearity between them), which is where most of the model's "knowledge capacity" lives.

Stacking many such blocks lets the model build up increasingly abstract representations — this is the same core idea A1's ADR-001 alludes to when it separates the tokenizer (representation of text as tokens) from the model (representation of meaning from tokens).

## 10. Positional Encoding — Why Learned Absolute, Not RoPE

Learned absolute positional embeddings add a per-position learned vector to each token's embedding before the first transformer block, giving the model a way to distinguish "token at position 3" from "the same token at position 30." This is the simplest positional scheme to implement and verify correctly — it is literally one more embedding table, no additional math beyond addition.

RoPE (rotary positional embeddings) rotates query/key vectors as a function of position instead of adding a learned vector, which gives better generalization to sequence lengths not seen during training and is the current default in most production-scale models. It's deferred to §16/A4: it is the *right long-term choice*, but adds a layer of "why does rotating vectors like this preserve relative-position information" that is not necessary to prove out the core architecture correctness this milestone is actually testing. Once A6's eval harness exists, swapping in RoPE and measuring the difference is exactly the kind of "measure, don't assume" comparison TDD §9.4 asks for.

## 11. Attention — The Explicit Formula (for understanding, even though a library call implements it)

For each attention head: given query, key, value projections `Q`, `K`, `V` of a sequence, the output is:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_head + causal_mask ) · V
```

where `causal_mask` sets all "future" positions (`j > i` for query position `i`) to `-∞` before the softmax, so they receive zero attention weight. `d_head = n_embd / n_head` is the per-head dimension, and the `√d_head` scaling keeps the dot-product magnitudes (and therefore softmax gradients) stable regardless of head size. Multi-head attention runs this independently across `n_head` heads with separate learned projections, then concatenates and linearly projects the result back to `n_embd`.

`F.scaled_dot_product_attention(q, k, v, is_causal=True)` computes exactly this formula — the library call is a fused, numerically-stable implementation of the equation above, not a different algorithm.

## 12. Model Sizing — Default Config Trade-offs

| Config | Approx. params (16k vocab) | Notes |
|---|---|---|
| `n_layer=4, n_head=4, n_embd=256, block_size=128` | ~7M | Faster to sanity-check on CPU; may underfit even the small placeholder corpus. |
| `n_layer=6, n_head=6, n_embd=384, block_size=256` (chosen default) | ~13M | Matches TDD's "10–125M" range at the small end; short context length (256 tokens) is appropriate given the placeholder corpus's short lines (A1's held-out set had under 2,000 unique lines) — no reason to pay for a longer context the current corpus can't exercise. |
| `n_layer=12, n_head=12, n_embd=768, block_size=1024` (GPT-2-small scale) | ~124M | Reserved for a later version once a real, larger training corpus exists (A2/A4 boundary) — training this on the current placeholder corpus would simply memorize it, which teaches nothing about the architecture. |

**Decision: the 13M-parameter default.** This is a config value, not a hardcoded constant — resizing for a future milestone is a one-line config change, not a code change (§14 test plan verifies this).

## 13. Special-Token / Vocabulary Integration

The model's embedding table size is exactly A1's configured `vocab_size` (16,000) — no separate handling for special tokens is needed at the architecture level, since they're just additional vocabulary entries at fixed IDs 0–5 (A1 §13). This is a direct, deliberate consequence of A1 reserving `<user>`/`<assistant>` tokens ahead of time: A2 never needs to know special tokens exist as a distinct concept.

## 14. Test Plan

Unit tests in `tests/unit/model/test_transformer.py`:
1. **Shape correctness:** for at least two different configs (the tiny 7M and the default 13M from §12), a forward pass on a random `(batch=2, seq_len=32)` input produces `(2, 32, vocab_size)` logits.
2. **Causal-mask correctness:** run a forward pass, then change only the *last* token in the input and re-run; assert that logits at all positions *before* the last are bit-for-bit identical. This directly proves no future information leaks backward.
3. **Parameter count:** compute total parameter count for the default config and assert it falls within ±5% of the hand-calculated estimate in §12 (accounting for weight tying).
4. **Determinism:** with a fixed random seed, two freshly-constructed models with identical configs produce bit-identical initial weights and identical logits on the same input.
5. **Single-batch overfit sanity check:** construct one small fixed batch (e.g. 4 sequences of 16 tokens from the A1 tokenizer), run a plain training loop (Adam, cross-entropy loss against next-token targets) for a bounded number of steps (e.g. 200) directly in the test, and assert the loss drops below a fixed low threshold. This is a test of A2's architecture correctness (gradients flow, loss decreases), explicitly *not* a claim about real-corpus training quality (that's A3/A6).
6. **Config-driven resizing:** load two different `n_layer`/`n_embd` configs and assert the resulting models have different parameter counts consistent with the config change — proves no dimension is hardcoded.

## 15. Expected Outputs

- `research/model/transformer.py` — `HelixTransformer` model class, attention block, MLP block.
- `research/model/config.py` — typed config loader for `config/model/transformer.yaml`.
- `config/model/transformer.yaml` — `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `vocab_size` (cross-referenced from A1's tokenizer config, not duplicated by value — see §17 risk).
- `tests/unit/model/test_transformer.py` — the six tests in §14.
- `docs/decisions/ADR-002-transformer-architecture-and-positional-encoding.md` — documents §7/§10's decisions (architecture family, positional encoding choice, weight tying, default sizing).

## 16. Future Extensibility

- RoPE or ALiBi positional encoding can replace the learned-embedding module without touching the attention/MLP blocks — the positional scheme is isolated to embedding + attention input, not threaded through the whole model.
- KV-caching for fast autoregressive inference is a pure addition (an inference-time optimization) and doesn't change the trained weights or architecture — deferred until A3/inference work needs it.
- Longer `block_size` for larger future corpora is a config change; the architecture makes no assumption about context length beyond the positional embedding table's size.
- Mixed precision / gradient checkpointing are A3 training-pipeline concerns, not architecture concerns — the model class defined here is precision-agnostic.

## 17. Risks

- **Config duplication between A1 and A2 (`vocab_size` needed by both):** mitigate by having `config/model/transformer.yaml` reference A1's `config/model/tokenizer.yaml` vocab size at load time (read it, don't hand-copy the number) — prevents the two configs silently drifting apart if the tokenizer is ever retrained with a different vocab size.
- **Overfit sanity check (§14.5) accidentally becoming a real training claim:** mitigate by naming it explicitly as an architecture-correctness test in code comments and the ADR, not as a v0.1 model-quality result — this distinction matters so nobody later cites "the model overfit 4 sentences" as if it were A6 evaluation evidence.
- **Silent shape mismatches at large batch/sequence sizes not covered by unit tests:** mitigate by testing at least one shape combination other than the trivial (batch=1, seq_len=1) case — already covered in §14.1's (2, 32) test.

## 18. Repository Locations

All paths match HELIX_TDD.md §3 exactly — `research/model/` and `tests/unit/model/` are new subfolders of already-defined top-level directories, no new top-level structure introduced.
