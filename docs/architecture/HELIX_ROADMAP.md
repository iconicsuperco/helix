# Helix — Roadmap v2 (Two-Track Model)

**Philosophy:** Own the stack, grow incrementally, learn deeply. Not competing with frontier labs. Optimizing for a real, working, owned system that gets better every version — not a fast demo, not a wrapper.

**Scope constraint (current):** Web application only. No desktop/mobile until explicitly requested.

**Core architectural rule:** Track B (platform) must NEVER hardcode assumptions about which model is running. All model access goes through a single `ModelProvider` interface. Swapping "wrapped external API" → "our own Helix model" must be a config change, not a rewrite.

---

## Current Repository Status

- A1 Tokenizer, A2 Tiny Transformer, and A3 Training Pipeline are implemented.
- The current default transformer has 16,889,856 parameters.
- M8 provides local greedy generation from Forge checkpoints; it is a research CLI, not the
  Track B `ModelProvider` integration.
- R1 Evaluation Framework is implemented. R2 Dataset Expansion and R3 Model Scaling remain
  unimplemented, as do all Track B milestones.

**Canonical sequencing update:** Infrastructure (M1-M8) -> R1 Evaluation Framework -> R2
Dataset Expansion -> R3 Model Scaling. This supersedes the older A4-to-A6 ordering retained
below for historical roadmap context.

---

## Track A — Core AI Research

Goal: a real, understood, owned model lineage — small, honest, and yours.

### A0. Research Foundations
- Pick one reference implementation to *learn from, not copy* (e.g. nanoGPT-style decoder-only transformer). Read it end to end before writing code.
- Decide compute budget up front: single consumer GPU / rented cloud GPU (A10/A100 spot) / Apple Silicon MPS. This budget determines every other decision below — set it before A1.
- Acceptance criteria: you can explain, without notes, what a tokenizer, embedding, attention head, and training loop each do.

### A1. Tokenizer (Implemented)
- Train a custom BPE tokenizer (small vocab, 8k–32k) on a modest, clearly-licensed corpus.
- Deliverable: reproducible tokenizer training script + vocab artifact + round-trip encode/decode tests.

### A2. Tiny Transformer (Helix v0.1, Implemented)
- Decoder-only transformer, small (10M–125M params depending on A0 budget).
- Built from scratch in PyTorch — this is the whole point, don't import a pretrained backbone here.
- Deliverable: model architecture code, config-driven (layers/heads/dim are parameters, not hardcoded).

### A3. Training Pipeline (Implemented)
- Data loading, checkpointing, structured training/validation metrics, and resumability.
- Deliverable: a training run that completes without babysitting and produces a checkpoint + metrics.

### A4. Base Model — First Real Checkpoint
- Train on your chosen corpus until loss plateaus at your compute budget.
- Acceptance criteria: model produces *coherent local grammar* — not necessarily correct facts. This is v0.1's actual bar.

### A5. Instruction Tuning (Helix v0.2 → v0.3)
- Small SFT dataset, format design (chat template), fine-tune the base checkpoint.
- Deliverable: model that follows a basic instruction format.

### A6. Evaluation Harness
- Build this *before* you need it for v0.4, not after: perplexity tracking + a small fixed set of prompts you manually score every version, so improvement is measurable, not vibes-based.

### A7. Iterate (Helix v0.4+)
- Bigger data, longer training, architecture tweaks (better positional encoding, longer context) — only after A6 exists, so you can prove each change actually helped.

**Reality check, said once so it doesn't need repeating:** A0–A4 is genuinely doable in weeks, not years, at small scale. That first checkpoint being "coherent grammar, not facts" is not a disappointing outcome — it's the correct milestone for v0.1. Don't quietly move the goalposts to "it should sound smart" partway through; that's the instinct that kills projects like this.

---

## Track B — Helix Platform

Goal: a real, usable web app, model-agnostic from day one.

### B1. Model Provider Abstraction (build this FIRST, before any UI)
- Single interface: `generate()`, `stream()`, `listModels()`.
- Initial implementation wraps an external API purely as a placeholder backend.
- Acceptance criteria: swapping the backend implementation requires touching exactly one file.

### B2. Web Frontend
- Next.js chat UI, calls only the Model Provider interface, never a specific vendor SDK directly.

### B3. Backend + Auth + Database
- Sessions, users, persistence layer (Postgres/SQLite depending on scale).

### B4. Chat Persistence + Memory
- Conversation history, basic long-term memory retrieval.

### B5. File Uploads

### B6. Tool-Calling System
- Generic tool-call interface the Model Provider can route through — doesn't matter which model is behind it.

### B7. First Real Swap-In
- Once Track A produces a usable checkpoint (post-A5), add it as a second `ModelProvider` implementation.
- Run it side-by-side with the wrapped model in the same UI. This is the moment the two tracks actually merge — treat it as a milestone, not an afterthought.

---

## Current Sequencing

The canonical R1 sequencing update above governs current work. Track B has not started in this
repository.
