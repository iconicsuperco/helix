# Project Status

## Version

Current version: `0.1.0`

## Completed Milestones

- TDD: architecture, roadmap, repository direction.
- A1 Tokenizer: byte-level BPE tokenizer, config, artifact, card, and unit tests.
- A2 Transformer: config-driven decoder-only transformer and architecture tests.
- A2.5 Research Infrastructure: packaging, common utilities, quality gates, hooks, and CI.
- A3 Forge: deterministic dataset pipeline, training engine, metrics, checkpointing, and resume.
- M1 Dependency Locking: uv lockfile, frozen installs, and aligned contributor/CI workflows.
- M2 Resume Compatibility: model, optimizer, tokenizer, dataset, batching, ordering, and
  scheduler compatibility checks before checkpoint restoration.
- M3 Dataset Provenance: stable dataset identity and verified Project Gutenberg eBook #15
  provenance for the Moby-Dick corpus.
- M4 Checkpoint Hardening: restricted `weights_only=True` checkpoint loading.
- M5 Config Consolidation: YAML loading routed through `helix.common.config`.
- M6 Documentation Accuracy: repository documentation synchronized with implemented behavior.
- M7 Integration Smoke Test: subprocess training, checkpoint creation, and resume coverage.
- M8 Minimal Inference: checkpoint-to-text greedy generation with tokenizer identity validation.
- R1 Evaluation Framework: manifest-verified in-corpus and out-of-corpus perplexity, a fixed
  prompt suite, automated repetition/distinct-n/memorization metrics, and manual-rubric reports.
- R2 Dataset Expansion: verified Gutenberg *The Time Machine* onboarding, shared cleaning,
  reviewable within- and cross-source deduplication, a versioned Moby-Dick/Time Machine composite
  corpus, and a separately versioned composite-corpus tokenizer artifact.
- R3 Model Scaling: complete; scaling outcome not yet justified. A 33,543,168-parameter model
  improved in-corpus perplexity but did not improve out-of-corpus perplexity over the
  16,889,856-parameter composite baseline.
- R4 Dataset Expansion: verified Gutenberg *Pride and Prejudice* onboarding, a substantially
  larger three-source composite, objective diversity and balance evidence, and an evidence-based
  decision to reuse the existing composite tokenizer unchanged.

## Current Repository State

- The A3 training pipeline, M8 local inference path, R1 evaluation framework, and R2 composite
  corpus pipeline are implemented.
- Evaluation uses the Moby-Dick held-out split in-corpus and the independently versioned Project
  Gutenberg Alice's Adventures in Wonderland corpus out-of-corpus.
- The current R4 training composite contains verified Moby-Dick, *The Time Machine*, and *Pride
  and Prejudice* sources; Alice remains outside the training corpus for out-of-corpus evaluation.
- R4 version `r4-v1-688ecae1740e` contains 29,927 training lines and 3,371 held-out lines, compared
  with R2's 19,729 and 2,236. Fragmentation analysis supported reusing the existing
  `helix-gutenberg-prose-bpe-v1` tokenizer artifact without retraining.
- R3 produced and evaluated distinct same-size composite-baseline and scaled checkpoint lineages;
  the evidence and conclusion are recorded in `docs/research/R3-scaling-comparison.md`.
- The verified suite has 90 tests: 86 unit tests and 4 integration tests.
- The default transformer has 16,889,856 trainable parameters.

## Next Milestone

- R4 is complete. The R4 composite corpus and reused tokenizer are ready to serve as a future R5
  scaling experiment's training input; R5 requires a separately approved task.

## Known Technical Debt

- Existing training configs still default to the Moby-Dick-only tokenizer for checkpoint
  compatibility; selecting the separate composite tokenizer for R4 training requires an explicit
  future config choice.
- Raw datasets are local/untracked; dataset versioning is still manifest-based rather than
  backed by DVC or object storage.
- The current tokenizer artifact is tracked for testability, but future artifact promotion
  needs a stricter release process.
- YAML parsing is centralized in `helix.common.config`; domain-specific validation remains in
  the tokenizer, model, and training configuration modules.
- Forge currently tokenizes datasets eagerly in memory; A4-scale corpora will need streaming or
  memory-mapped token storage.
- Training is single-process and full precision. Distributed execution, gradient accumulation,
  and mixed precision remain future work.
- Checkpoints use the local filesystem and duplicate the newest numbered file as `latest.pt`;
  remote artifact storage and retention policies are not implemented.
