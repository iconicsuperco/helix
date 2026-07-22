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

## Current Repository State

- The A3 training pipeline, M8 local inference path, R1 evaluation framework, and R2 composite
  corpus pipeline are implemented.
- Evaluation uses the Moby-Dick held-out split in-corpus and the independently versioned Project
  Gutenberg Alice's Adventures in Wonderland corpus out-of-corpus.
- The R2 training composite contains the verified Moby-Dick and *The Time Machine* sources; Alice
  remains outside the training corpus for out-of-corpus evaluation.
- R3 produced and evaluated distinct same-size composite-baseline and scaled checkpoint lineages;
  the evidence and conclusion are recorded in `docs/research/R3-scaling-comparison.md`.
- The verified suite has 82 tests: 78 unit tests and 4 integration tests.
- The default transformer has 16,889,856 trainable parameters.

## Next Milestone

- R3 is complete. Any later milestone requires a separately approved task.

## Known Technical Debt

- Existing training configs still default to the Moby-Dick-only tokenizer for checkpoint
  compatibility; adopting the separate R2 composite tokenizer requires an explicit future config
  selection.
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
