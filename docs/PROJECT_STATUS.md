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

## Current Repository State

- The A3 training pipeline, M8 local inference path, and R1 evaluation framework are implemented.
- Evaluation uses the Moby-Dick held-out split in-corpus and the independently versioned Project
  Gutenberg Alice's Adventures in Wonderland corpus out-of-corpus.
- The verified suite has 66 tests: 63 unit tests and 3 subprocess integration tests.
- The default transformer has 16,889,856 trainable parameters.

## Next Milestone

- R2 Dataset Expansion, followed by R3 Model Scaling. This canonical sequence supersedes the
  older A4-to-A6 ordering.

## Known Technical Debt

- The current tokenizer corpus is marked as a placeholder and must be replaced before a
  serious v0.1 training run.
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
