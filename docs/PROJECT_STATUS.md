# Project Status

## Version

Current version: `0.1.0`

## Completed Milestones

- TDD: architecture, roadmap, repository direction.
- A1 Tokenizer: byte-level BPE tokenizer, config, artifact, card, and unit tests.
- A2 Transformer: config-driven decoder-only transformer and architecture tests.
- A2.5 Research Infrastructure: packaging, common utilities, quality gates, hooks, and CI.
- A3 Forge: deterministic dataset pipeline, training engine, metrics, checkpointing, and resume.

## Current Milestone

- A3 Forge Training Pipeline. The implementation and bounded smoke run are complete.

## Next Milestone

- A4 Base Model: select the production corpus and compute budget, then run Forge to a first
  plateaued checkpoint.

## Known Technical Debt

- The current tokenizer corpus is marked as a placeholder and must be replaced before a
  serious v0.1 training run.
- Raw datasets are local/untracked; dataset versioning is still manifest-based rather than
  backed by DVC or object storage.
- The current tokenizer artifact is tracked for testability, but future artifact promotion
  needs a stricter release process.
- Some legacy A1/A2 modules still contain local validation helpers that can migrate to
  `helix.common` when the next milestone touches them.
- Forge currently tokenizes datasets eagerly in memory; A4-scale corpora will need streaming or
  memory-mapped token storage.
- Training is single-process and full precision. Distributed execution, gradient accumulation,
  and mixed precision remain future work.
- Checkpoints use the local filesystem and duplicate the newest numbered file as `latest.pt`;
  remote artifact storage and retention policies are not implemented.
