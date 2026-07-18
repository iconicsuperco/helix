# Project Status

## Version

Current version: `0.1.0`

## Completed Milestones

- TDD: architecture, roadmap, repository direction.
- A1 Tokenizer: byte-level BPE tokenizer, config, artifact, card, and unit tests.
- A2 Transformer: config-driven decoder-only transformer and architecture tests.

## Current Milestone

- A2.5 Research Infrastructure & Engineering Foundation.

## Next Milestone

- A3 Training Pipeline: data loading, checkpointing, logging, metrics, sample generation,
  resumability, and first unattended training run.

## Known Technical Debt

- The current tokenizer corpus is marked as a placeholder and must be replaced before a
  serious v0.1 training run.
- Raw datasets are local/untracked; dataset versioning is still manifest-based rather than
  backed by DVC or object storage.
- The current tokenizer artifact is tracked for testability, but future artifact promotion
  needs a stricter release process.
- Some legacy A1/A2 modules still contain local validation helpers that can migrate to
  `helix.common` when the next milestone touches them.
- No checkpoint registry exists yet; checkpoints are intentionally ignored until A3 defines
  the storage and metadata contract.
