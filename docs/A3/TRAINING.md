# A3 Forge Training Workflow

Forge is Helix's single-process, config-driven language-model training pipeline. It uses the
existing A1 tokenizer and A2 decoder-only transformer without changing their algorithms.

## Quick Validation

Run the bounded CPU smoke profile:

```bash
python train.py --config config/training/smoke.yaml
```

This profile trains the tiny smoke transformer for 20 steps against the checked-in sample
corpus, evaluates twice, and writes ignored checkpoints under `checkpoints/smoke/`.

## Forge Run

The default command loads `config/training/forge.yaml`:

```bash
python train.py
```

The training YAML owns all run parameters: dataset paths, validation strategy, context
length, DataLoader settings, optimizer and scheduler values, device selection, cadence,
checkpoint paths, seed, and logging format. Model dimensions remain in the referenced model
configuration.

Use either explicit `validation_paths` with `validation_fraction: 0.0`, or omit validation
paths and set a fraction in `(0.0, 1.0)`. Fractional splits shuffle blank-line-separated
documents with the configured seed before tokenization.

## Dataset Provenance

The checked-in tokenizer-training splits are documented in
`datasets/manifests/gutenberg-moby-dick.json`. M3a restructures that manifest under
ADR-003 with a stable dataset identity, an explicit raw dataset version, verified raw
retrieval metadata, raw SHA-256, UTF-8 encoding, and processed-output lineage back to the
verified raw bytes.

The current processed splits verify against Project Gutenberg eBook #15 as retrieved on
2026-07-20. Project Gutenberg eBook #2701 was also reachable during M3a verification, but
the existing deterministic preparation algorithm produced different train and held-out
outputs from those bytes. ADR-004 therefore remains a proposed future corpus decision and
is not adopted by the current manifest.

Raw corpus files remain gitignored; the manifest records the expected local path and the
checksum required to verify a local copy.

## Metrics

Forge emits structured log records through `helix.common.logging`. Training metric records
contain:

- `train_loss`
- `validation_loss`
- `learning_rate`
- `tokens_per_second`
- `steps_per_second`
- `eta_seconds`
- `global_step`

Set `logging.format` to `json` for machine-readable records or `text` for local debugging.

## Checkpoints And Resume

Every checkpoint contains:

- model state
- optimizer state
- scheduler state
- resolved training and model configuration
- tokenizer artifact identity
- training and validation dataset file identities
- global step
- data epoch and consumed-batch position
- Python, NumPy, PyTorch, and available CUDA/MPS RNG state

Numbered checkpoints and `latest.pt` are written atomically beneath the configured directory.
Checkpoint files are intentionally ignored by git.

To resume, set the training configuration to a checkpoint while keeping compatibility-affecting
configuration values the same as the checkpointed run:

```yaml
checkpoint:
  directory: checkpoints/forge
  resume_from: checkpoints/forge/latest.pt
```

Then run the same entry point:

```bash
python train.py
```

Before restoring model, optimizer, scheduler, or RNG state, Forge validates that the
checkpoint configuration is compatible with the current run. Compatibility includes model
dimensions and dropout, optimizer settings, dataset identity, tokenizer artifact identity,
batch-shaping settings, data-order settings, and scheduler-shaping loop settings. Cosmetic
and operational fields such as logging configuration, device selection, worker count, pin
memory, and checkpoint cadence are not part of compatibility validation.

If any validated field differs, resume fails before training continues. The error lists every
mismatched field with the previous checkpoint value and the current value.

Checkpoint format note: M2 adds tokenizer artifact SHA-256 and dataset file SHA-256 metadata
inside the saved checkpoint configuration. No repository legacy checkpoints existed when this
validation was introduced.

Only load checkpoints produced by a trusted Helix run. Complete optimizer and RNG restoration
requires Python object deserialization and is not intended for untrusted artifacts.

## Current Scope

Forge intentionally keeps the first implementation narrow: one process, eager in-memory
tokenization, full precision, and local filesystem checkpoints. The model, checkpoint, and
DataLoader ownership boundaries leave room for a later distributed launcher without embedding
rank assumptions in the training engine.
