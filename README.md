# Helix

Helix is an AI research codebase for building a small owned language-model lineage from
first principles: tokenizer, transformer, training, evaluation, inference, and eventually
platform integration.

Current focus: **A3 - Forge Training Pipeline**.

## What Exists

- A1 tokenizer: byte-level BPE wrapper, corpus preparation script, training script, and
  tokenizer artifact.
- A2 transformer: config-driven decoder-only PyTorch architecture with causal attention,
  learned positional embeddings, tied output weights, and unit tests.
- A2.5 foundation: Python packaging, quality tooling, CI, shared common utilities,
  contribution docs, and project status docs.
- A3 Forge: deterministic text datasets, PyTorch DataLoaders, training and validation,
  structured metrics, atomic checkpoints, and resumable runs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Quality Gates

```bash
ruff check .
ruff format --check .
mypy
pytest
```

The same gates run in GitHub Actions on every push and pull request.

## Repository Layout

```text
config/                 YAML configuration for tokenizer, model, and training domains
datasets/               Dataset manifests plus selected checked-in processed splits
docs/                   Architecture notes, ADRs, audits, and status
helix/common/           Shared infrastructure utilities
research/model/         A2 transformer architecture and config loader
research/tokenizer/     A1 tokenizer scripts, wrapper, and baseline artifact
research/training/      A3 data, engine, configuration, and checkpoint modules
tests/                  Unit tests and shared pytest setup
train.py                Config-driven Forge training entry point
```

Raw datasets, checkpoints, run outputs, and large generated model artifacts are ignored by
git. Manifests and small baseline artifacts document the current reproducible state.

## Useful Commands

```bash
python -m research.tokenizer.prepare_corpus
python -m research.tokenizer.train_tokenizer
python train.py --config config/training/smoke.yaml
python train.py
pytest tests/unit/tokenizer
pytest tests/unit/model
pytest tests/unit/training
```

## Documentation

- [Project status](docs/PROJECT_STATUS.md)
- [Forge training workflow](docs/A3/TRAINING.md)
- [A2.5 audit](docs/A2.5/AUDIT.md)
- [Architecture docs](docs/architecture)
- [Architecture decision records](docs/decisions)
