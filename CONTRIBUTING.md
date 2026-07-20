# Contributing

## Development Setup

```bash
uv sync --frozen --extra dev
uv run pre-commit install
```

## Before Opening a PR

Run the same checks as CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Engineering Rules

- Keep tokenizer algorithms and transformer logic stable unless a milestone explicitly
  calls for a behavioral change.
- Prefer small, typed, well-tested changes over broad rewrites.
- Put shared infrastructure in `helix/common/` when it will be reused across milestones.
- Keep configuration in `config/`; avoid hardcoded repository paths in new code.
- Use structured logging through `helix.common.logging` for new scripts and research jobs.
- Use `set_global_seed(seed)` from `helix.common.seed` for reproducibility-sensitive code.

## Data And Artifacts

- Do not commit raw datasets, checkpoints, training runs, or large generated artifacts.
- Keep dataset manifests in `datasets/manifests/` with source, license, hashes, and split
  metadata.
- The current A1 tokenizer artifact remains tracked as the baseline required by tests.

## Documentation

- Update `docs/PROJECT_STATUS.md` when milestone status changes.
- Add or update ADRs in `docs/decisions/` for non-trivial architecture decisions.
- Document audits and milestone-specific engineering notes under `docs/<milestone>/`.
