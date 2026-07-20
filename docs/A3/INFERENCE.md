# Minimal Inference

M8 loads a Forge checkpoint, reconstructs its transformer from saved model configuration,
validates the active tokenizer artifact against the checkpoint's SHA-256 identity, and generates
text with greedy autoregressive decoding.

Create a smoke checkpoint:

```bash
uv run python train.py --config config/training/smoke.yaml
```

Generate text from it:

```bash
uv run python infer.py \
  --checkpoint checkpoints/smoke/latest.pt \
  --prompt "Call me Ishmael." \
  --max-new-tokens 32 \
  --device cpu
```

`--device` accepts `auto`, `cpu`, `cuda`, or `mps`. The default is `cpu`. Generation is
deterministic greedy decoding: each step selects the highest-logit token and retains only the
latest model block when the growing sequence exceeds the configured context window.

Checkpoint loading reuses the M4 `weights_only=True` safe-loading path. Inference rejects a
checkpoint if its recorded tokenizer artifact SHA-256 does not match the tokenizer selected by
the active `HELIX_TOKENIZER_CONFIG` configuration.

Run the train-to-inference integration smoke test with:

```bash
uv run pytest tests/integration/test_infer_smoke.py
```

The current path is intentionally minimal: greedy decoding only, with no KV cache, sampling,
serving API, or platform integration.
