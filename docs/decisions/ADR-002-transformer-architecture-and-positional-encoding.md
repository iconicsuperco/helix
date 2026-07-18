# ADR-002: Decoder-only transformer architecture and positional encoding

- Status: Accepted
- Date: 2026-07-18
- Milestone: A2 (Tiny Transformer)

## Context

Helix v0.1 needs a small, understandable language-model architecture that consumes A1's
token IDs and can later be trained by A3. The architecture must enforce causal prediction,
fit the project's modest compute budget, and remain stable enough to train without adding
premature positional-encoding or inference optimizations.

The corrected hand calculation for the selected configuration is 16,889,856 parameters:
6,144,000 token-embedding parameters, 98,304 positional-embedding parameters, 10,646,784
parameters across six transformer blocks, and 768 final layer-normalization parameters.
The tied output head adds no independent parameters.

## Decision

Use a decoder-only, GPT-2-style pre-norm transformer with six blocks, six attention heads,
a 384-dimensional embedding, a 256-token context, and dropout of 0.1. Each block applies
layer normalization before causal self-attention and before its four-times-expanded GELU
MLP, with residual additions around both sublayers.

Use learned absolute positional embeddings. Compute attention through PyTorch's scaled
dot-product attention with its causal mode enabled. Tie the bias-free output head weight
to the input token-embedding weight. Initialize embedding and linear weights from a normal
distribution with standard deviation 0.02, then scale attention and MLP residual-output
projection initialization by `1 / sqrt(2 * n_layer)`.

The model vocabulary size is loaded from A1's tokenizer configuration rather than copied
into the transformer configuration.

## Alternatives considered

- **Post-norm transformer:** Matches the original Transformer ordering, but is less stable
  at depth and generally depends on more delicate learning-rate warmup. Pre-norm is the
  simpler and safer baseline for A3's limited tuning budget.
- **Rotary positional embeddings (RoPE):** Encode relative position by rotating query and
  key vectors and usually extrapolate beyond trained sequence lengths better. They add
  conceptual and implementation complexity that v0.1's fixed 256-token context does not
  need; a later eval-backed milestone can compare them.
- **ALiBi:** Adds head-specific distance biases directly to attention and can extrapolate
  to longer contexts without a position table. It diverges from the deliberately familiar
  GPT-2 baseline and adds attention-specific behavior before Helix has an eval harness.
- **Untied output weights:** Allow the input and output representations to specialize, but
  duplicate the vocabulary-sized matrix. That would add 6,144,000 parameters to this small
  model without a demonstrated v0.1 benefit.
- **Smaller or GPT-2-small sizing:** A four-layer, 256-dimensional model is cheaper but has
  less capacity; a 12-layer, 768-dimensional model is excessive for the placeholder corpus.
  The selected 16.9M-parameter model stays near the low end of Helix's intended range.

## Consequences

- Every output position depends only on its current and earlier input positions.
- The architecture is straightforward to inspect and sanity-check before A3 adds training
  infrastructure.
- Weight tying materially reduces parameters and requires the embedding dimension to match
  the output-head input dimension.
- Learned positions limit inference to the configured context window and do not promise
  length extrapolation; changing the positional scheme requires a new experiment and ADR.
- The bounded unit-test overfit proves gradient flow only and is not evidence of model
  quality, corpus learning, or generalization.
