# R3 Compute And Memory Estimate

## Environment

- Measured on 2026-07-22 on an Apple M4 MacBook Air with 10 CPU cores and 16 GiB of unified
  memory.
- PyTorch version: `2.13.0`.
- Training device selected by the existing `auto` path: MPS.
- CUDA availability: false.
- Available disk space before full training: 124 GiB.
- Existing constraints remain unchanged: FP32, single process, no mixed precision, no gradient
  accumulation, batch size 8, context length 128, and block size 256.

## Parameter Counts

The count uses `HelixTransformer.parameter_count`, which sums `numel()` over unique model
parameters. The equivalent architecture formula is:

`vocab_size * n_embd + block_size * n_embd + n_layer * (12 * n_embd^2 + 13 * n_embd) + 2 * n_embd`

The tied output head contributes no independent parameters.

| Configuration | Layers | Heads | Embedding | Exact parameters |
|---|---:|---:|---:|---:|
| Existing default / composite baseline | 6 | 6 | 384 | 16,889,856 |
| Scaled v1 | 8 | 8 | 512 | 33,543,168 |

Applying the method to the existing default reproduces ADR-002 exactly: 6,144,000 token
embeddings, 98,304 positional embeddings, 10,646,784 block parameters, and 768 final
layer-normalization parameters, totaling 16,889,856.

## Measured Throughput

Both new configurations completed a 12-step MPS smoke run over the composite corpus and wrote a
checkpoint. The steady interval at steps 6 through 10 produced:

| Configuration | Steps/second | Tokens/second | 1,000 optimization steps |
|---|---:|---:|---:|
| Composite baseline | 8.5799 | 8,785.83 | about 117 seconds |
| Scaled v1 | 4.5777 | 4,687.57 | about 218 seconds |

The direct extrapolation excludes ten scheduled 20-batch validation passes, four numbered
checkpoint writes, `latest.pt` refreshes, eager corpus tokenization, and process startup. Allowing
for those costs gives a conservative wall-clock estimate of 3 to 5 minutes for the composite
baseline and 5 to 8 minutes for scaled v1, or about 8 to 13 minutes combined.

## Memory And Storage

The persistent FP32 training-state estimate is 16 bytes per parameter: 4 bytes for model weights,
4 for gradients, and 8 for AdamW first and second moments. This excludes activations, logits,
framework workspaces, and allocator caches.

| Configuration | Persistent FP32 state | Measured peak RSS | Step-12 checkpoint |
|---|---:|---:|---:|
| Composite baseline | about 258 MiB | about 521 MiB | 193 MiB |
| Scaled v1 | about 512 MiB | about 755 MiB | 384 MiB |

MPS uses unified memory, so process RSS is an empirical bound from the smoke process rather than a
guarantee that every driver allocation is attributed identically. Even with substantial headroom
for activation and allocator variation, both measurements remain well below the available 16 GiB.
At four numbered checkpoints plus one `latest.pt` per lineage, expected retained checkpoint storage
is roughly 1.0 GiB for the baseline and 1.9 GiB for scaled v1, well below available disk space.

## Practicality Determination

Both full 1,000-step runs are practical in this execution environment. The measured MPS memory
footprints fit comfortably in 16 GiB, the expected combined training time is measured in minutes
rather than hours, and disk capacity is sufficient for both distinct checkpoint lineages. R3 may
therefore proceed with full training and evaluation; no reduced target is warranted by the observed
compute or memory limits.

## Observed Full Runs

The gate decision was confirmed by both completed runs:

| Configuration | Wall time | Peak RSS | Retained checkpoint storage |
|---|---:|---:|---:|
| Composite baseline | 126.75 seconds | 640.45 MiB | 967 MiB |
| Scaled v1 | 252.62 seconds | 749.42 MiB | 1.9 GiB |

The observed 6.3-minute combined training time was inside the conservative 8-to-13-minute estimate.
Both lineages completed without memory pressure, swaps, or checkpoint failures.
