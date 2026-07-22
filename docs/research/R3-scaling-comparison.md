# R3 Scaling Comparison

## Experiment

R3 compares two models trained for 1,000 optimizer steps on the same composite corpus with the same
tokenizer, context length, batch size, data ordering, optimizer, schedule, evaluation cadence, and
checkpoint cadence. The only model-capacity changes are layers, heads, and embedding width.

| Lineage | Model config | Parameters | Final checkpoint SHA-256 |
|---|---|---:|---|
| Composite baseline | `config/model/transformer-composite-baseline-v1.yaml` | 16,889,856 | `4aa9808c5d8f26514f3ce516c04171b844da045b73b908f53cd24668674e410e` |
| Scaled v1 | `config/model/transformer-scaled-v1.yaml` | 33,543,168 | `8f12fad69c2439759af4936eed72aa8a0c5991847add0b3d3a8d869fa905931b` |

Both checkpoints record composite tokenizer SHA-256
`1af9f66318bfd8a547bc20a4adda652b272309ba01699be85e62d7277712bc8f`, composite training SHA-256
`75384e3c51779dfa32b72e48bddb806d2313ff1c51d581b743b3f9ff254d4226`, and composite validation
SHA-256 `9c4c01ae1950742d219381a00ba0fa9226965d77eb3001c6a5ff8a4b75679e66`.

## Training Outcomes

| Metric | Composite baseline | Scaled v1 |
|---|---:|---:|
| Completed steps | 1,000 | 1,000 |
| Initial loss | 9.769117 | 9.823781 |
| Final training loss | 5.493929 | 5.457029 |
| Final validation loss | 5.809814 | 5.788510 |
| Wall time | 126.75 seconds | 252.62 seconds |
| Peak RSS | 640.45 MiB | 749.42 MiB |

## R1 Perplexity

The existing unmodified `evaluation/config.yaml` and evaluation code were used for both models. The
active tokenizer was selected with `HELIX_TOKENIZER_CONFIG` to match checkpoint identity. The R1
in-corpus target is the Moby-Dick held-out split; the out-of-corpus target is Alice's Adventures in
Wonderland.

| Dataset | Composite baseline | Scaled v1 | Scaled change |
|---|---:|---:|---:|
| Moby-Dick held-out | 448.148275 | 438.887119 | -9.261156 (-2.066538%) |
| Alice out-of-corpus | 480.634811 | 480.939539 | +0.304728 (+0.063401%) |

Lower perplexity is better. The scaled model improved the in-corpus result but did not improve the
out-of-corpus result.

## Automated Generation Metrics

The table below reports averages across the fixed prompt suite. Lower repetition and higher
distinct-2 are better.

| Prompt group | Metric | Composite baseline | Scaled v1 |
|---|---|---:|---:|
| All 12 prompts | Repetition-4 | 0.362609 | 0.319536 |
| All 12 prompts | Distinct-2 | 0.403789 | 0.480130 |
| In-distribution prompts | Repetition-4 | 0.474950 | 0.385409 |
| In-distribution prompts | Distinct-2 | 0.342258 | 0.413037 |
| Out-of-distribution prompts | Repetition-4 | 0.250267 | 0.253663 |
| Out-of-distribution prompts | Distinct-2 | 0.465321 | 0.547222 |

Per-prompt results:

| Prompt | Baseline repetition-4 | Scaled repetition-4 | Baseline distinct-2 | Scaled distinct-2 |
|---|---:|---:|---:|---:|
| `moby-opening-reflection` | 0.473684 | 0.266667 | 0.285714 | 0.529412 |
| `whaling-voyage` | 0.352941 | 0.500000 | 0.421053 | 0.285714 |
| `captain-at-sea` | 0.266667 | 0.071429 | 0.470588 | 0.625000 |
| `harpooner-description` | 0.833333 | 0.416667 | 0.142857 | 0.357143 |
| `white-whale-rumor` | 0.153846 | 0.307692 | 0.533333 | 0.466667 |
| `nineteenth-century-storm` | 0.769231 | 0.750000 | 0.200000 | 0.214286 |
| `garden-story` | 0.312500 | 0.142857 | 0.388889 | 0.562500 |
| `tea-instructions` | 0.230769 | 0.071429 | 0.400000 | 0.687500 |
| `leaves-explanation` | 0.333333 | 0.307692 | 0.411765 | 0.533333 |
| `train-arithmetic` | 0.000000 | 0.285714 | 0.750000 | 0.562500 |
| `friendly-letter` | 0.500000 | 0.571429 | 0.285714 | 0.375000 |
| `market-scene` | 0.125000 | 0.142857 | 0.555556 | 0.562500 |

Both reports found no exact memorization overlap meeting the configured 12-token threshold. The
scaled model improved aggregate repetition and distinct-2, but individual prompts remain mixed and
the completions from both models remain strongly biased toward repetitive whaling language.

## Qualitative Rubric

Manual human review is pending. The generated reports retain blank grammaticality, local-coherence,
and repetition/degeneracy score fields as required by the R1 rubric. No automated or inferred scores
were inserted.

## Conclusion

Scaling is not yet justified by the available R3 evidence. Doubling model parameters produced a
2.07% in-corpus perplexity improvement and better aggregate lexical-diversity metrics, but the
decisive out-of-corpus Alice perplexity worsened by 0.06%. The generated samples also remain visibly
domain-collapsed and repetitive. R3 therefore does not establish a genuine generalization gain from
scale under this corpus and 1,000-step training budget.

The correct outcome is inconclusive for scaling efficacy and negative for the specific R3 success
criterion. Further scaling should not proceed from these results alone.

## Reproduction

```bash
HELIX_TOKENIZER_CONFIG=research/data/tokenizers/helix-gutenberg-prose-bpe-v1/tokenizer.yaml \
  uv run python train.py --config config/training/forge-composite-baseline-v1.yaml

HELIX_TOKENIZER_CONFIG=research/data/tokenizers/helix-gutenberg-prose-bpe-v1/tokenizer.yaml \
  uv run python train.py --config config/training/forge-scaled-v1.yaml

HELIX_TOKENIZER_CONFIG=research/data/tokenizers/helix-gutenberg-prose-bpe-v1/tokenizer.yaml \
  uv run python evaluate.py \
    --checkpoint checkpoints/forge-composite-baseline-v1/latest.pt \
    --config evaluation/config.yaml

HELIX_TOKENIZER_CONFIG=research/data/tokenizers/helix-gutenberg-prose-bpe-v1/tokenizer.yaml \
  uv run python evaluate.py \
    --checkpoint checkpoints/forge-scaled-v1/latest.pt \
    --config evaluation/config.yaml
```

Committed evaluation reports:

- `evaluation/reports/4aa9808c5d8f26514f3ce516c04171b844da045b73b908f53cd24668674e410e/evaluation-20260722T053446376146Z.md`
- `evaluation/reports/8f12fad69c2439759af4936eed72aa8a0c5991847add0b3d3a8d869fa905931b/evaluation-20260722T053503932017Z.md`
