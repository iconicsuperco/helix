# Helix Tokenizer Card

## Model

- Algorithm: byte-level BPE
- Vocabulary size: 16000
- Normalization: NFC, no lowercasing
- Pre-tokenizer: ByteLevel
- `tokenizers` version: 0.23.1
- Training date (UTC): 2026-07-22

## Special tokens

| Token | Reserved ID |
|---|---:|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<unk>` | 3 |
| `<user>` | 4 |
| `<assistant>` | 5 |

## Corpus

- Name: helix-gutenberg-prose
- Source: Composite of approved Project Gutenberg source datasets
- License: Public domain in the USA; distributed under the Project Gutenberg License. Verify status in the deployment jurisdiction.
- Training content SHA-256: `75384e3c51779dfa32b72e48bddb806d2313ff1c51d581b743b3f9ff254d4226`
- Held-out content SHA-256: `9c4c01ae1950742d219381a00ba0fa9226965d77eb3001c6a5ff8a4b75679e66`
- Corpus status: This corpus is approved for the v0.1 model training run.

The held-out split is assigned deterministically from each unique cleaned line's SHA-256
hash and is excluded from BPE training.

## Evaluation

- Held-out compression ratio: **3.900578 characters/token**

The ratio is total Unicode characters divided by total emitted tokens over the held-out
sample. Empty samples would report `0.0`; this training run used a non-empty sample.

## Reproducibility

Corpus cleaning, exact-line deduplication, and held-out assignment are deterministic for
the same raw files and config. Hugging Face `tokenizers` does not expose a random seed for
`BpeTrainer`; this run uses its deterministic frequency-based training path with one
ordered training file.
