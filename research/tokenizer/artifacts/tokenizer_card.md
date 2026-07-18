# Helix Tokenizer Card

## Model

- Algorithm: byte-level BPE
- Vocabulary size: 16000
- Normalization: NFC, no lowercasing
- Pre-tokenizer: ByteLevel
- `tokenizers` version: 0.22.1
- Training date (UTC): 2026-07-18

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

- Name: gutenberg-moby-dick
- Source: https://www.gutenberg.org/ebooks/15
- License: Public domain in the USA; verify status in the deployment jurisdiction.
- Training content SHA-256: `bc1a181e166a877841874cb8edf81b8b3af6e858b4c49ed276d1ef0d44c2772b`
- Held-out content SHA-256: `6e5a61a28f1cf81606f0499bdd069a9bf5f1d59a8e5fa2ac3b89280a9a1ae4d9`
- Corpus status: This is a placeholder training corpus and must be replaced before the v0.1 model training run.

The held-out split is assigned deterministically from each unique cleaned line's SHA-256
hash and is excluded from BPE training.

## Evaluation

- Held-out compression ratio: **3.936784 characters/token**

The ratio is total Unicode characters divided by total emitted tokens over the held-out
sample. Empty samples would report `0.0`; this training run used a non-empty sample.

## Reproducibility

Corpus cleaning, exact-line deduplication, and held-out assignment are deterministic for
the same raw files and config. Hugging Face `tokenizers` does not expose a random seed for
`BpeTrainer`; this run uses its deterministic frequency-based training path with one
ordered training file.
