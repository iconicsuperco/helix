"""Compute causal language-model perplexity over a UTF-8 text file."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as functional

from research.model.transformer import HelixTransformer

PERPLEXITY_BATCH_SIZE = 8


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface required for perplexity."""

    def encode(self, text: str) -> list[int]: ...


def compute_perplexity(
    model: HelixTransformer,
    tokenizer: TokenizerProtocol,
    eval_text_path: Path,
) -> float:
    """Return token-weighted perplexity without writing artifacts or changing model mode."""

    text = eval_text_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        raise ValueError(f"Evaluation text at {eval_text_path} must encode to at least two tokens")

    windows_by_length: defaultdict[int, list[tuple[list[int], list[int]]]] = defaultdict(list)
    block_size = model.config.block_size
    for start in range(0, len(token_ids) - 1, block_size):
        window = token_ids[start : start + block_size + 1]
        if len(window) >= 2:
            windows_by_length[len(window) - 1].append((window[:-1], window[1:]))

    device = next(model.parameters()).device
    was_training = model.training
    total_negative_log_likelihood = 0.0
    total_predicted_tokens = 0
    model.eval()
    try:
        with torch.inference_mode():
            for windows in windows_by_length.values():
                for offset in range(0, len(windows), PERPLEXITY_BATCH_SIZE):
                    batch = windows[offset : offset + PERPLEXITY_BATCH_SIZE]
                    inputs = torch.tensor(
                        [item[0] for item in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    targets = torch.tensor(
                        [item[1] for item in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    logits = model(inputs)
                    loss = functional.cross_entropy(
                        logits.reshape(-1, model.config.vocab_size),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    total_negative_log_likelihood += float(loss.item())
                    total_predicted_tokens += targets.numel()
    finally:
        model.train(was_training)

    return math.exp(total_negative_log_likelihood / total_predicted_tokens)
