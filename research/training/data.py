"""Deterministic text loading, tokenization, and PyTorch batching."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from research.tokenizer.tokenizer import encode, encode_batch
from research.training.config import DatasetConfig, LoaderConfig

Batch = tuple[torch.Tensor, torch.Tensor]
BatchEncoder = Callable[[list[str]], list[list[int]]]
PARAGRAPH_SEPARATOR = re.compile(r"(?:\r?\n[ \t]*){2,}")


@dataclass(frozen=True)
class DocumentSplit:
    """Deterministic train and validation document partitions."""

    train: tuple[str, ...]
    validation: tuple[str, ...]


@dataclass(frozen=True)
class TrainingDataLoaders:
    """DataLoaders plus the generator controlling training order."""

    train: DataLoader[Batch]
    validation: DataLoader[Batch]
    train_generator: torch.Generator
    seed: int


class TokenSequenceDataset(Dataset[Batch]):
    """Expose a token stream as fixed-length next-token prediction examples."""

    def __init__(self, token_ids: Sequence[int], context_length: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")

        self.context_length = context_length
        self._tokens = torch.tensor(token_ids, dtype=torch.long)
        self._sequence_count = max(0, (len(token_ids) - 1) // context_length)
        if self._sequence_count == 0:
            raise ValueError(
                f"Token stream needs at least {context_length + 1} tokens for one sequence"
            )

    def __len__(self) -> int:
        return self._sequence_count

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < self._sequence_count:
            raise IndexError(index)
        start = index * self.context_length
        chunk = self._tokens[start : start + self.context_length + 1]
        return chunk[:-1], chunk[1:]


def load_text_documents(paths: Sequence[Path]) -> tuple[str, ...]:
    """Load UTF-8 text files and treat blank-line-separated paragraphs as documents."""

    documents: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Text dataset does not exist: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Text dataset must be UTF-8: {path}") from error
        documents.extend(
            paragraph.strip() for paragraph in PARAGRAPH_SEPARATOR.split(text) if paragraph.strip()
        )
    if not documents:
        raise ValueError("Text dataset did not contain any non-empty documents")
    return tuple(documents)


def split_documents(
    documents: Sequence[str], validation_fraction: float, seed: int
) -> DocumentSplit:
    """Shuffle and partition documents reproducibly without overlap."""

    if len(documents) < 2:
        raise ValueError("At least two documents are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0.0, 1.0)")

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    return DocumentSplit(
        train=tuple(shuffled[validation_count:]),
        validation=tuple(shuffled[:validation_count]),
    )


def tokenize_documents(
    documents: Sequence[str],
    *,
    encoder: BatchEncoder = encode_batch,
    separator_token_id: int | None = None,
) -> list[int]:
    """Tokenize documents and concatenate them with an optional separator token."""

    encoded_documents = encoder(list(documents))
    if len(encoded_documents) != len(documents):
        raise ValueError("Tokenizer returned a different number of documents than it received")

    token_ids: list[int] = []
    for encoded in encoded_documents:
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in encoded):
            raise ValueError("Tokenizer emitted an invalid token ID")
        token_ids.extend(encoded)
        if separator_token_id is not None:
            token_ids.append(separator_token_id)
    return token_ids


def build_datasets(
    config: DatasetConfig,
    *,
    seed: int,
    encoder: BatchEncoder = encode_batch,
    eos_token_id: int | None = None,
) -> tuple[TokenSequenceDataset, TokenSequenceDataset]:
    """Load, split, tokenize, and sequence configured text datasets."""

    train_documents = load_text_documents(config.train_paths)
    if config.validation_paths:
        validation_documents = load_text_documents(config.validation_paths)
    else:
        split = split_documents(train_documents, config.validation_fraction, seed=seed)
        train_documents = split.train
        validation_documents = split.validation

    separator_token_id = eos_token_id
    if config.append_eos and separator_token_id is None:
        eos_ids = encode("<eos>")
        if len(eos_ids) != 1:
            raise ValueError("The configured tokenizer must encode <eos> as one token")
        separator_token_id = eos_ids[0]
    if not config.append_eos:
        separator_token_id = None

    train_tokens = tokenize_documents(
        train_documents, encoder=encoder, separator_token_id=separator_token_id
    )
    validation_tokens = tokenize_documents(
        validation_documents, encoder=encoder, separator_token_id=separator_token_id
    )
    return (
        TokenSequenceDataset(train_tokens, config.context_length),
        TokenSequenceDataset(validation_tokens, config.context_length),
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_data_loaders(
    train_dataset: TokenSequenceDataset,
    validation_dataset: TokenSequenceDataset,
    config: LoaderConfig,
    *,
    seed: int,
) -> TrainingDataLoaders:
    """Create deterministic train and validation DataLoaders."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=config.pin_memory,
        worker_init_fn=_seed_worker,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training DataLoader has no batches; reduce batch_size or disable drop_last"
        )
    if len(validation_loader) == 0:
        raise ValueError("Validation DataLoader has no batches")
    return TrainingDataLoaders(
        train=train_loader,
        validation=validation_loader,
        train_generator=generator,
        seed=seed,
    )
