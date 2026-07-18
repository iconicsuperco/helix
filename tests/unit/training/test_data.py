"""Test deterministic dataset construction and PyTorch batching."""

from __future__ import annotations

from pathlib import Path

import torch

from research.training.config import DatasetConfig, LoaderConfig
from research.training.data import (
    TokenSequenceDataset,
    build_datasets,
    create_data_loaders,
    load_text_documents,
    split_documents,
)


def test_text_loading_and_seeded_split_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "documents.txt"
    path.write_text("\n\n".join(f"document {index}" for index in range(12)), encoding="utf-8")
    documents = load_text_documents((path,))

    first = split_documents(documents, validation_fraction=0.25, seed=17)
    replay = split_documents(documents, validation_fraction=0.25, seed=17)
    different = split_documents(documents, validation_fraction=0.25, seed=18)

    assert first == replay
    assert first != different
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train + first.validation) == set(documents)


def test_token_sequence_dataset_builds_shifted_fixed_length_examples() -> None:
    dataset = TokenSequenceDataset(list(range(10)), context_length=3)

    assert len(dataset) == 3
    inputs, targets = dataset[1]
    assert torch.equal(inputs, torch.tensor([3, 4, 5]))
    assert torch.equal(targets, torch.tensor([4, 5, 6]))


def test_dataset_pipeline_uses_existing_tokenizer(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text(
        "\n\n".join(["Helix trains on deterministic text sequences."] * 8), encoding="utf-8"
    )
    validation_path.write_text(
        "\n\n".join(["Validation remains separate from training."] * 4), encoding="utf-8"
    )
    config = DatasetConfig(
        train_paths=(train_path,),
        validation_paths=(validation_path,),
        validation_fraction=0.0,
        context_length=8,
        append_eos=True,
    )

    train_dataset, validation_dataset = build_datasets(config, seed=7)

    assert len(train_dataset) > 0
    assert len(validation_dataset) > 0
    assert train_dataset[0][0].shape == (8,)
    assert validation_dataset[0][1].shape == (8,)


def test_data_loader_batching_and_shuffle_replay() -> None:
    train_dataset = TokenSequenceDataset(list(range(100)), context_length=4)
    validation_dataset = TokenSequenceDataset(list(range(40)), context_length=4)
    config = LoaderConfig(
        batch_size=3,
        num_workers=0,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    first = create_data_loaders(train_dataset, validation_dataset, config, seed=23)
    replay = create_data_loaders(train_dataset, validation_dataset, config, seed=23)
    first_inputs, first_targets = next(iter(first.train))
    replay_inputs, replay_targets = next(iter(replay.train))

    assert first_inputs.shape == (3, 4)
    assert first_targets.shape == (3, 4)
    assert torch.equal(first_inputs, replay_inputs)
    assert torch.equal(first_targets, replay_targets)
