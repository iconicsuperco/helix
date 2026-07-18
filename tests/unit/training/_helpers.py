"""Shared fixtures for Forge training tests."""

from __future__ import annotations

from pathlib import Path

import torch

from helix.common.seed import set_global_seed
from research.model.config import TransformerConfig
from research.model.transformer import HelixTransformer
from research.training.config import (
    CheckpointConfig,
    DatasetConfig,
    LoaderConfig,
    LoggingConfig,
    LoopConfig,
    OptimizerConfig,
    TrainingConfig,
)
from research.training.data import TokenSequenceDataset, create_data_loaders
from research.training.engine import Trainer


def make_training_config(
    checkpoint_directory: Path,
    *,
    max_steps: int,
    learning_rate: float = 0.05,
    checkpoint_interval: int | None = None,
) -> TrainingConfig:
    """Create a tiny CPU training config with all intervals ending at max_steps."""

    return TrainingConfig(
        model_config_path=checkpoint_directory / "unused-model.yaml",
        data=DatasetConfig(
            train_paths=(checkpoint_directory / "unused-train.txt",),
            validation_paths=(checkpoint_directory / "unused-validation.txt",),
            validation_fraction=0.0,
            context_length=4,
            append_eos=False,
        ),
        loader=LoaderConfig(
            batch_size=4,
            num_workers=0,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        ),
        optimizer=OptimizerConfig(
            learning_rate=learning_rate,
            min_learning_rate=learning_rate / 10,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            gradient_clip_norm=1.0,
        ),
        loop=LoopConfig(
            seed=31,
            device="cpu",
            deterministic=True,
            max_steps=max_steps,
            warmup_steps=0,
            log_interval=max_steps,
            evaluation_interval=max_steps,
            evaluation_batches=2,
            checkpoint_interval=(
                checkpoint_interval if checkpoint_interval is not None else max_steps
            ),
        ),
        checkpoint=CheckpointConfig(directory=checkpoint_directory, resume_from=None),
        logging=LoggingConfig(level="INFO", format="json"),
    )


def make_trainer(
    checkpoint_directory: Path,
    *,
    max_steps: int,
    checkpoint_interval: int | None = None,
) -> Trainer:
    """Create a deterministic tiny transformer trainer over a repeated token pattern."""

    config = make_training_config(
        checkpoint_directory,
        max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
    )
    token_stream = [0, 1, 2, 3, 4, 5, 6, 7] * 40
    train_dataset = TokenSequenceDataset(token_stream, context_length=4)
    validation_dataset = TokenSequenceDataset(token_stream[:80], context_length=4)
    data_loaders = create_data_loaders(
        train_dataset, validation_dataset, config.loader, seed=config.loop.seed
    )
    set_global_seed(config.loop.seed)
    model = HelixTransformer(
        TransformerConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            block_size=4,
            dropout=0.0,
            vocab_size=8,
        )
    )
    return Trainer(
        model=model,
        data_loaders=data_loaders,
        config=config,
        checkpoint_config={"test": True},
        device=torch.device("cpu"),
    )
