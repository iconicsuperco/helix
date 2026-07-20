"""Shared fixtures for Forge training tests."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import yaml

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


def _write_resume_files(
    checkpoint_directory: Path,
    model_config: TransformerConfig,
) -> tuple[Path, Path, Path]:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    tokenizer_artifact_path = checkpoint_directory / "tokenizer.json"
    tokenizer_config_path = checkpoint_directory / "tokenizer.yaml"
    model_config_path = checkpoint_directory / "model.yaml"
    train_path = checkpoint_directory / "train.txt"
    validation_path = checkpoint_directory / "validation.txt"

    tokenizer_artifact_path.write_text('{"fixture": "tokenizer"}\n', encoding="utf-8")
    tokenizer_config_path.write_text(
        yaml.safe_dump(
            {
                "artifacts_path": str(checkpoint_directory),
                "tokenizer_file": tokenizer_artifact_path.name,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        yaml.safe_dump(
            {
                "n_layer": model_config.n_layer,
                "n_head": model_config.n_head,
                "n_embd": model_config.n_embd,
                "block_size": model_config.block_size,
                "dropout": model_config.dropout,
                "tokenizer_config": str(tokenizer_config_path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    train_path.write_text("train fixture\n", encoding="utf-8")
    validation_path.write_text("validation fixture\n", encoding="utf-8")
    return model_config_path, train_path, validation_path


def _checkpoint_config(
    config: TrainingConfig,
    model_config: TransformerConfig,
) -> dict[str, object]:
    return {
        "training": {
            "model_config": str(config.model_config_path),
            "data": {
                "train_paths": [str(path) for path in config.data.train_paths],
                "validation_paths": [str(path) for path in config.data.validation_paths],
                "validation_fraction": config.data.validation_fraction,
                "context_length": config.data.context_length,
                "append_eos": config.data.append_eos,
            },
            "loader": {
                "batch_size": config.loader.batch_size,
                "num_workers": config.loader.num_workers,
                "shuffle": config.loader.shuffle,
                "drop_last": config.loader.drop_last,
                "pin_memory": config.loader.pin_memory,
            },
            "optimizer": {
                "learning_rate": config.optimizer.learning_rate,
                "min_learning_rate": config.optimizer.min_learning_rate,
                "weight_decay": config.optimizer.weight_decay,
                "beta1": config.optimizer.beta1,
                "beta2": config.optimizer.beta2,
                "gradient_clip_norm": config.optimizer.gradient_clip_norm,
            },
            "loop": {
                "seed": config.loop.seed,
                "device": config.loop.device,
                "deterministic": config.loop.deterministic,
                "max_steps": config.loop.max_steps,
                "warmup_steps": config.loop.warmup_steps,
                "log_interval": config.loop.log_interval,
                "evaluation_interval": config.loop.evaluation_interval,
                "evaluation_batches": config.loop.evaluation_batches,
                "checkpoint_interval": config.loop.checkpoint_interval,
            },
            "checkpoint": {
                "directory": str(config.checkpoint.directory),
                "resume_from": (
                    str(config.checkpoint.resume_from)
                    if config.checkpoint.resume_from is not None
                    else None
                ),
            },
            "logging": asdict(config.logging),
        },
        "model": cast(dict[str, object], asdict(model_config)),
    }


def make_training_config(
    checkpoint_directory: Path,
    *,
    max_steps: int,
    learning_rate: float = 0.05,
    checkpoint_interval: int | None = None,
    model_config_path: Path | None = None,
    train_path: Path | None = None,
    validation_path: Path | None = None,
) -> TrainingConfig:
    """Create a tiny CPU training config with all intervals ending at max_steps."""

    return TrainingConfig(
        model_config_path=(
            model_config_path
            if model_config_path is not None
            else checkpoint_directory / "unused-model.yaml"
        ),
        data=DatasetConfig(
            train_paths=(
                train_path if train_path is not None else checkpoint_directory / "unused-train.txt",
            ),
            validation_paths=(
                validation_path
                if validation_path is not None
                else checkpoint_directory / "unused-validation.txt",
            ),
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
    learning_rate: float = 0.05,
    model_config: TransformerConfig | None = None,
) -> Trainer:
    """Create a deterministic tiny transformer trainer over a repeated token pattern."""

    if model_config is None:
        model_config = TransformerConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            block_size=4,
            dropout=0.0,
            vocab_size=8,
        )
    model_config_path, train_path, validation_path = _write_resume_files(
        checkpoint_directory,
        model_config,
    )
    config = make_training_config(
        checkpoint_directory,
        max_steps=max_steps,
        learning_rate=learning_rate,
        checkpoint_interval=checkpoint_interval,
        model_config_path=model_config_path,
        train_path=train_path,
        validation_path=validation_path,
    )
    token_stream = [0, 1, 2, 3, 4, 5, 6, 7] * 40
    train_dataset = TokenSequenceDataset(token_stream, context_length=4)
    validation_dataset = TokenSequenceDataset(token_stream[:80], context_length=4)
    data_loaders = create_data_loaders(
        train_dataset, validation_dataset, config.loader, seed=config.loop.seed
    )
    set_global_seed(config.loop.seed)
    model = HelixTransformer(model_config)
    return Trainer(
        model=model,
        data_loaders=data_loaders,
        config=config,
        checkpoint_config=_checkpoint_config(config, model_config),
        device=torch.device("cpu"),
    )
