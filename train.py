"""Run the config-driven Helix Forge training pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from helix.common.logging import configure_logging, get_logger
from helix.common.seed import set_global_seed
from research.model.config import load_config as load_model_config
from research.model.transformer import HelixTransformer
from research.training.config import DEFAULT_CONFIG_PATH, load_training_config
from research.training.data import build_datasets, create_data_loaders
from research.training.engine import Trainer, resolve_device

LOGGER = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Training YAML or JSON path (default: config/training/forge.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    """Build all configured training components and run Forge."""

    args = _parse_args()
    training_config = load_training_config(args.config.resolve())
    configure_logging(
        level=training_config.logging.level,
        json_logs=training_config.logging.format == "json",
    )
    set_global_seed(training_config.loop.seed)
    torch.use_deterministic_algorithms(training_config.loop.deterministic)

    model_config = load_model_config(training_config.model_config_path)
    if training_config.data.context_length > model_config.block_size:
        raise ValueError(
            f"Training context_length {training_config.data.context_length} exceeds model "
            f"block_size {model_config.block_size}"
        )

    train_dataset, validation_dataset = build_datasets(
        training_config.data, seed=training_config.loop.seed
    )
    data_loaders = create_data_loaders(
        train_dataset,
        validation_dataset,
        training_config.loader,
        seed=training_config.loop.seed,
    )
    device = resolve_device(training_config.loop.device)
    model = HelixTransformer(model_config)
    checkpoint_config: dict[str, object] = {
        "training": training_config.to_dict(),
        "model": asdict(model_config),
    }
    trainer = Trainer(
        model=model,
        data_loaders=data_loaders,
        config=training_config,
        checkpoint_config=checkpoint_config,
        device=device,
        logger=LOGGER,
    )

    LOGGER.info(
        "training_started",
        extra={
            "event": "training_started",
            "device": str(device),
            "parameters": model.parameter_count,
            "train_sequences": len(train_dataset),
            "validation_sequences": len(validation_dataset),
            "max_steps": training_config.loop.max_steps,
        },
    )
    if training_config.checkpoint.resume_from is not None:
        trainer.resume(training_config.checkpoint.resume_from)
    result = trainer.train()
    LOGGER.info(
        "training_completed",
        extra={
            "event": "training_completed",
            "start_step": result.start_step,
            "global_step": result.end_step,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "validation_loss": result.validation_loss,
            "checkpoint": str(result.last_checkpoint) if result.last_checkpoint else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
