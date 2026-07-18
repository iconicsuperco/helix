"""Load and validate configuration for Helix training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from helix.common.config import load_mapping, require_bool, require_positive_int, require_str
from helix.common.exceptions import HelixConfigurationError
from helix.common.paths import relative_to_repository, resolve_repository_path
from helix.common.seed import MAX_NUMPY_SEED

DEFAULT_CONFIG_PATH = resolve_repository_path("config/training/forge.yaml")
SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"})
SUPPORTED_LOG_FORMATS = frozenset({"json", "text"})


@dataclass(frozen=True)
class DatasetConfig:
    """Text sources and token sequence settings."""

    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    validation_fraction: float
    context_length: int
    append_eos: bool


@dataclass(frozen=True)
class LoaderConfig:
    """PyTorch DataLoader settings."""

    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool
    pin_memory: bool


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW, clipping, and learning-rate settings."""

    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip_norm: float


@dataclass(frozen=True)
class LoopConfig:
    """Training loop cadence and reproducibility settings."""

    seed: int
    device: str
    deterministic: bool
    max_steps: int
    warmup_steps: int
    log_interval: int
    evaluation_interval: int
    evaluation_batches: int
    checkpoint_interval: int


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint storage and optional resume source."""

    directory: Path
    resume_from: Path | None


@dataclass(frozen=True)
class LoggingConfig:
    """Structured logging settings for a training run."""

    level: str
    format: str


@dataclass(frozen=True)
class TrainingConfig:
    """Complete validated configuration for one training run."""

    model_config_path: Path
    data: DatasetConfig
    loader: LoaderConfig
    optimizer: OptimizerConfig
    loop: LoopConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, object]:
        """Return a repository-relative, checkpoint-safe representation."""

        resume_from = self.checkpoint.resume_from
        return {
            "model_config": str(relative_to_repository(self.model_config_path)),
            "data": {
                "train_paths": [
                    str(relative_to_repository(path)) for path in self.data.train_paths
                ],
                "validation_paths": [
                    str(relative_to_repository(path)) for path in self.data.validation_paths
                ],
                "validation_fraction": self.data.validation_fraction,
                "context_length": self.data.context_length,
                "append_eos": self.data.append_eos,
            },
            "loader": {
                "batch_size": self.loader.batch_size,
                "num_workers": self.loader.num_workers,
                "shuffle": self.loader.shuffle,
                "drop_last": self.loader.drop_last,
                "pin_memory": self.loader.pin_memory,
            },
            "optimizer": {
                "learning_rate": self.optimizer.learning_rate,
                "min_learning_rate": self.optimizer.min_learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "beta1": self.optimizer.beta1,
                "beta2": self.optimizer.beta2,
                "gradient_clip_norm": self.optimizer.gradient_clip_norm,
            },
            "loop": {
                "seed": self.loop.seed,
                "device": self.loop.device,
                "deterministic": self.loop.deterministic,
                "max_steps": self.loop.max_steps,
                "warmup_steps": self.loop.warmup_steps,
                "log_interval": self.loop.log_interval,
                "evaluation_interval": self.loop.evaluation_interval,
                "evaluation_batches": self.loop.evaluation_batches,
                "checkpoint_interval": self.loop.checkpoint_interval,
            },
            "checkpoint": {
                "directory": str(relative_to_repository(self.checkpoint.directory)),
                "resume_from": (
                    str(relative_to_repository(resume_from)) if resume_from is not None else None
                ),
            },
            "logging": {"level": self.logging.level, "format": self.logging.format},
        }


def _require_mapping(values: dict[str, object], key: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise HelixConfigurationError(f"Training config field '{key}' must be a mapping")
    return value


def _require_nonnegative_int(values: dict[str, object], key: str, description: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a non-negative integer"
        )
    return value


def _require_number(values: dict[str, object], key: str, description: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HelixConfigurationError(f"{description.capitalize()} field '{key}' must be numeric")
    return float(value)


def _require_path(values: dict[str, object], key: str, description: str) -> Path:
    return resolve_repository_path(require_str(values, key, description=description))


def _require_paths(values: dict[str, object], key: str, description: str) -> tuple[Path, ...]:
    value = values.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a list of non-empty paths"
        )
    return tuple(resolve_repository_path(item) for item in value)


def _optional_path(values: dict[str, object], key: str, description: str) -> Path | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be null or a non-empty path"
        )
    return resolve_repository_path(value)


def load_training_config(config_path: Path = DEFAULT_CONFIG_PATH) -> TrainingConfig:
    """Load the nested Forge training configuration from YAML or JSON."""

    values = load_mapping(config_path, description="training config")
    data_values = _require_mapping(values, "data")
    loader_values = _require_mapping(values, "loader")
    optimizer_values = _require_mapping(values, "optimizer")
    loop_values = _require_mapping(values, "loop")
    checkpoint_values = _require_mapping(values, "checkpoint")
    logging_values = _require_mapping(values, "logging")

    train_paths = _require_paths(data_values, "train_paths", "data config")
    if not train_paths:
        raise HelixConfigurationError("Data config field 'train_paths' must not be empty")
    validation_paths = _require_paths(data_values, "validation_paths", "data config")
    validation_fraction = _require_number(data_values, "validation_fraction", "data config")
    if validation_paths and validation_fraction != 0.0:
        raise HelixConfigurationError(
            "Data config must use either validation_paths or validation_fraction, not both"
        )
    if not validation_paths and not 0.0 < validation_fraction < 1.0:
        raise HelixConfigurationError(
            "Data config validation_fraction must be in (0.0, 1.0) when no validation_paths exist"
        )

    learning_rate = _require_number(optimizer_values, "learning_rate", "optimizer config")
    min_learning_rate = _require_number(optimizer_values, "min_learning_rate", "optimizer config")
    weight_decay = _require_number(optimizer_values, "weight_decay", "optimizer config")
    beta1 = _require_number(optimizer_values, "beta1", "optimizer config")
    beta2 = _require_number(optimizer_values, "beta2", "optimizer config")
    gradient_clip_norm = _require_number(optimizer_values, "gradient_clip_norm", "optimizer config")
    if not 0.0 < learning_rate or not 0.0 <= min_learning_rate <= learning_rate:
        raise HelixConfigurationError(
            "Optimizer learning rates must satisfy 0 <= min_learning_rate <= learning_rate"
        )
    if weight_decay < 0.0:
        raise HelixConfigurationError("Optimizer weight_decay must be non-negative")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise HelixConfigurationError("Optimizer beta values must be in [0.0, 1.0)")
    if gradient_clip_norm <= 0.0:
        raise HelixConfigurationError("Optimizer gradient_clip_norm must be positive")

    max_steps = require_positive_int(loop_values, "max_steps", description="loop config")
    warmup_steps = _require_nonnegative_int(loop_values, "warmup_steps", "loop config")
    if warmup_steps >= max_steps:
        raise HelixConfigurationError("Loop warmup_steps must be smaller than max_steps")
    device = require_str(loop_values, "device", description="loop config").lower()
    if device not in SUPPORTED_DEVICES:
        expected = ", ".join(sorted(SUPPORTED_DEVICES))
        raise HelixConfigurationError(f"Loop device must be one of: {expected}")
    seed = _require_nonnegative_int(loop_values, "seed", "loop config")
    if seed > MAX_NUMPY_SEED:
        raise HelixConfigurationError(f"Loop seed must not exceed {MAX_NUMPY_SEED}")

    log_format = require_str(logging_values, "format", description="logging config").lower()
    if log_format not in SUPPORTED_LOG_FORMATS:
        raise HelixConfigurationError("Logging format must be 'json' or 'text'")

    return TrainingConfig(
        model_config_path=_require_path(values, "model_config", "training config"),
        data=DatasetConfig(
            train_paths=train_paths,
            validation_paths=validation_paths,
            validation_fraction=validation_fraction,
            context_length=require_positive_int(
                data_values, "context_length", description="data config"
            ),
            append_eos=require_bool(data_values, "append_eos", description="data config"),
        ),
        loader=LoaderConfig(
            batch_size=require_positive_int(
                loader_values, "batch_size", description="loader config"
            ),
            num_workers=_require_nonnegative_int(loader_values, "num_workers", "loader config"),
            shuffle=require_bool(loader_values, "shuffle", description="loader config"),
            drop_last=require_bool(loader_values, "drop_last", description="loader config"),
            pin_memory=require_bool(loader_values, "pin_memory", description="loader config"),
        ),
        optimizer=OptimizerConfig(
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            gradient_clip_norm=gradient_clip_norm,
        ),
        loop=LoopConfig(
            seed=seed,
            device=device,
            deterministic=require_bool(loop_values, "deterministic", description="loop config"),
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            log_interval=require_positive_int(
                loop_values, "log_interval", description="loop config"
            ),
            evaluation_interval=require_positive_int(
                loop_values, "evaluation_interval", description="loop config"
            ),
            evaluation_batches=require_positive_int(
                loop_values, "evaluation_batches", description="loop config"
            ),
            checkpoint_interval=require_positive_int(
                loop_values, "checkpoint_interval", description="loop config"
            ),
        ),
        checkpoint=CheckpointConfig(
            directory=_require_path(checkpoint_values, "directory", "checkpoint config"),
            resume_from=_optional_path(checkpoint_values, "resume_from", "checkpoint config"),
        ),
        logging=LoggingConfig(
            level=require_str(logging_values, "level", description="logging config"),
            format=log_format,
        ),
    )
