"""Single-process training engine for the Helix decoder-only transformer."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from helix.common.config import load_mapping
from helix.common.exceptions import HelixPathError
from helix.common.logging import get_logger
from helix.common.paths import relative_to_repository, resolve_repository_path
from research.training.checkpoint import load_checkpoint, save_training_checkpoint
from research.training.config import TrainingConfig
from research.training.data import Batch, TrainingDataLoaders

LOGGER = get_logger(__name__)
MAX_GENERATOR_SEED = 2**63 - 1


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed or already-complete training invocation."""

    start_step: int
    end_step: int
    initial_loss: float | None
    final_loss: float | None
    validation_loss: float | None
    last_checkpoint: Path | None


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch training device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def create_lr_scheduler(optimizer: AdamW, config: TrainingConfig) -> LRScheduler:
    """Create a linear-warmup, cosine-decay learning-rate schedule."""

    loop = config.loop
    optimizer_config = config.optimizer
    minimum_ratio = optimizer_config.min_learning_rate / optimizer_config.learning_rate

    def multiplier(step: int) -> float:
        if loop.warmup_steps and step < loop.warmup_steps:
            return (step + 1) / loop.warmup_steps
        decay_steps = loop.max_steps - loop.warmup_steps
        progress = min(1.0, max(0.0, (step - loop.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _display_path(path: Path) -> str:
    try:
        return str(relative_to_repository(path))
    except HelixPathError:
        return str(path)


def _resolve_configured_path(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return resolve_repository_path(path)


def _file_sha256(path: Path, *, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, description: str) -> dict[str, str]:
    resolved_path = path.expanduser().resolve()
    return {
        "path": _display_path(resolved_path),
        "sha256": _file_sha256(resolved_path, description=description),
    }


def _dataset_identity(config: TrainingConfig) -> dict[str, object]:
    return {
        "train": [
            _file_identity(path, description="Training dataset") for path in config.data.train_paths
        ],
        "validation": [
            _file_identity(path, description="Validation dataset")
            for path in config.data.validation_paths
        ],
    }


def _tokenizer_identity(config: TrainingConfig) -> dict[str, str]:
    model_values = load_mapping(config.model_config_path, description="model config")
    tokenizer_config_path = _resolve_configured_path(
        model_values.get("tokenizer_config"),
        description="Model config field 'tokenizer_config'",
    )
    tokenizer_values = load_mapping(tokenizer_config_path, description="tokenizer config")
    artifacts_path = _resolve_configured_path(
        tokenizer_values.get("artifacts_path"),
        description="Tokenizer config field 'artifacts_path'",
    )
    tokenizer_file = tokenizer_values.get("tokenizer_file")
    if not isinstance(tokenizer_file, str) or not tokenizer_file:
        raise ValueError("Tokenizer config field 'tokenizer_file' must be a non-empty string")
    tokenizer_artifact_path = (artifacts_path / tokenizer_file).resolve()
    return {
        "config_path": _display_path(tokenizer_config_path),
        "artifact_path": _display_path(tokenizer_artifact_path),
        "artifact_sha256": _file_sha256(
            tokenizer_artifact_path,
            description="Tokenizer artifact",
        ),
    }


def _checkpoint_config_with_resume_metadata(
    checkpoint_config: dict[str, object],
    config: TrainingConfig,
) -> dict[str, object]:
    updated_config: dict[str, object] = deepcopy(checkpoint_config)
    updated_config["tokenizer_identity"] = _tokenizer_identity(config)
    updated_config["dataset_identity"] = _dataset_identity(config)
    return updated_config


class Trainer:
    """Train, evaluate, checkpoint, and resume one model process."""

    def __init__(
        self,
        *,
        model: nn.Module,
        data_loaders: TrainingDataLoaders,
        config: TrainingConfig,
        checkpoint_config: dict[str, object],
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model.to(device)
        self.data_loaders = data_loaders
        self.config = config
        self.checkpoint_config = _checkpoint_config_with_resume_metadata(
            checkpoint_config,
            config,
        )
        self.device = device
        self.logger = logger if logger is not None else LOGGER
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = create_lr_scheduler(self.optimizer, config)
        self.global_step = 0
        self.data_epoch = 0
        self.batches_consumed = 0
        self._train_iterator: Iterator[Batch] | None = None

    def resume(self, path: Path) -> None:
        """Restore all training state from a checkpoint."""

        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
            expected_config=self.checkpoint_config,
        )
        if state.global_step > self.config.loop.max_steps:
            raise ValueError(
                f"Checkpoint step {state.global_step} exceeds configured max_steps "
                f"{self.config.loop.max_steps}"
            )
        self.global_step = state.global_step
        self.data_epoch = state.data_epoch
        self.batches_consumed = state.batches_consumed
        self._train_iterator = None
        self.logger.info(
            "training_resumed",
            extra={
                "event": "training_resumed",
                "checkpoint": str(path),
                "global_step": self.global_step,
                "data_epoch": self.data_epoch,
                "batches_consumed": self.batches_consumed,
            },
        )

    def _reset_train_iterator(self) -> None:
        generator_seed = (self.data_loaders.seed + self.data_epoch) % MAX_GENERATOR_SEED
        self.data_loaders.train_generator.manual_seed(generator_seed)
        iterator = iter(self.data_loaders.train)
        for _ in range(self.batches_consumed):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError(
                    "Checkpoint data position exceeds the configured training DataLoader"
                ) from error
        self._train_iterator = iterator

    def _next_training_batch(self) -> Batch:
        if self._train_iterator is None:
            self._reset_train_iterator()
        if self._train_iterator is None:
            raise RuntimeError("Training iterator was not initialized")
        try:
            batch = next(self._train_iterator)
        except StopIteration as error:
            self.data_epoch += 1
            self.batches_consumed = 0
            self._reset_train_iterator()
            if self._train_iterator is None:
                raise RuntimeError("Training iterator was not initialized") from error
            batch = next(self._train_iterator)
        self.batches_consumed += 1
        return batch

    def _train_step(self, batch: Batch) -> float:
        inputs, targets = (
            tensor.to(self.device, non_blocking=self.config.loader.pin_memory) for tensor in batch
        )
        self.optimizer.zero_grad(set_to_none=True)
        logits = cast(torch.Tensor, self.model(inputs))
        loss = functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.gradient_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach().item())

    def evaluate(self) -> float:
        """Return token-weighted validation cross-entropy over configured batches."""

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(self.data_loaders.validation):
                if batch_index >= self.config.loop.evaluation_batches:
                    break
                inputs = inputs.to(self.device, non_blocking=self.config.loader.pin_memory)
                targets = targets.to(self.device, non_blocking=self.config.loader.pin_memory)
                logits = cast(torch.Tensor, self.model(inputs))
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
                )
                total_loss += float(loss.item())
                total_tokens += targets.numel()
        if was_training:
            self.model.train()
        if total_tokens == 0:
            raise RuntimeError("Validation produced no target tokens")
        return total_loss / total_tokens

    def _save_checkpoint(self) -> Path:
        checkpoint_path = save_training_checkpoint(
            self.config.checkpoint.directory,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.checkpoint_config,
            global_step=self.global_step,
            data_epoch=self.data_epoch,
            batches_consumed=self.batches_consumed,
        )
        self.logger.info(
            "checkpoint_saved",
            extra={
                "event": "checkpoint_saved",
                "checkpoint": str(checkpoint_path),
                "global_step": self.global_step,
            },
        )
        return checkpoint_path

    def train(self) -> TrainingResult:
        """Run optimization until `max_steps`, periodically evaluating and checkpointing."""

        start_step = self.global_step
        initial_loss: float | None = None
        final_loss: float | None = None
        validation_loss: float | None = None
        last_checkpoint: Path | None = None
        interval_loss = 0.0
        interval_tokens = 0
        interval_steps = 0
        interval_started = time.perf_counter()
        self.model.train()

        while self.global_step < self.config.loop.max_steps:
            batch = self._next_training_batch()
            loss = self._train_step(batch)
            self.global_step += 1
            token_count = batch[1].numel()
            initial_loss = loss if initial_loss is None else initial_loss
            final_loss = loss
            interval_loss += loss * token_count
            interval_tokens += token_count
            interval_steps += 1

            should_evaluate = (
                self.global_step % self.config.loop.evaluation_interval == 0
                or self.global_step == self.config.loop.max_steps
            )
            should_log = self.global_step % self.config.loop.log_interval == 0 or should_evaluate
            if should_evaluate:
                validation_loss = self.evaluate()
            if should_log:
                elapsed = max(time.perf_counter() - interval_started, 1e-12)
                steps_per_second = interval_steps / elapsed
                tokens_per_second = interval_tokens / elapsed
                eta_seconds = (
                    (self.config.loop.max_steps - self.global_step) / steps_per_second
                    if steps_per_second > 0.0
                    else None
                )
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                self.logger.info(
                    "training_metrics",
                    extra={
                        "event": "training_metrics",
                        "global_step": self.global_step,
                        "train_loss": interval_loss / interval_tokens,
                        "validation_loss": validation_loss if should_evaluate else None,
                        "learning_rate": learning_rate,
                        "tokens_per_second": tokens_per_second,
                        "steps_per_second": steps_per_second,
                        "eta_seconds": eta_seconds,
                    },
                )
                interval_loss = 0.0
                interval_tokens = 0
                interval_steps = 0
                interval_started = time.perf_counter()

            if (
                self.global_step % self.config.loop.checkpoint_interval == 0
                or self.global_step == self.config.loop.max_steps
            ):
                last_checkpoint = self._save_checkpoint()

        return TrainingResult(
            start_step=start_step,
            end_step=self.global_step,
            initial_loss=initial_loss,
            final_loss=final_loss,
            validation_loss=validation_loss,
            last_checkpoint=last_checkpoint,
        )
