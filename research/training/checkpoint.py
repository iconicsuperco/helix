"""Atomic checkpoint persistence and deterministic resume state."""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ResumeState:
    """Training position and persisted configuration from a checkpoint."""

    global_step: int
    data_epoch: int
    batches_consumed: int
    config: dict[str, object]


def capture_rng_state() -> dict[str, object]:
    """Capture all process RNGs used by the single-process trainer."""

    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore Python, NumPy, and PyTorch RNG state from a checkpoint."""

    required = {"python", "numpy", "torch"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint RNG state is missing: {', '.join(sorted(missing))}")

    random.setstate(cast(Any, state["python"]))
    np.random.set_state(cast(Any, state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch"]).cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in cast(list[torch.Tensor], cuda_state)]
        )
    mps_state = state.get("torch_mps")
    if mps_state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(cast(torch.Tensor, mps_state).cpu())


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: dict[str, object],
    global_step: int,
    data_epoch: int,
    batches_consumed: int,
) -> dict[str, object]:
    return {
        "version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "global_step": global_step,
        "data_state": {
            "epoch": data_epoch,
            "batches_consumed": batches_consumed,
        },
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: dict[str, object],
    global_step: int,
    data_epoch: int,
    batches_consumed: int,
) -> None:
    """Atomically save complete training state to one checkpoint file."""

    if global_step < 0 or data_epoch < 0 or batches_consumed < 0:
        raise ValueError("Checkpoint counters must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        global_step=global_step,
        data_epoch=data_epoch,
        batches_consumed=batches_consumed,
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_training_checkpoint(
    directory: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: dict[str, object],
    global_step: int,
    data_epoch: int,
    batches_consumed: int,
) -> Path:
    """Save a numbered checkpoint and atomically refresh `latest.pt`."""

    checkpoint_path = directory / f"step-{global_step:08d}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        global_step=global_step,
        data_epoch=data_epoch,
        batches_consumed=batches_consumed,
    )

    latest_path = directory / "latest.pt"
    temporary_latest = directory / ".latest.pt.tmp"
    try:
        shutil.copyfile(checkpoint_path, temporary_latest)
        os.replace(temporary_latest, latest_path)
    finally:
        temporary_latest.unlink(missing_ok=True)
    return checkpoint_path


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint field '{key}' must be a mapping")
    return value


def _require_nonnegative_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Checkpoint field '{key}' must be a non-negative integer")
    return value


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    map_location: torch.device,
) -> ResumeState:
    """Load model, optimizer, scheduler, counters, config, and all RNG state."""

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"Checkpoint at {path} must contain a mapping")
    payload = cast(dict[str, object], loaded)

    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version!r}; expected {CHECKPOINT_VERSION}"
        )

    model_state = _require_mapping(payload, "model_state")
    optimizer_state = _require_mapping(payload, "optimizer_state")
    scheduler_state = _require_mapping(payload, "scheduler_state")
    config = _require_mapping(payload, "config")
    data_state = _require_mapping(payload, "data_state")
    rng_state = _require_mapping(payload, "rng_state")

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    restore_rng_state(rng_state)
    return ResumeState(
        global_step=_require_nonnegative_int(payload, "global_step"),
        data_epoch=_require_nonnegative_int(data_state, "epoch"),
        batches_consumed=_require_nonnegative_int(data_state, "batches_consumed"),
        config=config,
    )
