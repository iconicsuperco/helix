"""Test complete Forge checkpoint persistence."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from helix.common.seed import set_global_seed
from research.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_training_and_rng_state(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / (step + 1))
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_parameters = {
        name: parameter.detach().clone() for name, parameter in model.state_dict().items()
    }

    set_global_seed(41)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={"run": "test"},
        global_step=7,
        data_epoch=2,
        batches_consumed=3,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(2)

    restored_model = nn.Linear(3, 2)
    restored_optimizer = AdamW(restored_model.parameters(), lr=0.5)
    restored_scheduler = LambdaLR(restored_optimizer, lr_lambda=lambda step: 1.0)
    set_global_seed(99)
    state = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location=torch.device("cpu"),
    )

    assert state.global_step == 7
    assert state.data_epoch == 2
    assert state.batches_consumed == 3
    assert state.config == {"run": "test"}
    for name, parameter in restored_model.state_dict().items():
        assert torch.equal(parameter, expected_parameters[name])
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(2), expected_torch)
