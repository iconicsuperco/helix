"""Test Forge optimization and checkpoint resume behavior."""

from __future__ import annotations

from pathlib import Path

import torch

from tests.unit.training._helpers import make_trainer


def test_training_loss_decreases_on_repeated_sample(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path / "loss", max_steps=30)

    result = trainer.train()

    assert result.initial_loss is not None
    assert result.final_loss is not None
    assert result.final_loss < result.initial_loss
    assert result.validation_loss is not None
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.is_file()


def test_training_resumes_model_optimizer_scheduler_and_step(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "resume"
    uninterrupted_trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    uninterrupted_trainer.train()
    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in uninterrupted_trainer.model.state_dict().items()
    }

    resumed_trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    resumed_trainer.resume(checkpoint_directory / "step-00000002.pt")

    assert resumed_trainer.global_step == 2
    assert resumed_trainer.scheduler.last_epoch == 2

    resumed_result = resumed_trainer.train()
    assert resumed_result.start_step == 2
    assert resumed_result.end_step == 4
    for name, parameter in resumed_trainer.model.state_dict().items():
        assert torch.equal(parameter, expected_parameters[name])
