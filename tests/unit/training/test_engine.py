"""Test Forge optimization and checkpoint resume behavior."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from research.model.config import TransformerConfig
from tests.unit.training._helpers import make_trainer


def _nested_mapping(values: dict[str, object], key: str) -> dict[str, object]:
    nested = values[key]
    assert isinstance(nested, dict)
    return cast(dict[str, object], nested)


def _assert_resume_rejected(
    trainer_checkpoint: Path,
    resumed_trainer_config: dict[str, object],
    field: str,
) -> None:
    resumed_trainer = make_trainer(
        trainer_checkpoint.parent,
        max_steps=4,
        checkpoint_interval=2,
    )
    resumed_trainer.checkpoint_config = resumed_trainer_config
    with pytest.raises(ValueError) as error:
        resumed_trainer.resume(trainer_checkpoint)
    message = str(error.value)
    assert "Checkpoint configuration is incompatible" in message
    assert field in message
    assert "previous=" in message
    assert "current=" in message


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


def test_training_checkpoint_records_resume_identity_metadata(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path / "metadata", max_steps=2, checkpoint_interval=2)

    result = trainer.train()

    assert result.last_checkpoint is not None
    payload = torch.load(
        result.last_checkpoint,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    assert isinstance(payload, dict)
    config = _nested_mapping(cast(dict[str, object], payload), "config")
    tokenizer_identity = _nested_mapping(config, "tokenizer_identity")
    dataset_identity = _nested_mapping(config, "dataset_identity")
    assert isinstance(tokenizer_identity["artifact_sha256"], str)
    assert dataset_identity["train"]
    assert dataset_identity["validation"]


def test_resume_rejects_incompatible_model_configuration(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "model-mismatch"
    trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    trainer.train()
    current_config = dict(trainer.checkpoint_config)
    model_config = _nested_mapping(current_config, "model")
    model_config["n_layer"] = 2

    _assert_resume_rejected(
        checkpoint_directory / "step-00000002.pt",
        current_config,
        "model.n_layer",
    )


def test_resume_rejects_incompatible_optimizer_configuration(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "optimizer-mismatch"
    trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    trainer.train()
    current_config = dict(trainer.checkpoint_config)
    training_config = _nested_mapping(current_config, "training")
    optimizer_config = _nested_mapping(training_config, "optimizer")
    optimizer_config["learning_rate"] = 0.01

    _assert_resume_rejected(
        checkpoint_directory / "step-00000002.pt",
        current_config,
        "training.optimizer.learning_rate",
    )


def test_resume_rejects_incompatible_tokenizer_identity(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "tokenizer-mismatch"
    trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    trainer.train()
    current_config = dict(trainer.checkpoint_config)
    tokenizer_identity = _nested_mapping(current_config, "tokenizer_identity")
    tokenizer_identity["artifact_sha256"] = "different-tokenizer"

    _assert_resume_rejected(
        checkpoint_directory / "step-00000002.pt",
        current_config,
        "tokenizer_identity.artifact_sha256",
    )


def test_resume_rejects_incompatible_dataset_identity(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "dataset-mismatch"
    trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    trainer.train()
    current_config = dict(trainer.checkpoint_config)
    dataset_identity = _nested_mapping(current_config, "dataset_identity")
    dataset_identity["train"] = [{"path": "changed.txt", "sha256": "changed"}]

    _assert_resume_rejected(
        checkpoint_directory / "step-00000002.pt",
        current_config,
        "dataset_identity.train",
    )


def test_resume_reports_all_mismatched_fields(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "multiple-mismatches"
    trainer = make_trainer(checkpoint_directory, max_steps=4, checkpoint_interval=2)
    trainer.train()
    resumed_trainer = make_trainer(
        checkpoint_directory,
        max_steps=4,
        checkpoint_interval=2,
        model_config=TransformerConfig(
            n_layer=1,
            n_head=1,
            n_embd=16,
            block_size=4,
            dropout=0.2,
            vocab_size=8,
        ),
    )
    training_config = _nested_mapping(resumed_trainer.checkpoint_config, "training")
    optimizer_config = _nested_mapping(training_config, "optimizer")
    optimizer_config["weight_decay"] = 0.5

    with pytest.raises(ValueError) as error:
        resumed_trainer.resume(checkpoint_directory / "step-00000002.pt")
    message = str(error.value)
    assert "model.dropout" in message
    assert "training.optimizer.weight_decay" in message
