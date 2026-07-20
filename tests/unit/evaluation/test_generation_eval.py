"""Verify fixed-suite validation and inherited tokenizer identity failures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

import research.inference.generation as inference_generation
from evaluation.generation_eval import load_prompt_suite, run_fixed_suite
from helix.common.exceptions import HelixConfigurationError
from research.model.config import TransformerConfig
from research.model.transformer import HelixTransformer
from research.training.checkpoint import save_checkpoint


def _write_prompt_suite(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "suite_id": "fixture-suite",
                "version": "v1",
                "max_new_tokens": 1,
                "prompts": [
                    {
                        "id": "fixture-prompt",
                        "category": "fixture",
                        "text": "A valid prompt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_missing_prompt_suite_hard_fails(tmp_path: Path) -> None:
    with pytest.raises(HelixConfigurationError, match="Unable to read prompt suite"):
        load_prompt_suite(tmp_path / "missing.json")


def test_malformed_prompt_suite_hard_fails(tmp_path: Path) -> None:
    prompt_path = tmp_path / "malformed.json"
    prompt_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(HelixConfigurationError, match="Invalid prompt suite"):
        load_prompt_suite(prompt_path)


def test_fixed_suite_propagates_m8_tokenizer_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    _write_prompt_suite(prompt_path)
    artifact_path = tmp_path / "tokenizer.json"
    artifact_path.write_bytes(b"current tokenizer bytes")
    expected_artifact = b"checkpoint tokenizer bytes"

    config = TransformerConfig(
        n_layer=1,
        n_head=1,
        n_embd=4,
        block_size=4,
        dropout=0.0,
        vocab_size=8,
    )
    model = HelixTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={
            "model": asdict(config),
            "tokenizer_identity": {
                "artifact_sha256": hashlib.sha256(expected_artifact).hexdigest()
            },
        },
        global_step=1,
        data_epoch=0,
        batches_consumed=1,
    )
    monkeypatch.setattr(
        inference_generation,
        "tokenizer_artifact_path",
        lambda: artifact_path,
    )

    with pytest.raises(ValueError, match="Tokenizer artifact is incompatible"):
        run_fixed_suite(checkpoint_path, prompt_path)
