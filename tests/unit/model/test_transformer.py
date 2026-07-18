"""Verify Helix transformer shapes, causality, sizing, determinism, and gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_MODULE_PATH = REPOSITORY_ROOT / "research" / "model"
sys.path.insert(0, str(MODEL_MODULE_PATH))
sys.path.insert(0, str(REPOSITORY_ROOT))

from config import DEFAULT_CONFIG_PATH, TransformerConfig, load_config
from research.tokenizer.tokenizer import encode
from transformer import HelixTransformer


DEFAULT_PARAMETER_ESTIMATE = 16_889_856
PARAMETER_COUNT_TOLERANCE = 0.05
OVERFIT_SEQUENCE_LENGTH = 16
OVERFIT_LOSS_THRESHOLD = 0.1
OVERFIT_MAX_STEPS = 200


def _write_transformer_config(
    path: Path,
    *,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    dropout: float,
) -> None:
    """Write a temporary transformer config that references A1's tokenizer config."""

    values: dict[str, object] = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "dropout": dropout,
        "tokenizer_config": "config/model/tokenizer.yaml",
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _small_config(path: Path) -> TransformerConfig:
    """Load the approximately seven-million-parameter config from spec section 12."""

    _write_transformer_config(
        path,
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=128,
        dropout=0.1,
    )
    return load_config(path)


def _expected_parameter_count(config: TransformerConfig) -> int:
    """Calculate the tied-weight architecture's parameter count by component."""

    embedding_parameters = (
        config.vocab_size * config.n_embd + config.block_size * config.n_embd
    )
    attention_parameters = (
        3 * config.n_embd * config.n_embd
        + 3 * config.n_embd
        + config.n_embd * config.n_embd
        + config.n_embd
    )
    mlp_width = 4 * config.n_embd
    mlp_parameters = (
        config.n_embd * mlp_width
        + mlp_width
        + mlp_width * config.n_embd
        + config.n_embd
    )
    block_layer_norm_parameters = 4 * config.n_embd
    final_layer_norm_parameters = 2 * config.n_embd
    block_parameters = attention_parameters + mlp_parameters + block_layer_norm_parameters
    return (
        embedding_parameters
        + config.n_layer * block_parameters
        + final_layer_norm_parameters
    )


def _overfit_batch() -> torch.Tensor:
    """Encode four fixed A1-tokenizer samples and pad or truncate each to 16 tokens."""

    texts = [
        "Helix learns a tiny fixed batch for this architecture check.",
        "Causal attention never reads tokens from a future position.",
        "Pre-norm residual blocks keep gradients flowing through depth.",
        "This overfit test is not a claim about language-model quality.",
    ]
    pad_token_id = encode("<pad>")[0]
    rows: list[list[int]] = []
    for text in texts:
        token_ids = encode(text)[:OVERFIT_SEQUENCE_LENGTH]
        rows.append(
            token_ids
            + [pad_token_id] * (OVERFIT_SEQUENCE_LENGTH - len(token_ids))
        )
    return torch.tensor(rows, dtype=torch.long)


def test_forward_shape_for_small_and_default_configs(tmp_path: Path) -> None:
    """Produce correctly shaped logits for both section 12 model sizes."""

    configs = [_small_config(tmp_path / "small.yaml"), load_config(DEFAULT_CONFIG_PATH)]
    for config in configs:
        model = HelixTransformer(config).eval()
        idx = torch.randint(config.vocab_size, (2, 32), dtype=torch.long)
        with torch.no_grad():
            logits = model(idx)
        assert logits.shape == (2, 32, config.vocab_size)


def test_causal_mask_prevents_future_token_leakage() -> None:
    """Keep every earlier logit bit-identical when only the last token changes."""

    torch.manual_seed(7)
    config = load_config(DEFAULT_CONFIG_PATH)
    model = HelixTransformer(config).eval()
    idx = torch.randint(config.vocab_size, (2, 8), dtype=torch.long)
    changed_idx = idx.clone()
    changed_idx[:, -1] = (changed_idx[:, -1] + 1) % config.vocab_size
    with torch.no_grad():
        logits = model(idx)
        changed_logits = model(changed_idx)
    assert torch.equal(logits[:, :-1, :], changed_logits[:, :-1, :])


def test_default_parameter_count_matches_corrected_estimate() -> None:
    """Keep the default count within five percent of the corrected hand estimate."""

    config = load_config(DEFAULT_CONFIG_PATH)
    model = HelixTransformer(config)
    parameter_count = model.parameter_count
    assert model.output_head.weight is model.token_embedding.weight
    assert parameter_count == _expected_parameter_count(config)
    assert parameter_count == DEFAULT_PARAMETER_ESTIMATE
    assert abs(parameter_count - DEFAULT_PARAMETER_ESTIMATE) <= (
        DEFAULT_PARAMETER_ESTIMATE * PARAMETER_COUNT_TOLERANCE
    )


def test_fixed_seed_produces_identical_weights_and_logits() -> None:
    """Reproduce every initial parameter and inference logit from a fixed seed."""

    config = TransformerConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        dropout=0.1,
        vocab_size=load_config(DEFAULT_CONFIG_PATH).vocab_size,
    )
    torch.manual_seed(19)
    first_model = HelixTransformer(config).eval()
    torch.manual_seed(19)
    second_model = HelixTransformer(config).eval()
    for first_parameter, second_parameter in zip(
        first_model.parameters(), second_model.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)

    idx = torch.randint(config.vocab_size, (2, 12), dtype=torch.long)
    with torch.no_grad():
        first_logits = first_model(idx)
        second_logits = second_model(idx)
    assert torch.equal(first_logits, second_logits)


def test_single_batch_can_overfit_as_architecture_sanity_check() -> None:
    """Drive one tiny batch near zero loss to prove end-to-end gradient flow."""

    # This bounded loop checks architecture correctness only. It is not a model-quality
    # claim and is not Helix's A3 training loop or A6 evaluation harness.
    vocab_size = load_config(DEFAULT_CONFIG_PATH).vocab_size
    config = TransformerConfig(
        n_layer=1,
        n_head=1,
        n_embd=32,
        block_size=OVERFIT_SEQUENCE_LENGTH,
        dropout=0.0,
        vocab_size=vocab_size,
    )
    torch.manual_seed(23)
    model = HelixTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = _overfit_batch()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    loss = torch.tensor(float("inf"))

    for _ in range(OVERFIT_MAX_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), targets.reshape(-1)
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if loss.item() < OVERFIT_LOSS_THRESHOLD:
            break

    assert loss.item() < OVERFIT_LOSS_THRESHOLD


def test_config_driven_resizing_changes_parameter_count(tmp_path: Path) -> None:
    """Resize all learned tables and blocks solely by loading different configs."""

    small_config = _small_config(tmp_path / "small-resized.yaml")
    default_config = load_config(DEFAULT_CONFIG_PATH)
    small_model = HelixTransformer(small_config)
    default_model = HelixTransformer(default_config)
    assert small_model.parameter_count == _expected_parameter_count(small_config)
    assert default_model.parameter_count == _expected_parameter_count(default_config)
    assert small_model.parameter_count < default_model.parameter_count
