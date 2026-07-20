"""Load a trained Helix checkpoint and generate text autoregressively."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from research.model.config import TransformerConfig
from research.model.transformer import HelixTransformer
from research.tokenizer.tokenizer import decode, encode, tokenizer_artifact_path
from research.training.checkpoint import load_model_checkpoint

SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda", "mps"})


def _require_mapping(
    values: dict[str, object],
    key: str,
    *,
    description: str,
) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} field '{key}' must be a mapping")
    return value


def _require_positive_int(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Checkpoint model config field '{key}' must be a positive integer")
    return value


def _checkpoint_model_config(checkpoint_config: dict[str, object]) -> TransformerConfig:
    values = _require_mapping(checkpoint_config, "model", description="checkpoint config")
    n_layer = _require_positive_int(values, "n_layer")
    n_head = _require_positive_int(values, "n_head")
    n_embd = _require_positive_int(values, "n_embd")
    block_size = _require_positive_int(values, "block_size")
    vocab_size = _require_positive_int(values, "vocab_size")
    dropout_value = values.get("dropout")
    if not isinstance(dropout_value, (int, float)) or isinstance(dropout_value, bool):
        raise ValueError("Checkpoint model config field 'dropout' must be numeric")
    dropout = float(dropout_value)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Checkpoint model config field 'dropout' must be in [0.0, 1.0)")
    if n_embd % n_head != 0:
        raise ValueError("Checkpoint model config field 'n_embd' must be divisible by 'n_head'")
    return TransformerConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        vocab_size=vocab_size,
    )


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer artifact does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tokenizer_identity(
    checkpoint_config: dict[str, object],
    artifact_path: Path,
) -> None:
    """Reject a tokenizer whose artifact hash differs from checkpoint metadata."""

    identity = _require_mapping(
        checkpoint_config,
        "tokenizer_identity",
        description="checkpoint config",
    )
    expected_hash = identity.get("artifact_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(
            "Checkpoint tokenizer_identity field 'artifact_sha256' must be a non-empty string"
        )
    actual_hash = _sha256(artifact_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Tokenizer artifact is incompatible with checkpoint metadata: "
            f"checkpoint_sha256={expected_hash}; current_sha256={actual_hash}"
        )


def resolve_device(requested: str) -> torch.device:
    """Resolve an inference device and reject unavailable accelerators."""

    normalized = requested.lower()
    if normalized not in SUPPORTED_DEVICES:
        expected = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"Inference device must be one of: {expected}")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(normalized)


def generate_token_ids(
    model: HelixTransformer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    """Greedily append token IDs, cropping model context to its block size."""

    if not prompt_ids:
        raise ValueError("Prompt must encode to at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated_ids = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-model.config.block_size :]
            context = torch.tensor([context_ids], dtype=torch.long, device=device)
            logits = model(context)
            next_token_id = int(torch.argmax(logits[0, -1], dim=-1).item())
            generated_ids.append(next_token_id)
    return generated_ids


@dataclass(frozen=True)
class InferenceSession:
    """A loaded model and device ready for deterministic text generation."""

    model: HelixTransformer
    device: torch.device
    checkpoint_step: int

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        """Encode a prompt, greedily generate tokens, and decode all resulting text."""

        generated_ids = generate_token_ids(
            self.model,
            encode(prompt),
            max_new_tokens=max_new_tokens,
            device=self.device,
        )
        return decode(generated_ids)


def load_inference_session(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> InferenceSession:
    """Load a hardened checkpoint and validate its model and tokenizer identity."""

    resolved_device = resolve_device(device)
    checkpoint = load_model_checkpoint(
        checkpoint_path,
        map_location=resolved_device,
    )
    validate_tokenizer_identity(checkpoint.config, tokenizer_artifact_path())
    model = HelixTransformer(_checkpoint_model_config(checkpoint.config)).to(resolved_device)
    model.load_state_dict(checkpoint.model_state)
    model.eval()
    return InferenceSession(
        model=model,
        device=resolved_device,
        checkpoint_step=checkpoint.global_step,
    )
