"""Run a frozen prompt suite through the existing M8 generation path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from helix.common.config import load_mapping
from helix.common.exceptions import HelixConfigurationError
from research.inference.generation import generate_token_ids, load_inference_session
from research.tokenizer.tokenizer import decode, encode


@dataclass(frozen=True)
class FixedPrompt:
    """One stable prompt in a versioned generation suite."""

    prompt_id: str
    category: str
    text: str


@dataclass(frozen=True)
class PromptSuite:
    """Validated fixed prompts and their deterministic generation length."""

    suite_id: str
    version: str
    max_new_tokens: int
    prompts: tuple[FixedPrompt, ...]


def _require_str(values: dict[str, object], key: str, *, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise HelixConfigurationError(
            f"{description.capitalize()} field '{key}' must be a non-empty string"
        )
    return value


def load_prompt_suite(prompt_suite_path: Path) -> PromptSuite:
    """Load and strictly validate one frozen JSON prompt suite."""

    values = load_mapping(
        prompt_suite_path,
        description="prompt suite",
        file_format="json",
    )
    suite_id = _require_str(values, "suite_id", description="prompt suite")
    version = _require_str(values, "version", description="prompt suite")
    max_new_tokens = values.get("max_new_tokens")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise HelixConfigurationError(
            "Prompt suite field 'max_new_tokens' must be a positive integer"
        )
    prompt_values = values.get("prompts")
    if not isinstance(prompt_values, list) or not prompt_values:
        raise HelixConfigurationError("Prompt suite field 'prompts' must be a non-empty list")

    prompts: list[FixedPrompt] = []
    prompt_ids: set[str] = set()
    for index, value in enumerate(prompt_values):
        if not isinstance(value, dict):
            raise HelixConfigurationError(f"Prompt suite prompt at index {index} must be a mapping")
        prompt = cast(dict[str, object], value)
        description = f"prompt suite prompt at index {index}"
        prompt_id = _require_str(prompt, "id", description=description)
        if prompt_id in prompt_ids:
            raise HelixConfigurationError(f"Prompt suite contains duplicate id {prompt_id!r}")
        prompt_ids.add(prompt_id)
        prompts.append(
            FixedPrompt(
                prompt_id=prompt_id,
                category=_require_str(prompt, "category", description=description),
                text=_require_str(prompt, "text", description=description),
            )
        )
    return PromptSuite(
        suite_id=suite_id,
        version=version,
        max_new_tokens=max_new_tokens,
        prompts=tuple(prompts),
    )


def run_fixed_suite(
    checkpoint_path: Path,
    prompt_suite_path: Path,
) -> dict[str, str]:
    """Generate one completion per prompt through M8's hardened inference path."""

    suite = load_prompt_suite(prompt_suite_path)
    session = load_inference_session(checkpoint_path, device="cpu")
    completions: dict[str, str] = {}
    for prompt in suite.prompts:
        prompt_ids = encode(prompt.text)
        generated_ids = generate_token_ids(
            session.model,
            prompt_ids,
            max_new_tokens=suite.max_new_tokens,
            device=session.device,
        )
        completions[prompt.prompt_id] = decode(generated_ids[len(prompt_ids) :])
    return completions
