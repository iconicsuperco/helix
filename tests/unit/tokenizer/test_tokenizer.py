"""Verify Helix tokenizer round trips, reserved IDs, and batch behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "config" / "model" / "tokenizer.yaml"
TOKENIZER_MODULE_PATH = REPOSITORY_ROOT / "research" / "tokenizer"
sys.path.insert(0, str(TOKENIZER_MODULE_PATH))

from tokenizer import decode, encode, encode_batch


PLAIN_TEXTS = [
    "Helix owns every stage from text to tokens.",
    "A small, real system beats a large imaginary one.",
]
UNICODE_TEXTS = [
    "Café déjà vu — 😀🚀",
    "नमस्ते दुनिया",
    "你好，世界",
]
CODE_TEXTS = [
    "def compare(value: int) -> bool:\n\treturn value <= 10 and value != 3\n",
    "const template = `<section data-id=\"${id}\">& text</section>`;\n  // two spaces\n",
]
EDGE_TEXTS = ["", "a" * 10_000, " \t\n  \t"]
ALL_TEST_TEXTS = PLAIN_TEXTS + UNICODE_TEXTS + CODE_TEXTS + EDGE_TEXTS


def _tokenizer_config() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise AssertionError("Tokenizer config must be a mapping")
    return cast(dict[str, object], loaded)


def _configured_special_tokens() -> list[str]:
    special_tokens = _tokenizer_config().get("special_tokens")
    if not isinstance(special_tokens, list) or not all(isinstance(token, str) for token in special_tokens):
        raise AssertionError("Tokenizer special_tokens must be a list of strings")
    return cast(list[str], special_tokens)


def test_round_trip_plain_english_sentences() -> None:
    for text in PLAIN_TEXTS:
        assert decode(encode(text)) == text


def test_round_trip_emoji_accented_hindi_and_cjk_text() -> None:
    for text in UNICODE_TEXTS:
        assert decode(encode(text)) == text


def test_round_trip_code_with_mixed_indentation_and_special_characters() -> None:
    for text in CODE_TEXTS:
        assert decode(encode(text)) == text


def test_round_trip_empty_repeated_and_whitespace_only_edges() -> None:
    for text in EDGE_TEXTS:
        assert decode(encode(text)) == text


def test_unknown_token_is_never_emitted_for_test_inputs() -> None:
    unknown_token = _tokenizer_config().get("unk_token")
    assert isinstance(unknown_token, str)
    unknown_id = encode(unknown_token)[0]
    for text in ALL_TEST_TEXTS:
        assert unknown_id not in encode(text)


def test_special_tokens_use_fixed_reserved_ids_and_clean_text_stays_clean() -> None:
    special_tokens = _configured_special_tokens()
    for reserved_id, special_token in enumerate(special_tokens):
        assert encode(special_token) == [reserved_id]
    clean_text = "Text without a reserved marker."
    assert not set(encode(clean_text)).intersection(range(len(special_tokens)))
    assert decode(encode(clean_text)) == clean_text


def test_encode_batch_matches_individual_encode_calls() -> None:
    assert encode_batch(ALL_TEST_TEXTS) == [encode(text) for text in ALL_TEST_TEXTS]
