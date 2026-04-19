"""Tests for JSON extraction and schema enforcement."""

from __future__ import annotations

import json

import pytest

from kaiwacoach.models.json_enforcement import (
    ConversationReply,
    extract_first_json_object,
    parse_with_schema,
)


def test_extracts_first_object_with_trailing_text() -> None:
    text = '{"reply": "ok"} trailing text'
    obj = extract_first_json_object(text)
    assert obj == {"reply": "ok"}


def test_extracts_first_object_with_multiple_objects() -> None:
    text = '{"reply": "ok"} {"reply": "later"}'
    obj = extract_first_json_object(text)
    assert obj == {"reply": "ok"}


def test_parse_with_schema_success() -> None:
    result = parse_with_schema("conversation", '{"reply": "hello"}')
    assert result.error is None
    assert isinstance(result.model, ConversationReply)
    assert result.model.reply == "hello"
    assert result.repaired is False


def test_parse_with_schema_repair_once() -> None:
    def _repair(_: str) -> str:
        return '{"reply": "fixed"}'

    result = parse_with_schema("conversation", "not json", repair_fn=_repair)
    assert result.error is None
    assert result.model is not None
    assert result.model.reply == "fixed"
    assert result.repaired is True


def test_parse_with_schema_fails_after_repair() -> None:
    def _repair(_: str) -> str:
        return '{"reply": 123}'

    result = parse_with_schema("conversation", "not json", repair_fn=_repair)
    assert result.model is None
    assert result.error is not None
    assert result.repaired is True


def test_extracts_json_after_think_tags() -> None:
    text = '<think>Let me compare {user} vs corrected...</think>\n{"explanation": "Fixed spelling."}'
    obj = extract_first_json_object(text)
    assert obj == {"explanation": "Fixed spelling."}


def test_extracts_json_after_think_tags_no_braces_in_think() -> None:
    text = "<think>Analysing the sentence.</think>\n{\"corrected\": \"Bonjour.\"}"
    obj = extract_first_json_object(text)
    assert obj == {"corrected": "Bonjour."}


def test_parse_with_schema_strips_think_tags() -> None:
    text = '<think>reasoning here</think>{"explanation": "Fixed capitalization.", "native": "Bonjour."}'
    result = parse_with_schema("explain_and_native", text)
    assert result.error is None
    assert result.model is not None
    assert result.model.explanation == "Fixed capitalization."


def test_unknown_role_returns_error() -> None:
    result = parse_with_schema("unknown", '{"x": 1}')
    assert result.model is None
    assert result.error == "Unknown role: unknown"


def test_detect_and_correct_valid() -> None:
    result = parse_with_schema(
        "detect_and_correct",
        '{"errors": ["Missing period."], "corrected": "こんにちは。"}',
    )
    assert result.error is None
    assert result.model is not None
    assert result.model.errors == ["Missing period."]
    assert result.model.corrected == "こんにちは。"
    assert result.repaired is False


def test_detect_and_correct_empty_errors() -> None:
    result = parse_with_schema(
        "detect_and_correct",
        '{"errors": [], "corrected": "Bonjour."}',
    )
    assert result.error is None
    assert result.model is not None
    assert result.model.errors == []


def test_detect_and_correct_missing_field_fails() -> None:
    result = parse_with_schema("detect_and_correct", '{"errors": []}')
    assert result.model is None
    assert result.error is not None


def test_explain_and_native_valid() -> None:
    result = parse_with_schema(
        "explain_and_native",
        '{"explanation": "The period was missing.", "native": "こんにちは。"}',
    )
    assert result.error is None
    assert result.model is not None
    assert result.model.explanation == "The period was missing."
    assert result.model.native == "こんにちは。"
    assert result.repaired is False


def test_explain_and_native_missing_field_fails() -> None:
    result = parse_with_schema("explain_and_native", '{"explanation": "ok"}')
    assert result.model is None
    assert result.error is not None


def test_extracts_json_after_gemma_channel_tags() -> None:
    """Gemma 4 26B-A4B thought blocks should be stripped before JSON extraction."""
    text = "<|channel>thought\nLet me analyse the sentence.\n<channel|>\n{\"reply\": \"Bonjour.\"}"
    obj = extract_first_json_object(text)
    assert obj == {"reply": "Bonjour."}


def test_extracts_json_after_gemma_channel_tags_with_embedded_braces() -> None:
    """Channel tag stripping should handle JSON-like content inside the thought block."""
    text = '<|channel>thought\n{"internal": "reasoning"}\n<channel|>\n{"corrected": "ok"}'
    obj = extract_first_json_object(text)
    assert obj == {"corrected": "ok"}


def test_gemma_channel_tag_stripping_is_harmless_when_absent() -> None:
    """The Gemma channel tag regex should not affect output that contains no such tags."""
    text = '{"reply": "こんにちは。"}'
    obj = extract_first_json_object(text)
    assert obj == {"reply": "こんにちは。"}


def test_gemma_truncated_channel_tag_raises_decode_error() -> None:
    """Truncated thought-block output (no closing tag) is not stripped and raises JSONDecodeError.

    This documents the known limitation: _GEMMA_CHANNEL_RE requires the closing <channel|>
    tag. For Ollama, suppress_thinking=True prevents this situation entirely.
    For MLX, truncated output will surface as a parse failure and trigger the repair path.
    """
    truncated = "<|channel>thought\nThis is reasoning that was cut off mid-stream"
    with pytest.raises(json.JSONDecodeError):
        extract_first_json_object(truncated)


def test_explain_and_native_repair_path() -> None:
    def _repair(_: str) -> str:
        return '{"explanation": "Fixed spelling.", "native": "Bonjour."}'

    result = parse_with_schema("explain_and_native", "not json", repair_fn=_repair)
    assert result.error is None
    assert result.model is not None
    assert result.model.explanation == "Fixed spelling."
    assert result.model.native == "Bonjour."
    assert result.repaired is True
