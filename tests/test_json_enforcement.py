"""Tests for JSON extraction and schema enforcement."""

from __future__ import annotations

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


def test_unknown_role_returns_error() -> None:
    result = parse_with_schema("unknown", '{"x": 1}')
    assert result.model is None
    assert result.error == "Unknown role: unknown"
