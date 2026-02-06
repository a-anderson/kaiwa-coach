"""Tests for Japanese invariants."""

from __future__ import annotations

from kaiwacoach.textnorm.invariants import (
    check_japanese_invariant,
    enforce_japanese_invariant,
    extract_japanese_spans,
)


def test_extract_japanese_spans() -> None:
    text = "Hello こんにちは world 世界"
    spans = extract_japanese_spans(text)
    assert spans == ["こんにちは", "世界"]


def test_invariant_ok() -> None:
    original = "こんにちは world"
    candidate = "こんにちは world"
    result = check_japanese_invariant(original, candidate)
    assert result.ok is True
    assert result.mismatches == []


def test_invariant_mismatch() -> None:
    original = "こんにちは world"
    candidate = "コンニチハ world"
    result = check_japanese_invariant(original, candidate)
    assert result.ok is False
    assert result.mismatches[0][0] == "こんにちは"


def test_enforce_invariant_fallback_and_log() -> None:
    original = "こんにちは world"
    candidate = "コンニチハ world"
    messages = []

    def _logger(message: str) -> None:
        messages.append(message)

    output, result = enforce_japanese_invariant(original, candidate, logger=_logger)
    assert output == original
    assert result.ok is False
    assert messages


def test_enforce_invariant_no_logger() -> None:
    original = "こんにちは world"
    candidate = "コンニチハ world"
    output, result = enforce_japanese_invariant(original, candidate, logger=None)
    assert output == original
    assert result.ok is False


def test_invariant_detects_missing_candidate_span() -> None:
    original = "こんにちは 世界"
    candidate = "こんにちは"
    result = check_japanese_invariant(original, candidate)
    assert result.ok is False
    assert len(result.mismatches) >= 1


def test_invariant_multiple_spans_match() -> None:
    original = "こんにちは 世界"
    candidate = "こんにちは 世界"
    result = check_japanese_invariant(original, candidate)
    assert result.ok is True
