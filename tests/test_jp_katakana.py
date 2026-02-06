"""Tests for katakana normalization."""

from __future__ import annotations

from kaiwacoach.textnorm.jp_katakana import contains_japanese, contains_latin, normalise_katakana


def test_contains_japanese_and_latin() -> None:
    assert contains_japanese("こんにちは")
    assert not contains_japanese("hello")
    assert contains_latin("hello")
    assert not contains_latin("こんにちは")


def test_normalise_katakana_uses_llm_and_restores_spans() -> None:
    text = "This is a test https://example.com"

    def _rewrite(s: str) -> str:
        # LLM rewrites only the English phrase (simulated).
        return s.replace("This is a test", "ディス イズ ア テスト")

    result = normalise_katakana(text, llm_rewrite_fn=_rewrite)
    assert "ディス イズ ア テスト" in result.text
    assert "https://example.com" in result.text
