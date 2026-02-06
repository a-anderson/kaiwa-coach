"""Tests for TTS punctuation normalization."""

from __future__ import annotations

from kaiwacoach.textnorm.tts_punctuation import (
    normalize_for_tts,
    normalize_repeated_punctuation,
    normalize_sentence_breaks,
)


def test_normalize_sentence_breaks() -> None:
    text = "こんにちは。元気ですか？はい！"
    out = normalize_sentence_breaks(text)
    assert "。 " in out
    assert "？ " in out
    assert "！ " in out


def test_normalize_repeated_punctuation() -> None:
    text = "すごい！！！本当？？"
    out = normalize_repeated_punctuation(text)
    assert out == "すごい！本当？"


def test_normalize_for_tts_combines_steps() -> None:
    text = "すごい！！！本当？？"
    out = normalize_for_tts(text)
    assert out == "すごい！ 本当？"
