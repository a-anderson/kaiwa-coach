"""Tests for Japanese TTS normalisation."""

from __future__ import annotations

import json
from pathlib import Path

from kaiwacoach.textnorm.invariants import enforce_japanese_invariant
from kaiwacoach.textnorm.jp_katakana import normalise_katakana
from kaiwacoach.textnorm.tts_punctuation import normalize_for_tts


def _load_cases() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parent / "fixtures" / "jp_tts_normalisation_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_jp_tts_normalisation_cases() -> None:
    cases = _load_cases()
    for case in cases:
        original = case["input"]
        rewritten = case["llm_rewrite"]
        expected = case["expected"]

        def _fake_llm(text: str) -> str:
            return rewritten

        katakana_result = normalise_katakana(original, _fake_llm)
        candidate, _ = enforce_japanese_invariant(original, katakana_result.text)
        out = normalize_for_tts(candidate)

        assert out == expected, case["name"]
