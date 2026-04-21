"""Render all prompt templates with required variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaiwacoach.prompts.loader import PromptLoader

PROMPT_VARIABLES = {
    "conversation.md": {
        "language": "ja",
        "conversation_history": "<history>",
        "user_text": "こんにちは",
        "user_name": "Ashley",
        "user_level": "N3",
        "user_kanji_level": "N2",
    },
    "detect_and_correct.md": {
        "language": "ja",
        "user_text": "こんにちは",
        "user_level": "N3",
        "user_kanji_level": "N2",
    },
    "explain_and_native.md": {
        "language": "ja",
        "user_text": "こんにちは",
        "corrected_text": "こんにちは。",
        "errors": "- Missing period at end of sentence",
        "user_level": "N3",
    },
    "jp_tts_normalise.md": {
        "original_text": "これはテストです",
        "user_name": "Ashley",
    },
    "romanise_name.md": {
        "name": "田中",
    },
    "monologue_summary.md": {
        "language": "ja",
        "errors": "Wrong particle used",
        "corrected": "私は行きます",
        "explanation": "を should be は",
        "user_level": "N3",
    },
    "repair_json.md": {
        "json_schema": "{\"reply\": \"<text>\"}",
        "raw_output": "not json",
    },
    "summarise_conversation.md": {
        "language": "ja",
        "corrections_text": "[1] Errors: Wrong particle  |  Corrected: 私は行きます",
        "user_level": "N3",
    },
}


@pytest.mark.parametrize("filename", sorted(PROMPT_VARIABLES.keys()))
def test_prompt_renders_with_required_variables(filename: str) -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts"
    loader = PromptLoader(root)
    result = loader.render(filename, PROMPT_VARIABLES[filename])
    assert result.text


def test_conversation_prompt_renders_with_empty_profile_vars() -> None:
    """Empty user_name and user_kanji_level (non-Japanese session) must not raise."""
    root = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts"
    loader = PromptLoader(root)
    result = loader.render(
        "conversation.md",
        {
            "language": "fr",
            "conversation_history": "",
            "user_text": "Bonjour",
            "user_name": "",
            "user_level": "A1",
            "user_kanji_level": "",
        },
    )
    assert result.text


def test_detect_correct_prompt_renders_with_empty_kanji_level() -> None:
    """Empty user_kanji_level (non-Japanese session) must not raise."""
    root = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts"
    loader = PromptLoader(root)
    result = loader.render(
        "detect_and_correct.md",
        {
            "language": "fr",
            "user_text": "Bonjour",
            "user_level": "A1",
            "user_kanji_level": "",
        },
    )
    assert result.text
