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
    },
    "detect_errors.md": {
        "language": "ja",
        "user_text": "こんにちは",
    },
    "correct_sentence.md": {
        "language": "ja",
        "user_text": "こんにちは",
    },
    "native_rewrite.md": {
        "language": "ja",
        "user_text": "こんにちは",
    },
    "explain.md": {
        "language": "ja",
        "user_text": "こんにちは",
        "corrected_text": "こんにちは。",
    },
    "jp_tts_normalise.md": {
        "original_text": "これはテストです",
    },
    "repair_json.md": {
        "json_schema": "{\"reply\": \"<text>\"}",
        "raw_output": "not json",
    },
}


@pytest.mark.parametrize("filename", sorted(PROMPT_VARIABLES.keys()))
def test_prompt_renders_with_required_variables(filename: str) -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts"
    loader = PromptLoader(root)
    result = loader.render(filename, PROMPT_VARIABLES[filename])
    assert result.text
