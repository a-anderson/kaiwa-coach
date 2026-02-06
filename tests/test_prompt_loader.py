"""Tests for prompt loading and rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kaiwacoach.prompts.loader import PromptLoader


def test_render_replaces_variables(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello {name}", encoding="utf-8")

    loader = PromptLoader(tmp_path)
    result = loader.render("prompt.md", {"name": "Kaiwa"})

    assert result.text == "Hello Kaiwa"
    assert result.sha256 == hashlib.sha256(result.text.encode("utf-8")).hexdigest()
    assert result.path == prompt_path


def test_missing_variable_raises(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello {name}", encoding="utf-8")

    loader = PromptLoader(tmp_path)
    with pytest.raises(KeyError, match="Missing prompt variable: name"):
        loader.render("prompt.md", {})


def test_missing_prompt_file_raises(tmp_path: Path) -> None:
    loader = PromptLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.render("missing.md", {})


def test_brace_escaping_is_preserved(tmp_path: Path) -> None:
    """Double-brace JSON examples should render as single braces."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Schema:\\n{{\"reply\": \"{value}\"}}", encoding="utf-8")

    loader = PromptLoader(tmp_path)
    result = loader.render("prompt.md", {"value": "ok"})

    assert result.text.strip() == 'Schema:\\n{\"reply\": \"ok\"}'


def test_prompt_hash_changes_with_variables(tmp_path: Path) -> None:
    """Rendered SHA256 should change when variables change."""
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Hello {name}", encoding="utf-8")

    loader = PromptLoader(tmp_path)
    result_a = loader.render("prompt.md", {"name": "A"})
    result_b = loader.render("prompt.md", {"name": "B"})

    assert result_a.sha256 != result_b.sha256
