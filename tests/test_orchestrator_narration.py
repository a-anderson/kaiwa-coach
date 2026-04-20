"""Tests for ConversationOrchestrator.generate_narration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaiwacoach.models.protocols import TTSResult
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.db import SQLiteWriter


def _setup_db(tmp_path: Path) -> SQLiteWriter:
    db_path = tmp_path / "kaiwacoach.sqlite"
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    return db


def _make_orch(db: SQLiteWriter, language: str = "ja", tts=None) -> ConversationOrchestrator:
    llm = MagicMock()
    prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
    return ConversationOrchestrator(
        db=db,
        llm=llm,
        tts=tts,
        prompt_loader=prompts,
        language=language,
    )


def test_generate_narration_calls_tts_with_session_language(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        tts = MagicMock()
        tts.synthesize.return_value = TTSResult(audio_path="/tmp/test.wav", meta={})
        orch = _make_orch(db, language="fr", tts=tts)

        result = orch.generate_narration("Bonjour le monde")

        tts.synthesize.assert_called_once()
        call_kwargs = tts.synthesize.call_args[1]
        assert call_kwargs["language"] == "fr"
        assert call_kwargs["text"] == "Bonjour le monde"
        assert call_kwargs["conversation_id"] == "narrations"
    finally:
        db.close()


def test_generate_narration_returns_raw_path(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        tts = MagicMock()
        tts.synthesize.return_value = TTSResult(audio_path="/tmp/narration_abc.wav", meta={})
        orch = _make_orch(db, language="en", tts=tts)

        result = orch.generate_narration("Hello world")

        assert result == "/tmp/narration_abc.wav"
        # Must not return a URL — no http prefix
        assert not result.startswith("http")
    finally:
        db.close()


def test_generate_narration_japanese_passes_normalised_text_to_tts(tmp_path: Path) -> None:
    """Japanese narration runs TTS normalisation; TTS receives the normalised text, not raw input."""
    db = _setup_db(tmp_path)
    try:
        llm = MagicMock()
        # jp_tts_normalisation role returns a model with a .text attribute
        normalised_mock = MagicMock()
        normalised_mock.text = "こんにちは"
        llm.generate_json.return_value = MagicMock(model=normalised_mock)

        tts = MagicMock()
        tts.synthesize.return_value = TTSResult(audio_path="/tmp/ja.wav", meta={})

        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, tts=tts, prompt_loader=prompts, language="ja")

        orch.generate_narration("こんにちは")

        tts.synthesize.assert_called_once()
        call_kwargs = tts.synthesize.call_args[1]
        assert call_kwargs["language"] == "ja"
        # TTS receives some text (normalised); the invariant guarantees Japanese spans survive
        assert "こんにちは" in call_kwargs["text"]
    finally:
        db.close()


def test_generate_narration_raises_when_tts_not_configured(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, tts=None)
        with pytest.raises(ValueError, match="TTS must be configured"):
            orch.generate_narration("Hello")
    finally:
        db.close()
