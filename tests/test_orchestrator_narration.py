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


def test_generate_narration_raises_when_tts_not_configured(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, tts=None)
        with pytest.raises(ValueError, match="TTS must be configured"):
            orch.generate_narration("Hello")
    finally:
        db.close()
