"""Tests for monologue orchestrator methods."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaiwacoach.models.json_enforcement import (
    DetectAndCorrect,
    ExplainAndNative,
    MonologueSummary,
    ParseResult,
)
from kaiwacoach.orchestrator import ConversationOrchestrator, MonologueTurnResult
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.db import SQLiteWriter


def _setup_db(tmp_path: Path) -> SQLiteWriter:
    db_path = tmp_path / "kaiwacoach.sqlite"
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    return db


def _make_detect_result(errors=None, corrected="corrected text") -> ParseResult:
    if errors is None:
        errors = ["Wrong particle"]
    model = DetectAndCorrect(errors=errors, corrected=corrected)
    return ParseResult(model=model, raw_json={}, error=None, repaired=False)


def _make_explain_result(explanation="explanation here", native="native rewrite") -> ParseResult:
    model = ExplainAndNative(explanation=explanation, native=native)
    return ParseResult(model=model, raw_json={}, error=None, repaired=False)


def _make_summary_result(areas=None, assessment="good work") -> ParseResult:
    if areas is None:
        areas = ["particle usage"]
    model = MonologueSummary(improvement_areas=areas, overall_assessment=assessment)
    return ParseResult(model=model, raw_json={}, error=None, repaired=False)


def _make_llm():
    llm = MagicMock()
    llm.model_id = "mock-llm"  # truthy string satisfies _resolve_model_id without falling through
    # detect_and_correct and monologue_summary use generate_json
    # explain_and_native uses generate() via _generate_with_repair
    llm.generate_json.side_effect = [
        _make_detect_result(),
        _make_summary_result(),
    ]
    llm.generate.return_value = MagicMock(text='{"explanation": "exp", "native": "native"}')
    return llm


def _make_orch(db: SQLiteWriter, language: str = "ja") -> ConversationOrchestrator:
    prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
    return ConversationOrchestrator(db=db, llm=_make_llm(), prompt_loader=prompts, language=language)


# ── create_monologue_conversation ─────────────────────────────────────────────


def test_create_monologue_conversation_returns_id(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        assert isinstance(conv_id, str)
        assert len(conv_id) > 0
    finally:
        db.close()


def test_create_monologue_conversation_persists_type(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT conversation_type FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        assert row is not None
        assert row[0] == "monologue"
    finally:
        db.close()


def test_create_monologue_conversation_title_is_null(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        with db.read_connection() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        assert row is not None
        assert row[0] is None
    finally:
        db.close()


def test_process_monologue_turn_sets_auto_title(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        orch.process_monologue_turn(conversation_id=conv_id, text="私は昨日学校に行きました。")
        with db.read_connection() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        assert row is not None
        assert row[0] == "私は昨日学校に行きました。"
    finally:
        db.close()


def test_process_monologue_turn_truncates_long_title(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        long_text = "あ" * 60
        orch.process_monologue_turn(conversation_id=conv_id, text=long_text)
        with db.read_connection() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        assert row is not None
        assert len(row[0]) <= 50
        assert row[0].endswith("…")
    finally:
        db.close()


# ── list_conversations with conversation_type filter ─────────────────────────


def test_list_conversations_type_filter_monologue(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        chat_id = orch.create_conversation()
        mono_id = orch.create_monologue_conversation()

        monologue_list = orch.list_conversations(conversation_type="monologue")
        ids = [c["id"] for c in monologue_list]
        assert mono_id in ids
        assert chat_id not in ids
    finally:
        db.close()


def test_list_conversations_type_filter_chat(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        chat_id = orch.create_conversation()
        mono_id = orch.create_monologue_conversation()

        chat_list = orch.list_conversations(conversation_type="chat")
        ids = [c["id"] for c in chat_list]
        assert chat_id in ids
        assert mono_id not in ids
    finally:
        db.close()


def test_list_conversations_no_filter_returns_all(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        chat_id = orch.create_conversation()
        mono_id = orch.create_monologue_conversation()

        all_convos = orch.list_conversations()
        ids = [c["id"] for c in all_convos]
        assert chat_id in ids
        assert mono_id in ids
    finally:
        db.close()


def test_list_conversations_includes_conversation_type(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        mono_id = orch.create_monologue_conversation()
        convos = orch.list_conversations()
        mono = next(c for c in convos if c["id"] == mono_id)
        assert mono["conversation_type"] == "monologue"
    finally:
        db.close()


# ── process_monologue_turn (text input) ──────────────────────────────────────


def test_process_monologue_turn_text_creates_user_turn(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        result = orch.process_monologue_turn(
            conversation_id=conv_id, text="私は昨日学校に行きました。"
        )
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT id, input_text FROM user_turns WHERE conversation_id = ?", (conv_id,)
            ).fetchone()
        assert row is not None
        assert row[0] == result.user_turn_id
        assert row[1] == "私は昨日学校に行きました。"
    finally:
        db.close()


def test_process_monologue_turn_text_no_assistant_turn(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        result = orch.process_monologue_turn(conversation_id=conv_id, text="テスト文章です。")
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT id FROM assistant_turns WHERE conversation_id = ?", (conv_id,)
            ).fetchone()
        assert row is None
    finally:
        db.close()


def test_process_monologue_turn_text_creates_correction(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        result = orch.process_monologue_turn(conversation_id=conv_id, text="テスト")
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT id FROM corrections WHERE user_turn_id = ?", (result.user_turn_id,)
            ).fetchone()
        assert row is not None
    finally:
        db.close()


def test_process_monologue_turn_returns_result_fields(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        result = orch.process_monologue_turn(
            conversation_id=conv_id, text="テスト"
        )
        assert isinstance(result, MonologueTurnResult)
        assert result.conversation_id == conv_id
        assert isinstance(result.user_turn_id, str)
        assert result.input_text == "テスト"
        assert result.asr_text is None
        assert result.asr_meta is None
        assert "errors" in result.corrections
        assert "corrected" in result.corrections
        assert "native" in result.corrections
        assert "explanation" in result.corrections
        assert "improvement_areas" in result.summary
        assert "overall_assessment" in result.summary
    finally:
        db.close()


def test_process_monologue_turn_text_emits_stage_events(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        stages: list[tuple[str, str]] = []

        def on_stage(stage, status, data):
            stages.append((stage, status))

        orch.process_monologue_turn(conversation_id=conv_id, text="テスト", on_stage=on_stage)
        assert ("corrections", "running") in stages
        assert ("corrections", "complete") in stages
        assert ("summary", "running") in stages
        assert ("summary", "complete") in stages
        # No asr stage for text input
        assert ("asr", "running") not in stages
    finally:
        db.close()


# ── process_monologue_turn validation ────────────────────────────────────────


def test_process_monologue_turn_requires_text_or_audio(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        with pytest.raises(ValueError, match="Either text or pcm_bytes must be provided"):
            orch.process_monologue_turn(conversation_id=conv_id)
    finally:
        db.close()


def test_process_monologue_turn_audio_requires_meta(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_monologue_conversation()
        with pytest.raises(ValueError, match="audio_meta is required"):
            orch.process_monologue_turn(
                conversation_id=conv_id, pcm_bytes=b"\x00\x01", audio_meta=None
            )
    finally:
        db.close()
