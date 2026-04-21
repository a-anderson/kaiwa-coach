"""Tests for ConversationOrchestrator.summarise_conversation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaiwacoach.models.json_enforcement import ConversationSummaryResult, ParseResult
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.db import SQLiteWriter


def _setup_db(tmp_path: Path) -> SQLiteWriter:
    db_path = tmp_path / "kaiwacoach.sqlite"
    schema_path = (
        Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    )
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    return db


def _make_summary_result(
    patterns=None,
    areas=None,
    notes="Keep practising.",
) -> ParseResult:
    if patterns is None:
        patterns = ["Particle errors"]
    if areas is None:
        areas = ["Particle usage"]
    model = ConversationSummaryResult(
        top_error_patterns=patterns,
        priority_areas=areas,
        overall_notes=notes,
    )
    return ParseResult(model=model, raw_json={}, error=None, repaired=False)


def _make_llm(summary_result: ParseResult | None = None) -> MagicMock:
    llm = MagicMock()
    llm.model_id = "mock-llm"
    llm.generate_json.return_value = summary_result or _make_summary_result()
    return llm


def _make_orch(db: SQLiteWriter, llm=None, language: str = "ja") -> ConversationOrchestrator:
    prompts = PromptLoader(
        Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts"
    )
    return ConversationOrchestrator(
        db=db,
        llm=llm or _make_llm(),
        prompt_loader=prompts,
        language=language,
    )


def _insert_correction(db: SQLiteWriter, conv_id: str, errors: list[str], corrected: str) -> None:
    """Insert a user_turn + correction row for testing."""
    import uuid

    turn_id = str(uuid.uuid4())
    correction_id = str(uuid.uuid4())
    errors_json = json.dumps(errors)

    def _write(conn) -> None:
        conn.execute(
            "INSERT INTO user_turns (id, conversation_id, input_text, created_at) VALUES (?, ?, ?, datetime('now'))",
            (turn_id, conv_id, "test input"),
        )
        conn.execute(
            "INSERT INTO corrections (id, user_turn_id, errors_json, corrected_text, native_text, explanation_text, created_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (correction_id, turn_id, errors_json, corrected, "", ""),
        )
        conn.commit()

    db.run_write(_write)


# ── get_corrections_for_conversation ─────────────────────────────────────────


def test_get_corrections_empty_when_no_rows(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_conversation()
        result = orch.get_corrections_for_conversation(conv_id)
        assert result == []
    finally:
        db.close()


def test_get_corrections_returns_all_rows(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, ["Error A"], "corrected A")
        _insert_correction(db, conv_id, ["Error B"], "corrected B")
        rows = orch.get_corrections_for_conversation(conv_id)
        assert len(rows) == 2
        corrected_texts = {r["corrected_text"] for r in rows}
        assert corrected_texts == {"corrected A", "corrected B"}
    finally:
        db.close()


def test_get_corrections_caps_at_20(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_conversation()
        for i in range(25):
            _insert_correction(db, conv_id, [f"Error {i}"], f"corrected {i}")
        rows = orch.get_corrections_for_conversation(conv_id)
        assert len(rows) == 20
    finally:
        db.close()


# ── summarise_conversation ────────────────────────────────────────────────────


def test_summarise_empty_conversation_returns_no_conversation_message(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_conversation()
        result = orch.summarise_conversation(conv_id)
        assert result is not None
        assert "No conversation" in result["overall_notes"]
        assert result["top_error_patterns"] == []
        assert result["priority_areas"] == []
    finally:
        db.close()


def test_summarise_turns_with_no_errors_returns_no_corrections_message(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, [], "corrected text")
        result = orch.summarise_conversation(conv_id)
        assert result is not None
        assert "No corrections" in result["overall_notes"]
        assert result["top_error_patterns"] == []
        assert result["priority_areas"] == []
    finally:
        db.close()


def test_summarise_returns_all_three_fields(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, ["Wrong particle"], "corrected")
        result = orch.summarise_conversation(conv_id)
        assert result is not None
        assert "top_error_patterns" in result
        assert "priority_areas" in result
        assert "overall_notes" in result
    finally:
        db.close()


def test_summarise_multi_error_row_joined_with_semicolon(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = _make_llm()
        orch = _make_orch(db, llm=llm)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, ["Error one", "Error two"], "corrected")
        orch.summarise_conversation(conv_id)

        call_args = llm.generate_json.call_args
        prompt_text = call_args[0][0] if call_args[0] else call_args[1]["prompt"]
        assert "Error one; Error two" in prompt_text
    finally:
        db.close()


def test_summarise_prompt_contains_corrections_text(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = _make_llm()
        orch = _make_orch(db, llm=llm)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, ["Particle error"], "私は行きます")
        orch.summarise_conversation(conv_id)

        call_args = llm.generate_json.call_args
        prompt_text = call_args[0][0] if call_args[0] else call_args[1]["prompt"]
        assert "[1] Errors:" in prompt_text
        assert "Particle error" in prompt_text
    finally:
        db.close()


def test_summarise_skips_rows_with_empty_error_list(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = _make_llm()
        orch = _make_orch(db, llm=llm)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, [], "no errors here")
        _insert_correction(db, conv_id, ["Real error"], "corrected")
        orch.summarise_conversation(conv_id)

        call_args = llm.generate_json.call_args
        prompt_text = call_args[0][0] if call_args[0] else call_args[1]["prompt"]
        assert "[1] Errors:" in prompt_text
        assert "[2]" not in prompt_text
        assert "no errors here" not in prompt_text
    finally:
        db.close()


def test_summarise_llm_failure_returns_empty_fallback(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        failed_result = ParseResult(model=None, raw_json=None, error="parse error", repaired=False)
        llm = _make_llm(summary_result=failed_result)
        orch = _make_orch(db, llm=llm)
        conv_id = orch.create_conversation()
        _insert_correction(db, conv_id, ["Error"], "corrected")
        result = orch.summarise_conversation(conv_id)
        assert result is not None
        assert isinstance(result["top_error_patterns"], list)
        assert isinstance(result["priority_areas"], list)
        assert isinstance(result["overall_notes"], str)
    finally:
        db.close()
