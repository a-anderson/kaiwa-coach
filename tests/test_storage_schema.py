"""Schema validation tests for the SQLite schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _load_schema(connection: sqlite3.Connection) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    connection.executescript(schema_sql)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'",
    ).fetchall()
    return {row[0] for row in rows}


def _trigger_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger'",
    ).fetchall()
    return {row[0] for row in rows}


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def test_schema_tables_and_columns_exist() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    tables = _table_names(connection)
    assert "schema_version" in tables
    assert "conversations" in tables
    assert "user_turns" in tables
    assert "assistant_turns" in tables
    assert "corrections" in tables
    assert "artifacts" in tables

    assistant_columns = _column_names(connection, "assistant_turns")
    assert "conversation_id" in assistant_columns
    assert "reply_audio_path" not in assistant_columns

    user_turns_columns = _column_names(connection, "user_turns")
    assert "input_audio_path" not in user_turns_columns


def test_schema_triggers_exist() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    triggers = _trigger_names(connection)
    assert "trg_conversations_updated_at" in triggers
    assert "trg_user_turns_updated_at" in triggers
    assert "trg_assistant_turns_updated_at" in triggers
    assert "trg_corrections_updated_at" in triggers
    assert "trg_artifacts_updated_at" in triggers


def test_schema_version_row_exists() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    row = connection.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    assert row is not None
    assert row[0] == 1


def test_schema_indexes_exist() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    indexes = {row[0] for row in rows}
    assert "idx_user_turns_conversation_id" in indexes
    assert "idx_assistant_turns_user_turn_id" in indexes
    assert "idx_assistant_turns_conversation_id" in indexes
    assert "idx_corrections_user_turn_id" in indexes
    assert "idx_artifacts_conversation_id" in indexes
