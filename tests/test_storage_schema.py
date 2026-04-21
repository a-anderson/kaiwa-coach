"""Schema validation tests for the SQLite schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kaiwacoach.storage.db import SQLiteWriter


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"


def _load_schema(connection: sqlite3.Connection) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    connection.executescript(schema_sql)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'",
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


def test_schema_version_row_exists() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    row = connection.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    assert row is not None
    assert row[0] == 4


def test_user_profile_table_exists() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    tables = _table_names(connection)
    assert "user_profile" in tables

    columns = _column_names(connection, "user_profile")
    assert "id" in columns
    assert "user_name" in columns
    assert "language_proficiency_json" in columns
    assert "created_at" in columns
    assert "updated_at" in columns


def test_user_profile_default_row_exists() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    row = connection.execute(
        "SELECT id, user_name, language_proficiency_json FROM user_profile WHERE id = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] is None
    assert row[2] == "{}"


def test_conversations_conversation_type_column() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    columns = _column_names(connection, "conversations")
    assert "conversation_type" in columns

    # Default value for new rows must be 'chat'
    connection.execute(
        "INSERT INTO conversations (id, title, language, asr_model_id, llm_model_id, tts_model_id) "
        "VALUES ('c1', 'Test', 'ja', 'asr', 'llm', 'tts')"
    )
    row = connection.execute(
        "SELECT conversation_type FROM conversations WHERE id = 'c1'"
    ).fetchone()
    assert row is not None
    assert row[0] == "chat"


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


_V2_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE schema_version (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO schema_version (id, version) VALUES (1, 2);
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  language TEXT NOT NULL,
  asr_model_id TEXT NOT NULL,
  llm_model_id TEXT NOT NULL,
  tts_model_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  model_metadata_json TEXT
);
CREATE TABLE user_turns (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  input_text TEXT,
  asr_text TEXT,
  asr_meta_json TEXT
);
CREATE TABLE assistant_turns (
  id TEXT PRIMARY KEY,
  user_turn_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  reply_text TEXT NOT NULL,
  llm_meta_json TEXT
);
CREATE TABLE corrections (
  id TEXT PRIMARY KEY,
  user_turn_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  errors_json TEXT,
  corrected_text TEXT,
  native_text TEXT,
  explanation_text TEXT,
  prompt_hash TEXT
);
CREATE TABLE artifacts (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def test_migration_v2_to_v3_preserves_existing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"

    conn = sqlite3.connect(db_path)
    conn.executescript(_V2_SCHEMA_SQL)
    conn.execute(
        "INSERT INTO conversations (id, title, language, asr_model_id, llm_model_id, tts_model_id) "
        "VALUES ('conv-1', 'Test Conversation', 'ja', 'asr', 'llm', 'tts')"
    )
    conn.commit()
    conn.close()

    writer = SQLiteWriter(db_path=db_path, schema_path=_schema_path())
    writer.start()
    try:
        with writer.read_connection() as rconn:
            row = rconn.execute(
                "SELECT id, title, conversation_type FROM conversations WHERE id = 'conv-1'"
            ).fetchone()
            version = rconn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    finally:
        writer.close()

    assert row is not None, "existing conversation row must be preserved after migration"
    assert row[0] == "conv-1"
    assert row[1] == "Test Conversation"
    assert row[2] == "chat", "migrated rows must get the default conversation_type"
    assert version is not None
    assert version[0] == 3


def test_migration_idempotent_on_v3_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"

    writer = SQLiteWriter(db_path=db_path, schema_path=_schema_path())
    writer.start()

    def _insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO conversations (id, title, language, asr_model_id, llm_model_id, tts_model_id) "
            "VALUES ('conv-1', 'Test', 'ja', 'asr', 'llm', 'tts')"
        )
        conn.commit()

    writer.run_write(_insert)
    writer.close()

    writer2 = SQLiteWriter(db_path=db_path, schema_path=_schema_path())
    writer2.start()
    try:
        with writer2.read_connection() as rconn:
            row = rconn.execute(
                "SELECT id, conversation_type FROM conversations WHERE id = 'conv-1'"
            ).fetchone()
    finally:
        writer2.close()

    assert row is not None
    assert row[0] == "conv-1"
    assert row[1] == "chat"
