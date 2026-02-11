"""Storage round-trip tests for the SQLite schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _load_schema(connection: sqlite3.Connection) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    connection.executescript(schema_sql)


def test_storage_roundtrip_and_cascade() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    connection.execute(
        """
        INSERT INTO conversations (
            id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("conv_1", "Test", "ja", "asr", "llm", "tts", "{}"),
    )

    connection.execute(
        """
        INSERT INTO user_turns (id, conversation_id, input_text)
        VALUES (?, ?, ?)
        """,
        ("user_1", "conv_1", "こんにちは"),
    )

    connection.execute(
        """
        INSERT INTO assistant_turns (id, user_turn_id, conversation_id, reply_text)
        VALUES (?, ?, ?, ?)
        """,
        ("assistant_1", "user_1", "conv_1", "こんにちは。元気ですか？"),
    )

    connection.execute(
        """
        INSERT INTO corrections (id, user_turn_id, corrected_text)
        VALUES (?, ?, ?)
        """,
        ("corr_1", "user_1", "こんにちは"),
    )

    connection.execute(
        """
        INSERT INTO artifacts (id, conversation_id, kind, path)
        VALUES (?, ?, ?, ?)
        """,
        ("artifact_1", "conv_1", "export", "exports/conv_1.zip"),
    )

    connection.execute("DELETE FROM conversations WHERE id = ?", ("conv_1",))

    counts = {
        "user_turns": connection.execute("SELECT COUNT(*) FROM user_turns").fetchone()[0],
        "assistant_turns": connection.execute("SELECT COUNT(*) FROM assistant_turns").fetchone()[0],
        "corrections": connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0],
        "artifacts": connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
    }
    assert counts == {
        "user_turns": 0,
        "assistant_turns": 0,
        "corrections": 0,
        "artifacts": 0,
    }


def test_foreign_key_enforcement() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON;")
    _load_schema(connection)

    try:
        connection.execute(
            """
            INSERT INTO user_turns (id, conversation_id, input_text)
            VALUES (?, ?, ?)
            """,
            ("user_missing", "conv_missing", "こんにちは"),
        )
        connection.commit()
        assert False, "Expected foreign key constraint to fail."
    except sqlite3.IntegrityError:
        pass
