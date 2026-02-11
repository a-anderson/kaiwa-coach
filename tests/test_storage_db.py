"""Tests for the SQLiteWriter single-writer queue implementation."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import threading
import pytest

from kaiwacoach.storage.db import SQLiteWriter


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"


@contextmanager
def _writer(db_path: Path) -> Generator[SQLiteWriter, None, None]:
    writer = SQLiteWriter(db_path=db_path, schema_path=_schema_path())
    writer.start()
    try:
        yield writer
    finally:
        writer.close()


def test_run_write_requires_start(tmp_path: Path) -> None:
    """Ensure writes cannot run before the writer thread starts."""
    writer = SQLiteWriter(db_path=tmp_path / "db.sqlite", schema_path=_schema_path())
    with pytest.raises(RuntimeError):
        writer.run_write(lambda conn: conn.execute("SELECT 1"))


def test_start_applies_schema(tmp_path: Path) -> None:
    """Starting the writer should create the expected schema tables."""
    with _writer(tmp_path / "db.sqlite") as writer:
        with writer.read_connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'conversations'"
            ).fetchall()
            assert rows == [("conversations",)]


def test_run_write_inserts_and_returns_value(tmp_path: Path) -> None:
    """run_write should execute on the writer connection and return results."""
    with _writer(tmp_path / "db.sqlite") as writer:
        def _insert(conn: sqlite3.Connection) -> int:
            conn.execute(
                """
                INSERT INTO conversations (
                    id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("conv_1", "Test", "ja", "asr", "llm", "tts", "{}"),
            )
            conn.commit()
            return conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

        count = writer.run_write(_insert)
        assert count == 1


def test_execute_write_inserts_row(tmp_path: Path) -> None:
    """execute_write should run a single statement on the writer thread."""
    with _writer(tmp_path / "db.sqlite") as writer:
        writer.execute_write(
            """
            INSERT INTO conversations (
                id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv_2", "Title", "fr", "asr", "llm", "tts", "{}"),
        )

        with writer.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert count == 1


def test_executemany_write_inserts_rows(tmp_path: Path) -> None:
    """executemany_write should insert multiple rows in a single batch."""
    with _writer(tmp_path / "db.sqlite") as writer:
        writer.execute_write(
            """
            INSERT INTO conversations (
                id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv_3", "Batch", "ja", "asr", "llm", "tts", "{}"),
        )

        writer.executemany_write(
            """
            INSERT INTO artifacts (id, conversation_id, kind, path)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("artifact_1", "conv_3", "export", "exports/conv_3.zip"),
                ("artifact_2", "conv_3", "export", "exports/conv_3_2.zip"),
            ],
        )

        with writer.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            assert count == 2


def test_executescript_write_runs_script(tmp_path: Path) -> None:
    """executescript_write should execute a multi-statement script."""
    with _writer(tmp_path / "db.sqlite") as writer:
        writer.executescript_write(
            """
            INSERT INTO conversations (
                id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
            )
            VALUES ('conv_4', 'Script', 'ja', 'asr', 'llm', 'tts', '{}');
            INSERT INTO user_turns (id, conversation_id, input_text)
            VALUES ('user_1', 'conv_4', 'こんにちは');
            """
        )

        with writer.read_connection() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            user_count = conn.execute("SELECT COUNT(*) FROM user_turns").fetchone()[0]
            assert conv_count == 1
            assert user_count == 1


def test_executescript_write_rejects_parameters(tmp_path: Path) -> None:
    """executescript_write should fail when parameter placeholders are used."""
    with _writer(tmp_path / "db.sqlite") as writer:
        with pytest.raises(sqlite3.IntegrityError):
            writer.executescript_write(
                """
                INSERT INTO conversations (
                    id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """
            )


def test_read_connection_sees_committed_writes(tmp_path: Path) -> None:
    """Read connections should observe committed writes from the writer thread."""
    with _writer(tmp_path / "db.sqlite") as writer:
        writer.execute_write(
            """
            INSERT INTO conversations (
                id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv_5", "Read", "fr", "asr", "llm", "tts", "{}"),
        )

        with writer.read_connection() as conn:
            row = conn.execute("SELECT id FROM conversations WHERE id = 'conv_5'").fetchone()
            assert row == ("conv_5",)


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Closing the writer multiple times should not raise."""
    writer = SQLiteWriter(db_path=tmp_path / "db.sqlite", schema_path=_schema_path())
    writer.start()
    writer.close()
    writer.close()


def test_execute_update_refreshes_updated_at(tmp_path: Path) -> None:
    """execute_update should refresh updated_at without triggers."""
    with _writer(tmp_path / "db.sqlite") as writer:
        writer.execute_write(
            """
            INSERT INTO conversations (
                id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv_update", "Before", "ja", "asr", "llm", "tts", "{}", "2000-01-01 00:00:00"),
        )

        writer.execute_update(
            table="conversations",
            set_values={"title": "After"},
            where={"id": "conv_update"},
        )

        with writer.read_connection() as conn:
            row = conn.execute(
                "SELECT title, updated_at FROM conversations WHERE id = ?",
                ("conv_update",),
            ).fetchone()
            assert row[0] == "After"
            assert row[1] != "2000-01-01 00:00:00"


def test_execute_write_rejects_update_sql(tmp_path: Path) -> None:
    """execute_write should reject raw UPDATE statements."""
    with _writer(tmp_path / "db.sqlite") as writer:
        with pytest.raises(ValueError, match="Raw UPDATE statements are not allowed"):
            writer.execute_write("UPDATE conversations SET title = 'X' WHERE id = '1'")


def test_execute_update_validates_table_and_columns(tmp_path: Path) -> None:
    """execute_update should validate allowed tables and columns."""
    with _writer(tmp_path / "db.sqlite") as writer:
        with pytest.raises(ValueError, match="Updates are not allowed for table"):
            writer.execute_update(
                table="not_a_table",
                set_values={"title": "X"},
                where={"id": "1"},
            )

        with pytest.raises(ValueError, match="set_values must not be empty"):
            writer.execute_update(
                table="conversations",
                set_values={},
                where={"id": "1"},
            )

        with pytest.raises(ValueError, match="where must not be empty"):
            writer.execute_update(
                table="conversations",
                set_values={"title": "X"},
                where={},
            )

        with pytest.raises(ValueError, match="Updates to column"):
            writer.execute_update(
                table="conversations",
                set_values={"created_at": "X"},
                where={"id": "1"},
            )


def test_run_write_propagates_exceptions(tmp_path: Path) -> None:
    """run_write should propagate exceptions raised by tasks."""
    with _writer(tmp_path / "db.sqlite") as writer:
        def _boom(_: sqlite3.Connection) -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            writer.run_write(_boom)


def test_concurrent_writes_are_serialized(tmp_path: Path) -> None:
    """Concurrent writes from multiple threads should serialize correctly."""
    with _writer(tmp_path / "db.sqlite") as writer:
        writer.execute_write(
            """
            INSERT INTO conversations (
                id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv_threads", "Threads", "ja", "asr", "llm", "tts", "{}"),
        )

        def _insert_artifact(index: int) -> None:
            writer.execute_write(
                """
                INSERT INTO artifacts (id, conversation_id, kind, path)
                VALUES (?, ?, ?, ?)
                """,
                (f"artifact_{index}", "conv_threads", "export", f"exports/{index}.zip"),
            )

        threads = [
            threading.Thread(target=_insert_artifact, args=(i,), daemon=True)
            for i in range(10)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with writer.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            assert count == 10
