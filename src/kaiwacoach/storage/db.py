"""SQLite access layer with a single-writer queue."""

from __future__ import annotations

import queue
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Optional


@dataclass(frozen=True)
class _WriteTask:
    func: Callable[[sqlite3.Connection], Any]
    result_queue: "queue.Queue[tuple[bool, Any]]"


class SQLiteWriter:
    """Single-writer SQLite access layer with safe concurrent reads."""

    # Keep this allowlist in sync with schema.sql and any future migrations.
    _ALLOWED_UPDATE_COLUMNS: dict[str, set[str]] = {
        "conversations": {
            "title",
            "language",
            "asr_model_id",
            "llm_model_id",
            "tts_model_id",
            "model_metadata_json",
            "updated_at",
        },
        "user_turns": {"input_text", "asr_text", "asr_meta_json", "updated_at"},
        "assistant_turns": {"reply_text", "llm_meta_json", "updated_at"},
        "corrections": {
            "errors_json",
            "corrected_text",
            "native_text",
            "explanation_text",
            "prompt_hash",
            "updated_at",
        },
        "artifacts": {"kind", "path", "meta_json", "updated_at"},
    }

    def __init__(self, db_path: str | Path, schema_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._schema_path = Path(schema_path)
        self._task_queue: "queue.Queue[Optional[_WriteTask]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sqlite-writer", daemon=True)
        self._ready_event = threading.Event()

    def start(self) -> None:
        """Start the writer thread and initialize the schema.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._thread.start()
        self._ready_event.wait()

    def close(self) -> None:
        """Stop the writer thread.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._task_queue.put(None)
        self._thread.join()
        if self._thread.is_alive():
            raise RuntimeError("SQLite writer thread failed to stop.")

    def run_write(self, func: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run a write task in the single-writer thread and return its result.

        Parameters
        ----------
        func : Callable[[sqlite3.Connection], Any]
            Callable that performs work on the writer connection.

        Returns
        -------
        Any
            The result of the write task.

        Raises
        ------
        RuntimeError
            If the writer thread is not running.
        Exception
            Propagates any exception raised by the write task.
        """
        if not self._thread.is_alive():
            raise RuntimeError("SQLite writer is not running. Call start() first.")
        result_queue: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
        self._task_queue.put(_WriteTask(func=func, result_queue=result_queue))
        ok, payload = result_queue.get()
        if ok:
            return payload
        raise payload

    def execute_write(self, sql: str, params: Iterable[Any] | None = None) -> None:
        """Execute a single SQL write statement on the writer thread.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        params : Iterable[Any] | None
            Parameters for the SQL statement.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the writer thread is not running.
        ValueError
            If the statement is an UPDATE (use execute_update instead).
        """
        params = params or ()
        if sql.lstrip().lower().startswith("update "):
            raise ValueError("Raw UPDATE statements are not allowed. Use execute_update().")

        def _exec(conn: sqlite3.Connection) -> None:
            conn.execute(sql, tuple(params))
            conn.commit()

        self.run_write(_exec)

    def executemany_write(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        """Execute a batch SQL write statement on the writer thread.

        Parameters
        ----------
        sql : str
            SQL statement to execute.
        rows : Iterable[Iterable[Any]]
            Iterable of parameter rows.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the writer thread is not running.
        ValueError
            If the statement is an UPDATE (use execute_update instead).
        """
        if sql.lstrip().lower().startswith("update "):
            raise ValueError("Raw UPDATE statements are not allowed. Use execute_update().")

        def _exec(conn: sqlite3.Connection) -> None:
            conn.executemany(sql, list(rows))
            conn.commit()

        self.run_write(_exec)

    def executescript_write(self, sql_script: str) -> None:
        """Execute a SQL script on the writer thread.

        Parameters
        ----------
        sql_script : str
            SQL script to execute.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the writer thread is not running.
        """

        def _exec(conn: sqlite3.Connection) -> None:
            conn.executescript(sql_script)
            conn.commit()

        self.run_write(_exec)

    def execute_update(
        self,
        table: str,
        set_values: dict[str, Any],
        where: dict[str, Any],
    ) -> None:
        """Execute an UPDATE that also refreshes the `updated_at` column.

        Parameters
        ----------
        table : str
            Table name to update.
        set_values : dict[str, Any]
            Column/value pairs to update.
        where : dict[str, Any]
            Column/value pairs to match (equality only).

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the writer thread is not running.
        ValueError
            If the table or columns are not allowed.
        """
        allowed_columns = self._ALLOWED_UPDATE_COLUMNS.get(table)
        if allowed_columns is None:
            raise ValueError(f"Updates are not allowed for table: {table}")
        if not set_values:
            raise ValueError("set_values must not be empty.")
        if not where:
            raise ValueError("where must not be empty.")

        for column in set_values:
            if column not in allowed_columns or column == "updated_at":
                raise ValueError(f"Updates to column '{column}' are not allowed for table '{table}'.")
        for column in where:
            if column not in allowed_columns and column != "id" and column != "conversation_id" and column != "user_turn_id":
                raise ValueError(f"Where clause column '{column}' is not allowed for table '{table}'.")

        set_parts = [f"{column} = ?" for column in set_values]
        set_parts.append("updated_at = datetime('now')")
        where_parts = [f"{column} = ?" for column in where]
        sql = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        params = list(set_values.values()) + list(where.values())

        def _exec(conn: sqlite3.Connection) -> None:
            conn.execute(sql, tuple(params))
            conn.commit()

        self.run_write(_exec)

    @contextmanager
    def read_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a short-lived read connection for concurrent reads.

        Parameters
        ----------
        None

        Returns
        -------
        Generator[sqlite3.Connection, None, None]
            Context-managed read-only connection.
        """
        connection = sqlite3.connect(self._db_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON;")
            yield connection
        finally:
            connection.close()

    def _run(self) -> None:
        connection = sqlite3.connect(self._db_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON;")
            self._apply_schema(connection)
            self._ready_event.set()
            while not self._stop_event.is_set():
                task = self._task_queue.get()
                if task is None:
                    break
                try:
                    result = task.func(connection)
                except Exception as exc:  # noqa: BLE001
                    task.result_queue.put((False, exc))
                else:
                    task.result_queue.put((True, result))
        finally:
            connection.close()

    def _apply_schema(self, connection: sqlite3.Connection) -> None:
        """Apply the schema to the writer connection.

        Parameters
        ----------
        connection : sqlite3.Connection
            The writer connection to initialize.

        Returns
        -------
        None
        """
        schema_sql = self._schema_path.read_text(encoding="utf-8")
        connection.executescript(schema_sql)
        connection.commit()
