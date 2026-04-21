"""Tests for user profile get/set and per-language level helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kaiwacoach.models.json_enforcement import NormalisedName, ParseResult
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.db import SQLiteWriter


def _setup_db(tmp_path: Path) -> SQLiteWriter:
    db_path = tmp_path / "kaiwacoach.sqlite"
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    return db


def _make_orch(db: SQLiteWriter, language: str = "ja") -> ConversationOrchestrator:
    llm = MagicMock()
    # Simulate LLM failure for name normalisation so no MagicMock leaks into SQLite.
    llm.generate_json.return_value = MagicMock(model=None)
    prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
    return ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language=language)


def test_get_profile_returns_defaults_on_fresh_db(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        profile = orch.get_user_profile()
        assert profile["user_name"] is None
        assert profile["language_proficiency"] == {}
    finally:
        db.close()


def test_set_and_get_profile_round_trip(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        orch.set_user_profile(
            user_name="Ashley",
            language_proficiency={"ja": "N3", "fr": "B1"},
        )
        profile = orch.get_user_profile()
        assert profile["user_name"] == "Ashley"
        assert profile["language_proficiency"]["ja"] == "N3"
        assert profile["language_proficiency"]["fr"] == "B1"
    finally:
        db.close()


def test_set_profile_clears_name(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        orch.set_user_profile(user_name="Ashley", language_proficiency={})
        orch.set_user_profile(user_name=None, language_proficiency={})
        profile = orch.get_user_profile()
        assert profile["user_name"] is None
    finally:
        db.close()


def test_user_level_for_defaults_ja(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="ja")
        assert orch._user_level_for("ja") == "N5"
    finally:
        db.close()


def test_user_level_for_defaults_fr(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="fr")
        assert orch._user_level_for("fr") == "A1"
    finally:
        db.close()


def test_user_level_for_returns_stored_value(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="ja")
        orch.set_user_profile(user_name=None, language_proficiency={"ja": "N2"})
        assert orch._user_level_for("ja") == "N2"
    finally:
        db.close()


def test_user_kanji_level_for_returns_empty_for_non_japanese(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="fr")
        assert orch._user_kanji_level_for() == ""
    finally:
        db.close()


def test_user_kanji_level_for_defaults_to_n5(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="ja")
        assert orch._user_kanji_level_for() == "N5"
    finally:
        db.close()


def test_user_kanji_level_for_falls_back_to_ja_level(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="ja")
        orch.set_user_profile(user_name=None, language_proficiency={"ja": "N3"})
        assert orch._user_kanji_level_for() == "N3"
    finally:
        db.close()


def test_user_kanji_level_for_uses_ja_kanji_independently(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db, language="ja")
        orch.set_user_profile(user_name=None, language_proficiency={"ja": "N3", "ja_kanji": "N1"})
        assert orch._user_kanji_level_for() == "N1"
        assert orch._user_level_for("ja") == "N3"
    finally:
        db.close()


def test_user_name_for_prompt_returns_empty_when_null(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        assert orch._user_name_for_prompt("ja") == ""
        assert orch._user_name_for_prompt("fr") == ""
    finally:
        db.close()


def test_user_name_for_prompt_falls_back_to_raw_name_when_derived_absent(tmp_path: Path) -> None:
    # LLM mock returns model=None so derived fields are not stored; raw name is the fallback.
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        orch.set_user_profile(user_name="Ashley", language_proficiency={})
        assert orch._user_name_for_prompt("fr") == "Ashley"
        assert orch._user_name_for_prompt("de") == "Ashley"
    finally:
        db.close()


def test_user_name_for_prompt_uses_katakana_for_latin_name_in_ja_session(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db = _setup_db(tmp_path)
    try:
        # Populate derived fields directly, bypassing the LLM.
        conn = _sqlite3.connect(db._db_path)
        conn.execute(
            "UPDATE user_profile SET user_name = ?, user_name_romanised = ?, user_name_katakana = ? WHERE id = 1",
            ("Ashley", "Ashley", "アシュリー"),
        )
        conn.commit()
        conn.close()
        orch = _make_orch(db, language="ja")
        assert orch._user_name_for_prompt("ja") == "アシュリー"
        assert orch._user_name_for_prompt("fr") == "Ashley"
    finally:
        db.close()


def test_user_name_for_prompt_uses_original_for_japanese_name_in_ja_session(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db = _setup_db(tmp_path)
    try:
        conn = _sqlite3.connect(db._db_path)
        conn.execute(
            "UPDATE user_profile SET user_name = ?, user_name_romanised = ?, user_name_katakana = ? WHERE id = 1",
            ("田中", "Tanaka", "タナカ"),
        )
        conn.commit()
        conn.close()
        orch = _make_orch(db, language="ja")
        # Japanese-script name stays as-is in a Japanese session.
        assert orch._user_name_for_prompt("ja") == "田中"
        assert orch._user_name_for_prompt("fr") == "Tanaka"
    finally:
        db.close()


def test_ensure_name_normalised_computes_missing_forms(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db = _setup_db(tmp_path)
    try:
        # Simulate a name set before the normalise_name feature existed (derived forms NULL).
        conn = _sqlite3.connect(db._db_path)
        conn.execute("UPDATE user_profile SET user_name = ? WHERE id = 1", ("田中",))
        conn.commit()
        conn.close()

        orch = _make_orch(db, language="fr")
        # Configure mock to return a valid NormalisedName on this call.
        orch._llm.generate_json.return_value = ParseResult(
            model=NormalisedName(romanised="Tanaka", katakana="タナカ"),
            raw_json={"romanised": "Tanaka", "katakana": "タナカ"},
            error=None,
            repaired=False,
        )

        profile = orch.get_user_profile()
        updated = orch._ensure_name_normalised(profile)

        assert updated["user_name_romanised"] == "Tanaka"
        assert updated["user_name_katakana"] == "タナカ"
        # Verify forms were persisted so subsequent turns don't re-call the LLM.
        persisted = orch.get_user_profile()
        assert persisted["user_name_romanised"] == "Tanaka"
        assert persisted["user_name_katakana"] == "タナカ"
    finally:
        db.close()


def test_ensure_name_normalised_is_noop_when_forms_present(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db = _setup_db(tmp_path)
    try:
        conn = _sqlite3.connect(db._db_path)
        conn.execute(
            "UPDATE user_profile SET user_name = ?, user_name_romanised = ?, user_name_katakana = ? WHERE id = 1",
            ("田中", "Tanaka", "タナカ"),
        )
        conn.commit()
        conn.close()

        orch = _make_orch(db, language="fr")
        profile = orch.get_user_profile()
        orch._ensure_name_normalised(profile)

        assert orch._llm.generate_json.call_count == 0
    finally:
        db.close()


def test_get_user_profile_handles_corrupt_json(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        import sqlite3
        conn = sqlite3.connect(tmp_path / "kaiwacoach.sqlite")
        conn.execute("UPDATE user_profile SET language_proficiency_json = 'not-json' WHERE id = 1")
        conn.commit()
        conn.close()

        orch = _make_orch(db)
        profile = orch.get_user_profile()
        assert profile["language_proficiency"] == {}
    finally:
        db.close()
