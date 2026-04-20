"""Tests for user profile get/set and per-language level helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
        assert orch._user_name_for_prompt() == ""
    finally:
        db.close()


def test_user_name_for_prompt_returns_name(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        orch = _make_orch(db)
        orch.set_user_profile(user_name="Ashley", language_proficiency={})
        assert orch._user_name_for_prompt() == "Ashley"
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
