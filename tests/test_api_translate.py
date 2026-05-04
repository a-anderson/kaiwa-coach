"""Tests for the POST /api/turns/{assistant_turn_id}/translate route."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from kaiwacoach.api.server import create_app
from kaiwacoach.storage.blobs import SessionAudioCache


@pytest.fixture
def mock_orchestrator():
    orc = MagicMock()
    orc.language = "ja"
    orc.get_user_profile.return_value = {
        "user_name": None,
        "language_proficiency": {},
        "translation_language": "English",
    }
    return orc


@pytest.fixture
def mock_audio_cache(tmp_path: Path):
    cache = MagicMock(spec=SessionAudioCache)
    cache.root_dir = tmp_path / "audio"
    cache.root_dir.mkdir()
    return cache


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.close = MagicMock()
    return db


@pytest.fixture
def client(mock_orchestrator, mock_audio_cache, mock_db):
    app = create_app(
        orchestrator=mock_orchestrator,
        audio_cache=mock_audio_cache,
        db=mock_db,
    )
    with TestClient(app) as c:
        yield c


def test_translate_turn_success(client, mock_orchestrator) -> None:
    mock_orchestrator.translate_assistant_turn.return_value = "こんにちは。"
    res = client.post(
        "/api/turns/at-1/translate",
        json={"target_language": "English"},
    )
    assert res.status_code == 200
    assert res.json()["translation"] == "こんにちは。"
    mock_orchestrator.translate_assistant_turn.assert_called_once_with(
        assistant_turn_id="at-1",
        target_language="English",
    )


def test_translate_turn_unknown_id_returns_404(client, mock_orchestrator) -> None:
    mock_orchestrator.translate_assistant_turn.side_effect = ValueError(
        "Unknown assistant_turn_id: bad-id"
    )
    res = client.post("/api/turns/bad-id/translate", json={})
    assert res.status_code == 404
    assert res.json()["detail"] == "Turn not found"


def test_translate_turn_llm_failure_returns_422(client, mock_orchestrator) -> None:
    mock_orchestrator.translate_assistant_turn.side_effect = ValueError(
        "Translation failed: schema validation error"
    )
    res = client.post("/api/turns/at-1/translate", json={})
    assert res.status_code == 422
    assert "Translation failed" in res.json()["detail"]


def test_translate_turn_unsupported_language_returns_422(client, mock_orchestrator) -> None:
    res = client.post(
        "/api/turns/at-1/translate",
        json={"target_language": "Klingon"},
    )
    assert res.status_code == 422
    assert "Unsupported translation language" in res.json()["detail"]
    mock_orchestrator.translate_assistant_turn.assert_not_called()


def test_translate_turn_default_language_is_english(client, mock_orchestrator) -> None:
    mock_orchestrator.translate_assistant_turn.return_value = "Hello."
    res = client.post("/api/turns/at-1/translate", json={})
    assert res.status_code == 200
    mock_orchestrator.translate_assistant_turn.assert_called_once_with(
        assistant_turn_id="at-1",
        target_language="English",
    )


def test_translate_turn_supported_non_english_language(client, mock_orchestrator) -> None:
    mock_orchestrator.translate_assistant_turn.return_value = "Bonjour."
    res = client.post(
        "/api/turns/at-1/translate",
        json={"target_language": "French"},
    )
    assert res.status_code == 200
    assert res.json()["translation"] == "Bonjour."
