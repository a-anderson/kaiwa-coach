"""Tests for the conversation API routes."""

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
    orc.list_conversations.return_value = [
        {
            "id": "conv1",
            "title": "Test Conversation",
            "language": "ja",
            "updated_at": "2026-03-17T10:00:00",
            "preview_text": "こんにちは",
        }
    ]
    orc.get_conversation.return_value = {
        "id": "conv1",
        "title": "Test Conversation",
        "language": "ja",
        "created_at": "2026-03-17T10:00:00",
        "updated_at": "2026-03-17T10:05:00",
        "turns": [],
    }
    orc.get_latest_corrections.return_value = {
        "errors": [],
        "corrected": "",
        "native": "",
        "explanation": "",
    }
    orc.create_conversation.return_value = "conv1"
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


# ── Conversation list ─────────────────────────────────────────────────────────

def test_list_conversations_returns_200(client, mock_orchestrator):
    response = client.get("/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["id"] == "conv1"
    assert data[0]["language"] == "ja"


def test_list_conversations_empty(client, mock_orchestrator):
    mock_orchestrator.list_conversations.return_value = []
    response = client.get("/api/conversations")
    assert response.status_code == 200
    assert response.json() == []


# ── Conversation detail ───────────────────────────────────────────────────────

def test_get_conversation_returns_200(client, mock_orchestrator):
    response = client.get("/api/conversations/conv1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "conv1"
    assert data["language"] == "ja"
    assert data["turns"] == []


def test_get_conversation_includes_corrections(client, mock_orchestrator):
    """Corrections must be fetched and embedded per turn."""
    mock_orchestrator.get_conversation.return_value = {
        "id": "conv1",
        "title": "Test",
        "language": "ja",
        "created_at": "2026-03-17T10:00:00",
        "updated_at": "2026-03-17T10:05:00",
        "turns": [
            {
                "user_turn_id": "ut1",
                "assistant_turn_id": "at1",
                "input_text": "ありがとう",
                "asr_text": None,
                "reply_text": "どういたしまして！",
            }
        ],
    }
    mock_orchestrator.get_latest_corrections.return_value = {
        "errors": ["Missing particle"],
        "corrected": "ありがとうございます",
        "native": "ありがとうございます",
        "explanation": "More polite form",
    }

    response = client.get("/api/conversations/conv1")
    assert response.status_code == 200
    turn = response.json()["turns"][0]
    assert turn["user_text"] == "ありがとう"
    assert turn["correction"]["corrected"] == "ありがとうございます"
    assert turn["correction"]["errors"] == ["Missing particle"]


def test_get_conversation_no_corrections_when_empty(client, mock_orchestrator):
    """When corrections are all empty strings, correction should be null."""
    mock_orchestrator.get_conversation.return_value = {
        "id": "conv1",
        "title": "Test",
        "language": "ja",
        "created_at": "2026-03-17T10:00:00",
        "updated_at": "2026-03-17T10:05:00",
        "turns": [
            {
                "user_turn_id": "ut1",
                "assistant_turn_id": "at1",
                "input_text": "hello",
                "asr_text": None,
                "reply_text": "hi",
            }
        ],
    }
    mock_orchestrator.get_latest_corrections.return_value = {
        "errors": [],
        "corrected": "",
        "native": "",
        "explanation": "",
    }

    response = client.get("/api/conversations/conv1")
    turn = response.json()["turns"][0]
    assert turn["correction"] is None


def test_get_conversation_not_found(client, mock_orchestrator):
    mock_orchestrator.get_conversation.side_effect = ValueError("not found")
    response = client.get("/api/conversations/missing")
    assert response.status_code == 404


# ── Create conversation ───────────────────────────────────────────────────────

def test_create_conversation_returns_201(client, mock_orchestrator):
    response = client.post("/api/conversations", json={})
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "conv1"
    mock_orchestrator.create_conversation.assert_called_once()


def test_create_conversation_with_language_calls_set_language(client, mock_orchestrator):
    client.post("/api/conversations", json={"language": "fr"})
    mock_orchestrator.set_language.assert_called_once_with("fr")


def test_create_conversation_with_title(client, mock_orchestrator):
    client.post("/api/conversations", json={"title": "My Session"})
    mock_orchestrator.create_conversation.assert_called_once_with(title="My Session")


# ── Delete conversation ───────────────────────────────────────────────────────

def test_delete_conversation_returns_204(client, mock_orchestrator):
    response = client.delete("/api/conversations/conv1")
    assert response.status_code == 204
    mock_orchestrator.delete_conversation.assert_called_once_with("conv1")


def test_delete_all_conversations_returns_204(client, mock_orchestrator):
    response = client.delete("/api/conversations")
    assert response.status_code == 204
    mock_orchestrator.delete_all_conversations.assert_called_once()


# ── Session / settings ────────────────────────────────────────────────────────

def test_get_settings_returns_language(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.json()["language"] == "ja"


def test_set_language_returns_204(client, mock_orchestrator):
    response = client.post("/api/session/language", json={"language": "fr"})
    assert response.status_code == 204
    mock_orchestrator.set_language.assert_called_once_with("fr")


def test_reset_session_returns_204(client, mock_orchestrator):
    response = client.post("/api/session/reset")
    assert response.status_code == 204
    mock_orchestrator.reset_session.assert_called_once()


# ── Audio serving ─────────────────────────────────────────────────────────────

def test_serve_audio_not_found(client):
    response = client.get("/api/audio/nonexistent/file.wav")
    assert response.status_code == 404


def test_serve_audio_path_traversal_rejected(client):
    response = client.get("/api/audio/../../../etc/passwd")
    assert response.status_code in (400, 404)


def test_serve_audio_returns_wav(client, mock_audio_cache, tmp_path):
    """A WAV file that exists in the cache root should be served."""
    audio_file = mock_audio_cache.root_dir / "conv1" / "turn1" / "tts_abc.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)  # minimal fake WAV header

    response = client.get("/api/audio/conv1/turn1/tts_abc.wav")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
