"""Tests for the POST /api/conversations/monologue endpoint."""

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
    orc.expected_sample_rate = 16000
    orc.create_monologue_conversation.return_value = "conv-mono-1"
    orc.list_conversations.return_value = []
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


# ── POST /api/conversations/monologue ─────────────────────────────────────────


def test_create_monologue_conversation_returns_201(client, mock_orchestrator):
    resp = client.post("/api/conversations/monologue")
    assert resp.status_code == 201


def test_create_monologue_conversation_returns_conversation_id(client, mock_orchestrator):
    resp = client.post("/api/conversations/monologue")
    data = resp.json()
    assert "conversation_id" in data
    assert data["conversation_id"] == "conv-mono-1"


def test_create_monologue_conversation_calls_orchestrator(client, mock_orchestrator):
    client.post("/api/conversations/monologue")
    mock_orchestrator.create_monologue_conversation.assert_called_once()


# ── GET /api/conversations?conversation_type= ────────────────────────────────


def test_list_conversations_passes_type_chat(client, mock_orchestrator):
    mock_orchestrator.list_conversations.return_value = []
    client.get("/api/conversations?conversation_type=chat")
    mock_orchestrator.list_conversations.assert_called_once_with(conversation_type="chat")


def test_list_conversations_passes_type_monologue(client, mock_orchestrator):
    mock_orchestrator.list_conversations.return_value = []
    client.get("/api/conversations?conversation_type=monologue")
    mock_orchestrator.list_conversations.assert_called_once_with(conversation_type="monologue")


def test_list_conversations_no_type_passes_none(client, mock_orchestrator):
    mock_orchestrator.list_conversations.return_value = []
    client.get("/api/conversations")
    mock_orchestrator.list_conversations.assert_called_once_with(conversation_type=None)


def test_list_conversations_returns_conversation_type_field(client, mock_orchestrator):
    mock_orchestrator.list_conversations.return_value = [
        {
            "id": "c1",
            "title": "Test",
            "language": "ja",
            "updated_at": "2026-01-01",
            "preview_text": None,
            "conversation_type": "monologue",
        }
    ]
    resp = client.get("/api/conversations")
    data = resp.json()
    assert data[0]["conversation_type"] == "monologue"


# ── GET /api/conversations/{id} includes conversation_type ───────────────────


def test_get_conversation_includes_conversation_type(client, mock_orchestrator):
    mock_orchestrator.get_conversation.return_value = {
        "id": "c1",
        "title": "Test",
        "language": "ja",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
        "conversation_type": "monologue",
        "turns": [],
    }
    resp = client.get("/api/conversations/c1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_type"] == "monologue"


def test_get_conversation_defaults_conversation_type_to_chat(client, mock_orchestrator):
    mock_orchestrator.get_conversation.return_value = {
        "id": "c1",
        "title": "Test",
        "language": "ja",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
        "turns": [],
        # conversation_type absent — old DB record
    }
    resp = client.get("/api/conversations/c1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_type"] == "chat"
