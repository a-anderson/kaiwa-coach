"""Tests for POST /api/conversations/{id}/summarise."""

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
    orc.list_conversations.return_value = []
    orc.get_conversation.return_value = {"id": "conv-1", "turns": [], "language": "ja"}
    orc.summarise_conversation.return_value = {
        "top_error_patterns": ["Particle errors"],
        "priority_areas": ["Particle usage"],
        "overall_notes": "Keep practising.",
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


# ── POST /api/conversations/{id}/summarise ────────────────────────────────────


def test_summarise_returns_200(client):
    resp = client.post("/api/conversations/conv-1/summarise")
    assert resp.status_code == 200


def test_summarise_returns_all_three_fields(client):
    resp = client.post("/api/conversations/conv-1/summarise")
    data = resp.json()
    assert "top_error_patterns" in data
    assert "priority_areas" in data
    assert "overall_notes" in data


def test_summarise_returns_correct_values(client, mock_orchestrator):
    mock_orchestrator.summarise_conversation.return_value = {
        "top_error_patterns": ["Wrong particle", "Missing subject"],
        "priority_areas": ["Particles", "Sentence structure"],
        "overall_notes": "Good progress overall.",
    }
    resp = client.post("/api/conversations/conv-1/summarise")
    data = resp.json()
    assert data["top_error_patterns"] == ["Wrong particle", "Missing subject"]
    assert data["priority_areas"] == ["Particles", "Sentence structure"]
    assert data["overall_notes"] == "Good progress overall."


def test_summarise_calls_orchestrator(client, mock_orchestrator):
    client.post("/api/conversations/conv-1/summarise")
    mock_orchestrator.summarise_conversation.assert_called_once_with("conv-1")


def test_summarise_404_for_unknown_conversation(client, mock_orchestrator):
    mock_orchestrator.get_conversation.side_effect = ValueError("Unknown conversation_id")
    resp = client.post("/api/conversations/unknown-id/summarise")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Conversation not found"


def test_summarise_returns_informational_message_when_no_corrections(client, mock_orchestrator):
    mock_orchestrator.summarise_conversation.return_value = {
        "top_error_patterns": [],
        "priority_areas": [],
        "overall_notes": "No corrections were recorded for this conversation.",
    }
    resp = client.post("/api/conversations/conv-1/summarise")
    assert resp.status_code == 200
    assert resp.json()["overall_notes"] == "No corrections were recorded for this conversation."
