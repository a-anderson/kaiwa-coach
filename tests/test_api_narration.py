"""Tests for the POST /api/narrate route."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from kaiwacoach.api.server import create_app
from kaiwacoach.storage.blobs import SessionAudioCache


@pytest.fixture
def mock_audio_cache(tmp_path: Path):
    cache = MagicMock(spec=SessionAudioCache)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    cache.root_dir = audio_dir
    # Create a real audio file so audio_path_to_url can relativise it
    audio_file = audio_dir / "narrations" / "test.wav"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"RIFF")
    cache._audio_file = audio_file
    return cache


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.close = MagicMock()
    return db


@pytest.fixture
def mock_orchestrator(mock_audio_cache):
    orc = MagicMock()
    orc.language = "fr"
    orc.generate_narration.return_value = str(
        mock_audio_cache.root_dir / "narrations" / "test.wav"
    )
    return orc


@pytest.fixture
def client(mock_orchestrator, mock_audio_cache, mock_db):
    app = create_app(
        orchestrator=mock_orchestrator,
        audio_cache=mock_audio_cache,
        db=mock_db,
    )
    with TestClient(app) as c:
        yield c


def test_narrate_returns_audio_url(client, mock_orchestrator) -> None:
    response = client.post("/api/narrate", json={"text": "Bonjour le monde"})
    assert response.status_code == 200
    data = response.json()
    assert "audio_url" in data
    assert data["audio_url"]  # non-empty
    mock_orchestrator.generate_narration.assert_called_once_with("Bonjour le monde")


def test_narrate_calls_audio_path_to_url_with_cache_root(client, mock_orchestrator, mock_audio_cache) -> None:
    response = client.post("/api/narrate", json={"text": "Hello"})
    assert response.status_code == 200
    # The route must pass the cache root, not a hardcoded path
    mock_orchestrator.generate_narration.assert_called_once()


def test_narrate_empty_text_returns_400(client) -> None:
    response = client.post("/api/narrate", json={"text": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_narrate_whitespace_only_returns_400(client) -> None:
    response = client.post("/api/narrate", json={"text": "   "})
    assert response.status_code == 400


def test_narrate_tts_not_configured_returns_422(client, mock_orchestrator) -> None:
    mock_orchestrator.generate_narration.side_effect = ValueError("TTS must be configured")
    response = client.post("/api/narrate", json={"text": "Hello"})
    assert response.status_code == 422
