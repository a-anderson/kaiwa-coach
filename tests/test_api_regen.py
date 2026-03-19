"""Tests for audio regeneration routes.

POST /api/turns/{assistant_turn_id}/regen-audio
POST /api/conversations/{conversation_id}/regen-audio
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from kaiwacoach.api.server import create_app
from kaiwacoach.storage.blobs import SessionAudioCache


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_orchestrator():
    orc = MagicMock()
    orc.language = "ja"
    orc.expected_sample_rate = 16000
    orc.create_conversation.return_value = "conv-abc"
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_sse(text: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:"):].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ── Single-turn regen tests ───────────────────────────────────────────────────


def test_regen_turn_audio_returns_audio_url(client, mock_orchestrator, mock_audio_cache):
    """Successful regen returns a /api/audio/... URL."""
    audio_file = mock_audio_cache.root_dir / "at-1" / "tts.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

    result = MagicMock()
    result.audio_path = str(audio_file)
    mock_orchestrator.regenerate_turn_audio.return_value = result

    resp = client.post("/api/turns/at-1/regen-audio")
    assert resp.status_code == 200
    assert resp.json()["audio_url"].startswith("/api/audio/")


def test_regen_turn_audio_unknown_id_returns_404(client, mock_orchestrator):
    """ValueError with 'Unknown assistant_turn_id' maps to 404 'Turn not found'."""
    mock_orchestrator.regenerate_turn_audio.side_effect = ValueError(
        "Unknown assistant_turn_id: bad-id"
    )
    resp = client.post("/api/turns/bad-id/regen-audio")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Turn not found"


def test_regen_turn_audio_tts_not_configured_returns_422(client, mock_orchestrator):
    """ValueError not about a missing ID maps to 422 'TTS not configured'."""
    mock_orchestrator.regenerate_turn_audio.side_effect = ValueError("TTS model not loaded")
    resp = client.post("/api/turns/at-1/regen-audio")
    assert resp.status_code == 422
    assert resp.json()["detail"] == "TTS not configured"


def test_regen_turn_audio_runtime_error_returns_500(client, mock_orchestrator):
    """RuntimeError maps to 500 with a noun-phrase detail."""
    mock_orchestrator.regenerate_turn_audio.side_effect = RuntimeError("synthesis failed")
    resp = client.post("/api/turns/at-1/regen-audio")
    assert resp.status_code == 500
    assert resp.json()["detail"].startswith("Audio regeneration failed:")


# ── Conversation regen SSE tests ──────────────────────────────────────────────


def test_regen_conversation_audio_emits_turn_done_and_complete(
    client, mock_orchestrator, mock_audio_cache
):
    """Successful regen streams turn_done events then a complete event."""
    audio_file = mock_audio_cache.root_dir / "at-1" / "tts.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

    def fake_regen(conversation_id, *, on_turn):
        on_turn("at-1", str(audio_file))

    mock_orchestrator.regenerate_conversation_audio.side_effect = fake_regen

    resp = client.post("/api/conversations/conv-abc/regen-audio")
    assert resp.status_code == 200
    events = parse_sse(resp.text)

    turn_done = [e for e in events if e.get("event") == "turn_done"]
    complete = [e for e in events if e.get("event") == "complete"]

    assert len(turn_done) == 1
    assert turn_done[0]["data"]["assistant_turn_id"] == "at-1"
    assert turn_done[0]["data"]["audio_url"].startswith("/api/audio/")
    assert len(complete) == 1


def test_regen_conversation_audio_complete_is_last_event(client, mock_orchestrator):
    """complete must be the final event in a successful stream."""
    mock_orchestrator.regenerate_conversation_audio.side_effect = lambda _id, *, on_turn: None

    resp = client.post("/api/conversations/conv-abc/regen-audio")
    events = parse_sse(resp.text)
    assert events[-1]["event"] == "complete"


def test_regen_conversation_audio_error_includes_request_id(client, mock_orchestrator):
    """SSE error event must include a request_id field for log correlation."""
    mock_orchestrator.regenerate_conversation_audio.side_effect = RuntimeError("TTS exploded")

    resp = client.post("/api/conversations/conv-abc/regen-audio")
    events = parse_sse(resp.text)

    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 1
    assert "TTS exploded" in error_events[0]["data"]["message"]
    assert "request_id" in error_events[0]["data"]
    assert error_events[0]["data"]["request_id"]  # non-empty


def test_regen_conversation_audio_error_is_last_event(client, mock_orchestrator):
    """No events must appear after the error event."""
    mock_orchestrator.regenerate_conversation_audio.side_effect = RuntimeError("crash")

    resp = client.post("/api/conversations/conv-abc/regen-audio")
    events = parse_sse(resp.text)
    assert events[-1]["event"] == "error"


def test_regen_conversation_audio_turn_done_before_error(client, mock_orchestrator, mock_audio_cache):
    """turn_done events emitted before a failure must appear before the error event."""
    audio_file = mock_audio_cache.root_dir / "at-1" / "tts.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

    def fake_regen(conversation_id, *, on_turn):
        on_turn("at-1", str(audio_file))
        raise RuntimeError("crashed after first turn")

    mock_orchestrator.regenerate_conversation_audio.side_effect = fake_regen

    resp = client.post("/api/conversations/conv-abc/regen-audio")
    events = parse_sse(resp.text)

    turn_done = [e for e in events if e.get("event") == "turn_done"]
    error_events = [e for e in events if e.get("event") == "error"]

    assert len(turn_done) == 1
    assert len(error_events) == 1
    assert events.index(error_events[0]) > events.index(turn_done[-1])
