"""SSE tests for monologue turn streaming routes."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from kaiwacoach.api.server import create_app
from kaiwacoach.orchestrator import MonologueTurnResult
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


def _make_monologue_result(
    conversation_id: str = "conv-mono",
    user_turn_id: str = "ut-mono-1",
    input_text: str = "テスト文章",
    asr_text: str | None = None,
) -> MonologueTurnResult:
    return MonologueTurnResult(
        conversation_id=conversation_id,
        user_turn_id=user_turn_id,
        input_text=input_text,
        asr_text=asr_text,
        asr_meta=None,
        corrections={"errors": ["e1"], "corrected": "corrected", "native": "native", "explanation": "exp"},
        summary={"improvement_areas": ["area1"], "overall_assessment": "good"},
    )


def _configure_process_monologue_turn(mock_orc, result, *, emit_asr: bool = False):
    """Configure mock so process_monologue_turn emits stage events and returns result."""

    def fake_process(conversation_id, text=None, pcm_bytes=None, audio_meta=None, on_stage=None):
        if on_stage:
            if emit_asr:
                on_stage("asr", "running", {})
                on_stage("asr", "complete", {"transcript": result.asr_text or ""})
            on_stage("corrections", "running", {})
            on_stage("corrections", "complete", {
                "errors": result.corrections["errors"],
                "corrected": result.corrections["corrected"],
                "native": result.corrections["native"],
                "explanation": result.corrections["explanation"],
            })
            on_stage("summary", "running", {})
            on_stage("summary", "complete", {
                "improvement_areas": result.summary["improvement_areas"],
                "overall_assessment": result.summary["overall_assessment"],
            })
        return result

    mock_orc.process_monologue_turn.side_effect = fake_process


# ── Text monologue route tests ────────────────────────────────────────────────


def test_monologue_text_returns_200(client, mock_orchestrator):
    _configure_process_monologue_turn(mock_orchestrator, _make_monologue_result())
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    assert resp.status_code == 200


def test_monologue_text_content_type_is_event_stream(client, mock_orchestrator):
    _configure_process_monologue_turn(mock_orchestrator, _make_monologue_result())
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    assert "text/event-stream" in resp.headers["content-type"]


def test_monologue_text_emits_stage_events(client, mock_orchestrator):
    _configure_process_monologue_turn(mock_orchestrator, _make_monologue_result())
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    events = parse_sse(resp.text)
    stage_events = [e for e in events if e.get("event") == "stage"]
    names = [(e["data"]["stage"], e["data"]["status"]) for e in stage_events]
    assert ("corrections", "running") in names
    assert ("corrections", "complete") in names
    assert ("summary", "running") in names
    assert ("summary", "complete") in names
    # No asr stage for text input
    assert ("asr", "running") not in names


def test_monologue_text_emits_complete_event(client, mock_orchestrator):
    result = _make_monologue_result()
    _configure_process_monologue_turn(mock_orchestrator, result)
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    events = parse_sse(resp.text)
    complete_events = [e for e in events if e.get("event") == "complete"]
    assert len(complete_events) == 1
    data = complete_events[0]["data"]
    assert data["conversation_id"] == result.conversation_id
    assert data["user_turn_id"] == result.user_turn_id
    assert data["input_text"] == result.input_text
    assert data["corrections"]["errors"] == result.corrections["errors"]
    assert data["corrections"]["corrected"] == result.corrections["corrected"]
    assert data["corrections"]["native"] == result.corrections["native"]
    assert data["corrections"]["explanation"] == result.corrections["explanation"]
    assert data["summary"]["improvement_areas"] == result.summary["improvement_areas"]
    assert data["summary"]["overall_assessment"] == result.summary["overall_assessment"]


def test_monologue_text_complete_event_is_last(client, mock_orchestrator):
    _configure_process_monologue_turn(mock_orchestrator, _make_monologue_result())
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    events = parse_sse(resp.text)
    assert events[-1]["event"] == "complete"


def test_monologue_text_stage_events_before_complete(client, mock_orchestrator):
    _configure_process_monologue_turn(mock_orchestrator, _make_monologue_result())
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    events = parse_sse(resp.text)
    types = [e.get("event") for e in events]
    complete_idx = types.index("complete")
    assert all(t == "stage" for t in types[:complete_idx])


# ── Audio monologue route tests ───────────────────────────────────────────────


def test_monologue_audio_returns_200(client, mock_orchestrator):
    result = _make_monologue_result(asr_text="テスト")
    mock_orchestrator.webm_to_pcm = MagicMock(return_value=(b"\x00" * 100, MagicMock()))
    _configure_process_monologue_turn(mock_orchestrator, result, emit_asr=True)

    with pytest.MonkeyPatch.context() as mp:
        import kaiwacoach.api.routes.monologue as mono_mod
        mp.setattr(mono_mod, "webm_to_pcm", lambda *a, **kw: (b"\x00" * 100, MagicMock()))
        resp = client.post(
            "/api/turns/monologue/audio",
            data={"conversation_id": "conv-mono"},
            files={"audio": ("test.webm", io.BytesIO(b"\x00" * 100), "audio/webm")},
        )
    assert resp.status_code == 200


def test_monologue_audio_emits_asr_stage(client, mock_orchestrator):
    result = _make_monologue_result(asr_text="テスト")
    _configure_process_monologue_turn(mock_orchestrator, result, emit_asr=True)

    with pytest.MonkeyPatch.context() as mp:
        import kaiwacoach.api.routes.monologue as mono_mod
        mp.setattr(mono_mod, "webm_to_pcm", lambda *a, **kw: (b"\x00" * 100, MagicMock()))
        resp = client.post(
            "/api/turns/monologue/audio",
            data={"conversation_id": "conv-mono"},
            files={"audio": ("test.webm", io.BytesIO(b"\x00" * 100), "audio/webm")},
        )
    events = parse_sse(resp.text)
    stage_names = [(e["data"]["stage"], e["data"]["status"]) for e in events if e.get("event") == "stage"]
    assert ("asr", "running") in stage_names
    assert ("asr", "complete") in stage_names


def test_monologue_audio_empty_upload_returns_400(client, mock_orchestrator):
    resp = client.post(
        "/api/turns/monologue/audio",
        data={"conversation_id": "conv-mono"},
        files={"audio": ("test.webm", io.BytesIO(b""), "audio/webm")},
    )
    assert resp.status_code == 400


# ── Mid-stream failure (text route) ──────────────────────────────────────────


def test_monologue_text_midstream_failure_emits_error_after_stage(client, mock_orchestrator):
    """Verify stage events appear before error, and nothing follows the error."""

    def failing_process(conversation_id, text=None, pcm_bytes=None, audio_meta=None, on_stage=None):
        if on_stage:
            on_stage("corrections", "running", {})
        raise RuntimeError("LLM blew up")

    mock_orchestrator.process_monologue_turn.side_effect = failing_process
    resp = client.post(
        "/api/turns/monologue/text",
        json={"conversation_id": "conv-mono", "text": "テスト"},
    )
    events = parse_sse(resp.text)
    types = [e.get("event") for e in events]
    assert "error" in types
    error_idx = types.index("error")
    # At least one stage event before the error
    assert any(t == "stage" for t in types[:error_idx])
    # Nothing after the error
    assert types[error_idx + 1:] == []
    # Error includes request_id
    error_data = events[error_idx]["data"]
    assert "request_id" in error_data
    assert "message" in error_data


# ── Mid-stream failure (audio route) ─────────────────────────────────────────


def test_monologue_audio_midstream_failure_emits_error_after_stage(client, mock_orchestrator):
    """Verify stage events appear before error for the audio route."""

    def failing_process(conversation_id, text=None, pcm_bytes=None, audio_meta=None, on_stage=None):
        if on_stage:
            on_stage("asr", "running", {})
            on_stage("asr", "complete", {"transcript": "テスト"})
            on_stage("corrections", "running", {})
        raise RuntimeError("corrections blew up")

    mock_orchestrator.process_monologue_turn.side_effect = failing_process

    with pytest.MonkeyPatch.context() as mp:
        import kaiwacoach.api.routes.monologue as mono_mod
        mp.setattr(mono_mod, "webm_to_pcm", lambda *a, **kw: (b"\x00" * 100, MagicMock()))
        resp = client.post(
            "/api/turns/monologue/audio",
            data={"conversation_id": "conv-mono"},
            files={"audio": ("test.webm", io.BytesIO(b"\x00" * 100), "audio/webm")},
        )
    events = parse_sse(resp.text)
    types = [e.get("event") for e in events]
    assert "error" in types
    error_idx = types.index("error")
    assert any(t == "stage" for t in types[:error_idx])
    assert types[error_idx + 1:] == []
    error_data = events[error_idx]["data"]
    assert "request_id" in error_data
