"""Tests for SSE turn streaming routes (Phase 2).

These tests use a mock orchestrator so no ML models are loaded (non-slow).
SSE events are verified by parsing the raw response text.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from fastapi.testclient import TestClient

from kaiwacoach.api.server import create_app
from kaiwacoach.orchestrator import AudioTurnResult, TextTurnResult
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


def _make_text_result(conversation_id="conv-abc", tts_audio_path=None):
    return TextTurnResult(
        conversation_id=conversation_id,
        user_turn_id="ut-1",
        assistant_turn_id="at-1",
        reply_text="こんにちは！",
        tts_audio_path=tts_audio_path,
    )


def _make_audio_result(conversation_id="conv-abc", tts_audio_path=None):
    return AudioTurnResult(
        conversation_id=conversation_id,
        user_turn_id="ut-1",
        assistant_turn_id="at-1",
        reply_text="こんにちは！",
        input_audio_path="/tmp/audio/user.wav",
        asr_text="こんにちは",
        asr_meta={},
        tts_audio_path=tts_audio_path,
    )


def _configure_process_text_turn(mock_orchestrator, result, *, stages=True):
    """Make process_text_turn emit stage events then return result."""

    def fake_process_text_turn(
        conversation_id,
        user_text,
        conversation_history="",
        corrections_enabled=True,
        on_stage=None,
    ):
        if on_stage and stages:
            on_stage("llm", "running", {})
            on_stage("llm", "complete", {"reply": result.reply_text})
            if corrections_enabled:
                on_stage("corrections", "running", {})
                on_stage("corrections", "complete", {"data": {"errors": [], "corrected": "", "native": "", "explanation": ""}})
            on_stage("tts", "running", {})
            on_stage("tts", "complete", {"audio_path": result.tts_audio_path})
        return result

    mock_orchestrator.process_text_turn.side_effect = fake_process_text_turn


def _configure_process_audio_turn(mock_orchestrator, result, *, stages=True):
    """Make process_audio_turn emit stage events then return result."""

    def fake_process_audio_turn(
        conversation_id,
        pcm_bytes,
        audio_meta,
        conversation_history="",
        corrections_enabled=True,
        on_stage=None,
    ):
        if on_stage and stages:
            on_stage("asr", "running", {})
            on_stage("asr", "complete", {"transcript": result.asr_text})
            on_stage("llm", "running", {})
            on_stage("llm", "complete", {"reply": result.reply_text})
            if corrections_enabled:
                on_stage("corrections", "running", {})
                on_stage("corrections", "complete", {"data": {"errors": [], "corrected": "", "native": "", "explanation": ""}})
            on_stage("tts", "running", {})
            on_stage("tts", "complete", {"audio_path": result.tts_audio_path})
        return result

    mock_orchestrator.process_audio_turn.side_effect = fake_process_audio_turn


# ── Text turn tests ───────────────────────────────────────────────────────────


def test_text_turn_returns_200(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    resp = client.post("/api/turns/text", json={"text": "こんにちは", "language": "ja"})
    assert resp.status_code == 200


def test_text_turn_content_type_is_event_stream(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    resp = client.post("/api/turns/text", json={"text": "hello"})
    assert "text/event-stream" in resp.headers["content-type"]


def test_text_turn_emits_stage_events(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    resp = client.post("/api/turns/text", json={"text": "こんにちは", "language": "ja"})
    events = parse_sse(resp.text)
    stage_events = [e for e in events if e.get("event") == "stage"]
    stage_names = [(e["data"]["stage"], e["data"]["status"]) for e in stage_events]
    assert ("llm", "running") in stage_names
    assert ("llm", "complete") in stage_names
    assert ("corrections", "running") in stage_names
    assert ("tts", "running") in stage_names
    assert ("tts", "complete") in stage_names
    assert stage_names.index(("corrections", "running")) < stage_names.index(("tts", "running"))


def test_text_turn_emits_complete_event(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    resp = client.post("/api/turns/text", json={"text": "こんにちは"})
    events = parse_sse(resp.text)
    complete = [e for e in events if e.get("event") == "complete"]
    assert len(complete) == 1
    data = complete[0]["data"]
    assert data["conversation_id"] == "conv-abc"
    assert data["user_turn_id"] == "ut-1"
    assert data["assistant_turn_id"] == "at-1"
    assert data["reply_text"] == "こんにちは！"


def test_text_turn_complete_event_last(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    resp = client.post("/api/turns/text", json={"text": "test"})
    events = parse_sse(resp.text)
    assert events[-1]["event"] == "complete"


def test_text_turn_audio_url_in_complete_when_file_exists(
    client, mock_orchestrator, mock_audio_cache, tmp_path
):
    """audio_url in the complete event resolves to /api/audio/... when the file exists."""
    audio_file = mock_audio_cache.root_dir / "conv-abc" / "at-1" / "tts.wav"
    audio_file.parent.mkdir(parents=True)
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

    result = _make_text_result(tts_audio_path=str(audio_file))
    _configure_process_text_turn(mock_orchestrator, result)
    resp = client.post("/api/turns/text", json={"text": "test"})
    events = parse_sse(resp.text)
    complete_data = next(e["data"] for e in events if e.get("event") == "complete")
    assert complete_data["audio_url"] is not None
    assert complete_data["audio_url"].startswith("/api/audio/")


def test_text_turn_audio_url_none_when_no_tts(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result(tts_audio_path=None))
    resp = client.post("/api/turns/text", json={"text": "test"})
    events = parse_sse(resp.text)
    complete_data = next(e["data"] for e in events if e.get("event") == "complete")
    assert complete_data["audio_url"] is None


def test_text_turn_auto_creates_conversation(client, mock_orchestrator):
    """If conversation_id is omitted, a new conversation must be created."""
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    client.post("/api/turns/text", json={"text": "hi"})
    mock_orchestrator.create_conversation.assert_called_once()


def test_text_turn_uses_provided_conversation_id(client, mock_orchestrator):
    """If conversation_id is provided, create_conversation must NOT be called."""
    _configure_process_text_turn(mock_orchestrator, _make_text_result(conversation_id="existing"))
    client.post("/api/turns/text", json={"text": "hi", "conversation_id": "existing"})
    mock_orchestrator.create_conversation.assert_not_called()


def test_text_turn_corrections_disabled_skips_correction_stages(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    resp = client.post(
        "/api/turns/text",
        json={"text": "test", "corrections_enabled": False},
    )
    events = parse_sse(resp.text)
    stage_names = [e["data"]["stage"] for e in events if e.get("event") == "stage"]
    assert "corrections" not in stage_names


def test_text_turn_sets_language(client, mock_orchestrator):
    _configure_process_text_turn(mock_orchestrator, _make_text_result())
    client.post("/api/turns/text", json={"text": "bonjour", "language": "fr"})
    mock_orchestrator.set_language.assert_called_once_with("fr")


def test_text_turn_orchestrator_exception_emits_error_event(client, mock_orchestrator):
    mock_orchestrator.process_text_turn.side_effect = RuntimeError("model exploded")
    resp = client.post("/api/turns/text", json={"text": "test"})
    events = parse_sse(resp.text)
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 1
    assert "model exploded" in error_events[0]["data"]["message"]
    assert "request_id" in error_events[0]["data"]


# ── Audio turn tests ──────────────────────────────────────────────────────────


def _make_wav_bytes(*, sample_rate: int = 16000, duration_frames: int = 160) -> bytes:
    """Return a minimal valid WAV file (mono, 16-bit, silent)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * duration_frames * 2)
    return buf.getvalue()


def test_audio_turn_empty_upload_returns_400(client):
    resp = client.post(
        "/api/turns/audio",
        files={"audio": ("rec.webm", b"", "audio/webm")},
    )
    assert resp.status_code == 400


def test_audio_turn_emits_asr_stage(client, mock_orchestrator, monkeypatch):
    """With a patched webm_to_pcm, the SSE stream contains an asr stage event."""
    from kaiwacoach.storage.blobs import AudioMeta

    fake_pcm = b"\x00" * 320
    fake_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2, num_frames=160)

    import kaiwacoach.api.routes.turns as turns_module

    monkeypatch.setattr(turns_module, "webm_to_pcm", lambda *_a, **_kw: (fake_pcm, fake_meta))

    _configure_process_audio_turn(mock_orchestrator, _make_audio_result())

    resp = client.post(
        "/api/turns/audio",
        files={"audio": ("rec.webm", b"fake-webm", "audio/webm")},
        data={"language": "ja"},
    )
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    stage_names = [(e["data"]["stage"], e["data"]["status"]) for e in events if e.get("event") == "stage"]
    assert ("asr", "running") in stage_names
    assert ("asr", "complete") in stage_names
    assert stage_names.index(("corrections", "running")) < stage_names.index(("tts", "running"))


def test_audio_turn_complete_event_includes_asr_text(client, mock_orchestrator, monkeypatch):
    from kaiwacoach.storage.blobs import AudioMeta

    fake_pcm = b"\x00" * 320
    fake_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2, num_frames=160)

    import kaiwacoach.api.routes.turns as turns_module

    monkeypatch.setattr(turns_module, "webm_to_pcm", lambda *_a, **_kw: (fake_pcm, fake_meta))
    _configure_process_audio_turn(mock_orchestrator, _make_audio_result())

    resp = client.post(
        "/api/turns/audio",
        files={"audio": ("rec.webm", b"fake-webm", "audio/webm")},
    )
    events = parse_sse(resp.text)
    complete_data = next(e["data"] for e in events if e.get("event") == "complete")
    assert complete_data["asr_text"] == "こんにちは"


def test_audio_turn_bad_audio_returns_422(client, mock_orchestrator, monkeypatch):
    """If webm_to_pcm raises, the route must return 422."""
    import kaiwacoach.api.routes.turns as turns_module

    monkeypatch.setattr(
        turns_module, "webm_to_pcm", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("bad audio"))
    )

    resp = client.post(
        "/api/turns/audio",
        files={"audio": ("rec.webm", b"garbage", "audio/webm")},
    )
    assert resp.status_code == 422


def test_audio_turn_orchestrator_exception_emits_error_event(client, mock_orchestrator, monkeypatch):
    """If process_audio_turn raises (e.g. ASR failure), the SSE stream must emit an error event."""
    from kaiwacoach.storage.blobs import AudioMeta

    fake_pcm = b"\x00" * 320
    fake_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2, num_frames=160)

    import kaiwacoach.api.routes.turns as turns_module

    monkeypatch.setattr(turns_module, "webm_to_pcm", lambda *_a, **_kw: (fake_pcm, fake_meta))
    mock_orchestrator.process_audio_turn.side_effect = RuntimeError("ASR model crashed")

    resp = client.post(
        "/api/turns/audio",
        files={"audio": ("rec.webm", b"fake-webm", "audio/webm")},
        data={"language": "ja"},
    )
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) == 1
    assert "ASR model crashed" in error_events[0]["data"]["message"]
    assert "request_id" in error_events[0]["data"]


# ── Mid-stream failure tests ───────────────────────────────────────────────────


def _configure_text_turn_fails_after_llm(mock_orchestrator, reply_text: str) -> None:
    """Emit LLM stage events then raise before corrections — simulates a mid-stream failure."""

    def fake_process_text_turn(
        conversation_id, user_text, conversation_history="", corrections_enabled=True, on_stage=None
    ):
        if on_stage:
            on_stage("llm", "running", {})
            on_stage("llm", "complete", {"reply": reply_text})
        raise RuntimeError("TTS model crashed")

    mock_orchestrator.process_text_turn.side_effect = fake_process_text_turn


def _configure_audio_turn_fails_after_asr(mock_orchestrator, asr_text: str) -> None:
    """Emit ASR stage events then raise during LLM — simulates a mid-stream failure."""

    def fake_process_audio_turn(
        conversation_id, pcm_bytes, audio_meta,
        conversation_history="", corrections_enabled=True, on_stage=None,
    ):
        if on_stage:
            on_stage("asr", "running", {})
            on_stage("asr", "complete", {"transcript": asr_text})
            on_stage("llm", "running", {})
        raise RuntimeError("LLM crashed after ASR")

    mock_orchestrator.process_audio_turn.side_effect = fake_process_audio_turn


def test_text_turn_mid_stream_failure_emits_stage_events_then_error(client, mock_orchestrator):
    """Stage events emitted before the failure must appear before the error event."""
    _configure_text_turn_fails_after_llm(mock_orchestrator, reply_text="partial reply")
    resp = client.post("/api/turns/text", json={"text": "test"})
    events = parse_sse(resp.text)

    stage_events = [e for e in events if e.get("event") == "stage"]
    error_events = [e for e in events if e.get("event") == "error"]

    # LLM stages were emitted before the failure.
    stage_names = [(e["data"]["stage"], e["data"]["status"]) for e in stage_events]
    assert ("llm", "running") in stage_names
    assert ("llm", "complete") in stage_names

    # Exactly one error event, after all stage events.
    assert len(error_events) == 1
    assert "TTS model crashed" in error_events[0]["data"]["message"]
    assert events.index(error_events[0]) > events.index(stage_events[-1])


def test_text_turn_mid_stream_failure_no_events_after_error(client, mock_orchestrator):
    """No events must appear after the error event."""
    _configure_text_turn_fails_after_llm(mock_orchestrator, reply_text="partial")
    resp = client.post("/api/turns/text", json={"text": "test"})
    events = parse_sse(resp.text)
    assert events[-1]["event"] == "error"


def test_audio_turn_mid_stream_failure_emits_asr_stages_then_error(
    client, mock_orchestrator, monkeypatch
):
    """ASR stage events emitted before an LLM crash must appear before the error event."""
    from kaiwacoach.storage.blobs import AudioMeta
    import kaiwacoach.api.routes.turns as turns_module

    fake_pcm = b"\x00" * 320
    fake_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2, num_frames=160)
    monkeypatch.setattr(turns_module, "webm_to_pcm", lambda *_a, **_kw: (fake_pcm, fake_meta))

    _configure_audio_turn_fails_after_asr(mock_orchestrator, asr_text="こんにちは")

    resp = client.post(
        "/api/turns/audio",
        files={"audio": ("rec.webm", b"fake-webm", "audio/webm")},
        data={"language": "ja"},
    )
    events = parse_sse(resp.text)

    stage_events = [e for e in events if e.get("event") == "stage"]
    error_events = [e for e in events if e.get("event") == "error"]

    stage_names = [(e["data"]["stage"], e["data"]["status"]) for e in stage_events]
    assert ("asr", "running") in stage_names
    assert ("asr", "complete") in stage_names

    assert len(error_events) == 1
    assert "LLM crashed after ASR" in error_events[0]["data"]["message"]
    assert events.index(error_events[0]) > events.index(stage_events[-1])
