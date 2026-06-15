"""Tests for VoiceVox TTS backend, wrapper, and language dispatch."""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kaiwacoach.models.tts_voicevox import LanguageDispatchTTS, VoiceVoxBackend, VoiceVoxTTS
from kaiwacoach.models.protocols import TTSResult
from kaiwacoach.storage.blobs import SessionAudioCache


# --- shared helpers ---


def _make_wav_bytes(sample_rate: int = 24000, num_frames: int = 100) -> bytes:
    """Create minimal valid WAV bytes for testing."""
    pcm = b"\x00\x01" * num_frames  # 16-bit mono, 2 bytes per frame
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _make_urlopen_ctx(read_value: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = read_value
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _two_call_urlopen(wav_bytes: bytes):
    """Return a urlopen side_effect that routes by URL: audio_query → JSON, synthesis → WAV."""
    def _fake(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "audio_query" in url:
            return _make_urlopen_ctx(json.dumps({"speedScale": 1.0}).encode())
        return _make_urlopen_ctx(wav_bytes)

    return _fake


# --- VoiceVoxBackend.check_available ---


def test_check_available_returns_true_when_server_responds() -> None:
    with patch("urllib.request.urlopen", return_value=_make_urlopen_ctx(b"0.20.0")):
        assert VoiceVoxBackend.check_available("http://localhost:50021") is True


def test_check_available_returns_false_on_oserror() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert VoiceVoxBackend.check_available("http://localhost:50021") is False


def test_check_available_returns_false_on_url_error() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
        assert VoiceVoxBackend.check_available("http://localhost:50021") is False


def test_check_available_hits_version_endpoint() -> None:
    captured = {}

    def _fake(req, timeout=None):
        captured["url"] = req if isinstance(req, str) else req.full_url
        return _make_urlopen_ctx(b"0.20.0")

    with patch("urllib.request.urlopen", side_effect=_fake):
        VoiceVoxBackend.check_available("http://localhost:50021")

    assert captured["url"] == "http://localhost:50021/version"


# --- VoiceVoxBackend.synthesize ---


def test_synthesize_hits_both_endpoints_with_correct_speaker() -> None:
    captured = []

    def _fake(req, timeout=None):
        captured.append(req.full_url)
        if "audio_query" in req.full_url:
            return _make_urlopen_ctx(json.dumps({"speedScale": 1.0}).encode())
        return _make_urlopen_ctx(_make_wav_bytes())

    with patch("urllib.request.urlopen", side_effect=_fake):
        VoiceVoxBackend.synthesize("こんにちは", speaker_id=74, speed=1.0, url="http://localhost:50021")

    assert any("audio_query" in u and "speaker=74" in u for u in captured)
    assert any("synthesis" in u and "speaker=74" in u for u in captured)


def test_synthesize_sets_speed_scale_in_audio_query_body() -> None:
    synthesis_body = {}

    def _fake(req, timeout=None):
        if "audio_query" in req.full_url:
            return _make_urlopen_ctx(json.dumps({"speedScale": 1.0}).encode())
        synthesis_body.update(json.loads(req.data.decode()))
        return _make_urlopen_ctx(_make_wav_bytes())

    with patch("urllib.request.urlopen", side_effect=_fake):
        VoiceVoxBackend.synthesize("hello", speaker_id=74, speed=1.5, url="http://localhost:50021")

    assert synthesis_body["speedScale"] == pytest.approx(1.5)


def test_synthesize_parses_wav_using_wave_module() -> None:
    wav_bytes = _make_wav_bytes(sample_rate=24000, num_frames=200)

    with patch("urllib.request.urlopen", side_effect=_two_call_urlopen(wav_bytes)):
        pcm_bytes, meta = VoiceVoxBackend.synthesize(
            "hello", speaker_id=74, speed=1.0, url="http://localhost:50021"
        )

    assert meta.sample_rate == 24000
    assert meta.channels == 1
    assert meta.sample_width == 2
    assert pcm_bytes == b"\x00\x01" * 200  # raw PCM, no WAV header


# --- VoiceVoxTTS ---


@pytest.mark.parametrize("speaker_id,expected", [(74, "voicevox:speaker=74"), (3, "voicevox:speaker=3")])
def test_voicevox_tts_model_id_format(speaker_id: int, expected: str, tmp_path: Path) -> None:
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=None)
    tts = VoiceVoxTTS(speaker_id=speaker_id, url="http://localhost:50021", speed=1.0, cache=cache)
    assert tts.model_id == expected


def test_voicevox_tts_cache_miss_calls_backend_and_stores_result(tmp_path: Path) -> None:
    wav_bytes = _make_wav_bytes()
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=None)
    tts = VoiceVoxTTS(speaker_id=74, url="http://localhost:50021", speed=1.0, cache=cache)

    with patch("urllib.request.urlopen", side_effect=_two_call_urlopen(wav_bytes)):
        result = tts.synthesize("conv", "turn1", "こんにちは", voice=None, speed=1.0)

    assert isinstance(result, TTSResult)
    assert Path(result.audio_path).exists()
    assert result.meta["cache_hit"] is False


def test_voicevox_tts_cache_hit_skips_backend(tmp_path: Path) -> None:
    wav_bytes = _make_wav_bytes()
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=None)
    tts = VoiceVoxTTS(speaker_id=74, url="http://localhost:50021", speed=1.0, cache=cache)

    with patch("urllib.request.urlopen", side_effect=_two_call_urlopen(wav_bytes)) as mock_open:
        result1 = tts.synthesize("conv", "turn1", "こんにちは", voice=None, speed=1.0)
        calls_after_first = mock_open.call_count
        result2 = tts.synthesize("conv", "turn2", "こんにちは", voice=None, speed=1.0)
        calls_after_second = mock_open.call_count

    assert result1.audio_path == result2.audio_path
    assert result2.meta["cache_hit"] is True
    assert calls_after_second == calls_after_first


def test_voicevox_tts_ignores_voice_param_uses_speaker_id(tmp_path: Path) -> None:
    """voice param from TTSProtocol call is ignored; constructor speaker_id is used."""
    captured_urls = []

    def _fake(req, timeout=None):
        captured_urls.append(req.full_url)
        if "audio_query" in req.full_url:
            return _make_urlopen_ctx(json.dumps({"speedScale": 1.0}).encode())
        return _make_urlopen_ctx(_make_wav_bytes())

    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=None)
    tts = VoiceVoxTTS(speaker_id=74, url="http://localhost:50021", speed=1.0, cache=cache)

    with patch("urllib.request.urlopen", side_effect=_fake):
        tts.synthesize("conv", "turn1", "hello", voice="jf_alpha", speed=1.0)

    assert all("speaker=74" in url for url in captured_urls)


def test_voicevox_tts_ignores_call_speed_uses_constructor_speed(tmp_path: Path) -> None:
    """speed param passed to synthesize() is ignored; constructor speed sets speedScale."""
    synthesis_body = {}

    def _fake(req, timeout=None):
        if "audio_query" in req.full_url:
            return _make_urlopen_ctx(json.dumps({"speedScale": 1.0}).encode())
        synthesis_body.update(json.loads(req.data.decode()))
        return _make_urlopen_ctx(_make_wav_bytes())

    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=None)
    tts = VoiceVoxTTS(speaker_id=74, url="http://localhost:50021", speed=0.8, cache=cache)

    with patch("urllib.request.urlopen", side_effect=_fake):
        tts.synthesize("conv", "turn1", "hello", voice=None, speed=2.0)  # call speed ignored

    assert synthesis_body["speedScale"] == pytest.approx(0.8)


# --- LanguageDispatchTTS ---


class _StubTTS:
    def __init__(self, model_id_val: str) -> None:
        self._model_id = model_id_val
        self.calls: list[dict] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def synthesize(self, conversation_id, turn_id, text, voice, speed, lang_code=None, language=None):
        self.calls.append({"text": text, "language": language})
        return TTSResult(audio_path=f"/{self._model_id}.wav", meta={"backend": self._model_id})


class _FailingJaTTS:
    @property
    def model_id(self) -> str:
        return "failing-ja"

    def synthesize(self, *args, **kwargs):
        raise OSError("VoiceVox not reachable")


def test_dispatch_routes_japanese_to_japanese_tts() -> None:
    ja_tts = _StubTTS("voicevox")
    other_tts = _StubTTS("kokoro")
    dispatch = LanguageDispatchTTS(japanese_tts=ja_tts, default_tts=other_tts)

    dispatch.synthesize("conv", "turn", "こんにちは", voice=None, speed=1.0, language="ja")

    assert len(ja_tts.calls) == 1
    assert len(other_tts.calls) == 0


def test_dispatch_routes_french_to_default_tts() -> None:
    ja_tts = _StubTTS("voicevox")
    other_tts = _StubTTS("kokoro")
    dispatch = LanguageDispatchTTS(japanese_tts=ja_tts, default_tts=other_tts)

    dispatch.synthesize("conv", "turn", "Bonjour", voice=None, speed=1.0, language="fr")

    assert len(ja_tts.calls) == 0
    assert len(other_tts.calls) == 1


def test_dispatch_routes_none_language_to_default_tts() -> None:
    ja_tts = _StubTTS("voicevox")
    other_tts = _StubTTS("kokoro")
    dispatch = LanguageDispatchTTS(japanese_tts=ja_tts, default_tts=other_tts)

    dispatch.synthesize("conv", "turn", "hello", voice=None, speed=1.0, language=None)

    assert len(other_tts.calls) == 1
    assert len(ja_tts.calls) == 0


def test_dispatch_falls_back_to_default_on_oserror() -> None:
    ja_tts = _FailingJaTTS()
    other_tts = _StubTTS("kokoro")
    dispatch = LanguageDispatchTTS(japanese_tts=ja_tts, default_tts=other_tts)

    result = dispatch.synthesize("conv", "turn", "こんにちは", voice=None, speed=1.0, language="ja")

    assert result.meta["backend"] == "kokoro"
    assert len(other_tts.calls) == 1


def test_dispatch_logs_warning_on_fallback(caplog: pytest.LogCaptureFixture) -> None:
    ja_tts = _FailingJaTTS()
    other_tts = _StubTTS("kokoro")
    dispatch = LanguageDispatchTTS(japanese_tts=ja_tts, default_tts=other_tts)

    with caplog.at_level(logging.WARNING, logger="kaiwacoach.models.tts_voicevox"):
        dispatch.synthesize("conv", "turn", "こんにちは", voice=None, speed=1.0, language="ja")

    assert caplog.records, "expected at least one warning"
    warning_text = " ".join(r.message for r in caplog.records)
    assert "ja" in warning_text or "VoiceVox" in warning_text


def test_dispatch_model_id_includes_both_backend_ids() -> None:
    ja_tts = _StubTTS("voicevox:speaker=74")
    other_tts = _StubTTS("mlx-community/Kokoro-82M-bf16")
    dispatch = LanguageDispatchTTS(japanese_tts=ja_tts, default_tts=other_tts)

    assert "voicevox:speaker=74" in dispatch.model_id
    assert "mlx-community/Kokoro-82M-bf16" in dispatch.model_id
