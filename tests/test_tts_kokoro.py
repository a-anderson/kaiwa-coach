"""Tests for the Kokoro TTS wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaiwacoach.models.tts_kokoro import KokoroTTS, TTSResult, TTSBackend, MlxAudioBackend
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache


class _Backend:
    def __init__(self) -> None:
        self.calls = []

    def synthesize(self, text: str, voice: str, speed: float, lang_code: str):
        self.calls.append((text, voice, speed, lang_code))
        pcm_bytes = b"\x00\x01" * 200
        meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)
        return pcm_bytes, meta, {"backend": "test"}


def test_synthesize_caches_by_text_voice_speed(tmp_path: Path) -> None:
    cache = SessionAudioCache(root_dir=tmp_path)
    backend = _Backend()
    tts = KokoroTTS(model_id="model-x", cache=cache, backend=backend)

    result1 = tts.synthesize("conv", "turn", "hello", "voice", 1.0)
    result2 = tts.synthesize("conv", "turn", "hello", "voice", 1.0)

    assert isinstance(result1, TTSResult)
    assert result1.audio_path == result2.audio_path
    assert result1.meta["cache_hit"] is False
    assert result2.meta["cache_hit"] is True
    assert len(backend.calls) == 1


def test_synthesize_raises_without_backend(tmp_path: Path) -> None:
    cache = SessionAudioCache(root_dir=tmp_path)
    class _FailBackend:
        def synthesize(self, text: str, voice: str, speed: float, lang_code: str):
            raise RuntimeError("no backend")

    tts = KokoroTTS(model_id="model-x", cache=cache, backend=_FailBackend())

    with pytest.raises(RuntimeError, match="no backend"):
        tts.synthesize("conv", "turn", "hello", "voice", 1.0)


def test_default_voice_from_lang_code(tmp_path: Path) -> None:
    cache = SessionAudioCache(root_dir=tmp_path)
    backend = _Backend()
    tts = KokoroTTS(model_id="model-x", cache=cache, backend=backend)

    tts.synthesize("conv", "turn", "hello", None, 1.0, lang_code="j")

    assert backend.calls[0][1] == "jf_alpha"


def test_default_voice_from_language(tmp_path: Path) -> None:
    cache = SessionAudioCache(root_dir=tmp_path)
    backend = _Backend()
    tts = KokoroTTS(model_id="model-x", cache=cache, backend=backend)

    tts.synthesize("conv", "turn", "hello", None, 1.0, language="fr")

    assert backend.calls[0][1] == "ff_siwis"


def test_mlx_audio_backend_synthesizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """MlxAudioBackend should load model and return PCM bytes."""
    class _FakeResult:
        def __init__(self, audio, sample_rate):
            self.audio = audio
            self.sample_rate = sample_rate

    class _FakeModel:
        def generate(self, text: str, voice: str, speed: float, lang_code: str):
            return [
                _FakeResult([0.0, 0.5], 22050),
                _FakeResult([0.5, -0.5], 22050),
            ]

    class _FakeMx:
        @staticmethod
        def concatenate(chunks, axis=0):
            out = []
            for chunk in chunks:
                out.extend(chunk)
            return out

    class _FakeArray:
        def __init__(self, data):
            self._data = data

        def __mul__(self, value):
            return _FakeArray([x * value for x in self._data])

        def astype(self, _dtype):
            return self

        def tobytes(self):
            # Simple bytes conversion for test purposes.
            return b"\\x00\\x01" * len(self._data)

    class _FakeNp:
        int16 = int

        @staticmethod
        def array(data):
            return _FakeArray(list(data))

        @staticmethod
        def clip(data, min_val, max_val):
            return _FakeArray([max(min(x, max_val), min_val) for x in data._data])

    def _load_model(_model_id: str):
        return _FakeModel()

    monkeypatch.setitem(__import__("sys").modules, "mlx_audio.tts.utils", type("m", (), {"load_model": _load_model}))
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _FakeMx)
    monkeypatch.setitem(__import__("sys").modules, "numpy", _FakeNp)

    backend = MlxAudioBackend("model-x")
    pcm_bytes, meta, meta_extra = backend.synthesize("text", "voice", 1.0, "j")

    assert isinstance(pcm_bytes, (bytes, bytearray))
    assert meta.sample_rate == 22050
    assert meta.channels == 1
    assert meta.sample_width == 2
    assert meta_extra["backend"] == "mlx_audio"
