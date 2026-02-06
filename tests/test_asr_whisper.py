"""Tests for the Whisper ASR wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaiwacoach.models.asr_whisper import ASRResult, WhisperASR


def _write_dummy_audio(path: Path) -> None:
    path.write_bytes(b"dummy-audio-bytes")


def test_rejects_unsupported_language(tmp_path: Path) -> None:
    """Unsupported languages should raise at init."""
    with pytest.raises(ValueError, match="Unsupported language"):
        WhisperASR(model_id="test", language="de")


def test_transcribe_uses_transcriber_and_forces_language(tmp_path: Path) -> None:
    """Transcribe should call the injected transcriber with the forced language."""
    audio_path = tmp_path / "audio.wav"
    _write_dummy_audio(audio_path)

    def _transcriber(path: Path, language: str) -> tuple[str, dict]:
        assert path == audio_path
        assert language == "ja"
        return "こんにちは", {"backend": "test"}

    asr = WhisperASR(model_id="model-x", language="ja", transcriber=_transcriber)
    result = asr.transcribe(audio_path)

    assert isinstance(result, ASRResult)
    assert result.text == "こんにちは"
    assert result.meta["backend"] == "test"
    assert result.meta["model_id"] == "model-x"
    assert result.meta["language"] == "ja"
    assert result.meta["cache_hit"] is False


def test_transcribe_caches_by_audio_hash(tmp_path: Path) -> None:
    """Repeated transcriptions of identical audio should hit the cache."""
    audio_path = tmp_path / "audio.wav"
    _write_dummy_audio(audio_path)
    call_count = {"count": 0}

    def _transcriber(_: Path, __: str) -> tuple[str, dict]:
        call_count["count"] += 1
        return "bonjour", {"backend": "test"}

    asr = WhisperASR(model_id="model-x", language="fr", transcriber=_transcriber)
    first = asr.transcribe(audio_path)
    second = asr.transcribe(audio_path)

    assert call_count["count"] == 1
    assert first.text == second.text == "bonjour"
    assert first.meta["cache_hit"] is False
    assert second.meta["cache_hit"] is True


def test_cache_can_be_disabled(tmp_path: Path) -> None:
    """Disabling cache should call the transcriber every time."""
    audio_path = tmp_path / "audio.wav"
    _write_dummy_audio(audio_path)
    call_count = {"count": 0}

    def _transcriber(_: Path, __: str) -> tuple[str, dict]:
        call_count["count"] += 1
        return "test", {}

    asr = WhisperASR(model_id="model-x", language="ja", transcriber=_transcriber, cache_enabled=False)
    asr.transcribe(audio_path)
    asr.transcribe(audio_path)

    assert call_count["count"] == 2


def test_default_transcriber_requires_mlx_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default transcriber should error if mlx_whisper is unavailable."""
    audio_path = tmp_path / "audio.wav"
    _write_dummy_audio(audio_path)

    import builtins

    original_import = builtins.__import__

    def _import_hook(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ):
        if name == "mlx_whisper":
            raise ImportError("no module named mlx_whisper")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_hook)
    asr = WhisperASR(model_id="model-x", language="ja")

    with pytest.raises(RuntimeError, match="mlx-whisper is not available"):
        asr.transcribe(audio_path)


def test_default_transcriber_rejects_unexpected_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default transcriber should reject unexpected result formats."""
    audio_path = tmp_path / "audio.wav"
    _write_dummy_audio(audio_path)

    class _FakeModule:
        @staticmethod
        def transcribe(_: str, **__: object) -> object:
            return ["not-a-dict"]

    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", _FakeModule)
    asr = WhisperASR(model_id="model-x", language="ja")

    with pytest.raises(RuntimeError, match="unexpected result format"):
        asr.transcribe(audio_path)


def test_default_transcriber_passes_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default transcriber should pass path_or_hf_repo to mlx_whisper."""
    audio_path = tmp_path / "audio.wav"
    _write_dummy_audio(audio_path)
    captured: dict[str, object] = {}

    class _FakeModule:
        @staticmethod
        def transcribe(path: str, **kwargs: object) -> dict:
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"text": "ok", "segments": []}

    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", _FakeModule)
    asr = WhisperASR(model_id="model-123", language="ja")

    result = asr.transcribe(audio_path)
    assert result.text == "ok"
    assert captured["kwargs"]["path_or_hf_repo"] == "model-123"
