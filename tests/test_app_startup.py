"""Tests for app startup wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kaiwacoach.app as app_module


def test_app_main_wires_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should construct ASR, LLM, and TTS with config values."""
    calls = SimpleNamespace(asr=None, llm=None, tts=None)

    def _load_config():
        return SimpleNamespace(
            models=SimpleNamespace(asr_id="asr", llm_id="llm", tts_id="tts"),
            session=SimpleNamespace(language="ja"),
            llm=SimpleNamespace(max_context_tokens=10, role_max_new_tokens=SimpleNamespace(**{"role": 3})),
        )

    class _ASR:
        def __init__(self, model_id: str, language: str) -> None:
            calls.asr = (model_id, language)

    class _Backend:
        def __init__(self, model_id: str) -> None:
            calls.llm = (model_id,)

    class _LLM:
        def __init__(self, model_id: str, max_context_tokens: int, role_max_new_tokens, backend) -> None:
            calls.llm = (model_id, max_context_tokens, dict(role_max_new_tokens), backend)

    class _TTS:
        def __init__(self, model_id: str, cache) -> None:
            calls.tts = (model_id, cache)

    class _Cache:
        def __init__(self) -> None:
            self.cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    monkeypatch.setattr(app_module, "load_config", _load_config)
    monkeypatch.setattr(app_module, "WhisperASR", _ASR)
    monkeypatch.setattr(app_module, "MlxLmBackend", _Backend)
    monkeypatch.setattr(app_module, "QwenLLM", _LLM)
    monkeypatch.setattr(app_module, "KokoroTTS", _TTS)
    monkeypatch.setattr(app_module, "SessionAudioCache", _Cache)
    monkeypatch.setattr(app_module.atexit, "register", lambda *_args, **_kwargs: None)

    app_module.main()

    assert calls.asr == ("asr", "ja")
    assert calls.llm[0] == "llm"
    assert calls.tts[0] == "tts"
