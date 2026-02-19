"""Tests for app startup wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kaiwacoach.app as app_module


def test_app_main_wires_components(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """main should construct components with config values."""
    calls = SimpleNamespace(asr=None, llm=None, tts=None, db=None, prompts=None, orchestrator=None)

    def _load_config():
        return SimpleNamespace(
            models=SimpleNamespace(asr_id="asr", llm_id="llm", tts_id="tts"),
            session=SimpleNamespace(language="ja"),
            llm=SimpleNamespace(max_context_tokens=10, role_max_new_tokens=SimpleNamespace(**{"role": 3})),
            storage=SimpleNamespace(root_dir=str(tmp_path / "storage"), expected_sample_rate=16000),
            tts=SimpleNamespace(voice="default", speed=1.0),
            logging=SimpleNamespace(timing_logs=True),
        )

    class _ASR:
        def __init__(self, model_id: str, language: str) -> None:
            calls.asr = (model_id, language)

    class _Backend:
        def __init__(self, model_id: str) -> None:
            calls.llm = (model_id,)

        def count_tokens(self, _text: str) -> int:
            return 0

    class _LLM:
        def __init__(
            self,
            model_id: str,
            max_context_tokens: int,
            role_max_new_tokens,
            backend,
            token_counter=None,
        ) -> None:
            calls.llm = (model_id, max_context_tokens, dict(role_max_new_tokens), backend, token_counter)

    class _TTS:
        def __init__(self, model_id: str, cache) -> None:
            calls.tts = (model_id, cache)

    class _DB:
        def __init__(self, db_path: str | Path, schema_path: str | Path) -> None:
            calls.db = (Path(db_path), Path(schema_path))

        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class _PromptLoader:
        def __init__(self, root_dir: str | Path) -> None:
            calls.prompts = Path(root_dir)

    class _Orchestrator:
        def __init__(self, **kwargs) -> None:
            calls.orchestrator = kwargs

    class _Cache:
        def __init__(self, **_kwargs) -> None:
            self.cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    monkeypatch.setattr(app_module, "load_config", _load_config)
    monkeypatch.setattr(app_module, "WhisperASR", _ASR)
    monkeypatch.setattr(app_module, "MlxLmBackend", _Backend)
    monkeypatch.setattr(app_module, "QwenLLM", _LLM)
    monkeypatch.setattr(app_module, "KokoroTTS", _TTS)
    monkeypatch.setattr(app_module, "SessionAudioCache", _Cache)
    monkeypatch.setattr(app_module, "SQLiteWriter", _DB)
    monkeypatch.setattr(app_module, "PromptLoader", _PromptLoader)
    monkeypatch.setattr(app_module, "ConversationOrchestrator", _Orchestrator)
    monkeypatch.setattr(app_module.atexit, "register", lambda *_args, **_kwargs: None)

    app_module.main(launch_ui=False)

    assert calls.asr == ("asr", "ja")
    assert calls.llm[0] == "llm"
    assert calls.tts[0] == "tts"
    assert calls.db[0].name == "kaiwacoach.sqlite"
    assert calls.db[1].name == "schema.sql"
    assert calls.prompts.name == "prompts"
    assert calls.orchestrator["tts_voice"] == "default"
    assert calls.orchestrator["tts_speed"] == 1.0
    assert calls.orchestrator["timing_logs_enabled"] is True


def test_app_main_passes_logo_dir_to_build_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """main should pass config.ui.logo_dir through to build_ui."""
    calls = SimpleNamespace(build_ui_logo_dir=None, launched=False)

    def _load_config():
        return SimpleNamespace(
            models=SimpleNamespace(asr_id="asr", llm_id="llm", tts_id="tts"),
            session=SimpleNamespace(language="ja"),
            llm=SimpleNamespace(max_context_tokens=10, role_max_new_tokens=SimpleNamespace(**{"role": 3})),
            storage=SimpleNamespace(root_dir=str(tmp_path / "storage"), expected_sample_rate=16000),
            tts=SimpleNamespace(voice="default", speed=1.0),
            logging=SimpleNamespace(timing_logs=True),
            ui=SimpleNamespace(logo_dir=str(tmp_path / "assets" / "logo")),
        )

    class _ASR:
        def __init__(self, model_id: str, language: str) -> None:
            return None

    class _Backend:
        def __init__(self, model_id: str) -> None:
            return None

        def count_tokens(self, _text: str) -> int:
            return 0

    class _LLM:
        def __init__(
            self,
            model_id: str,
            max_context_tokens: int,
            role_max_new_tokens,
            backend,
            token_counter=None,
        ) -> None:
            return None

    class _TTS:
        def __init__(self, model_id: str, cache) -> None:
            return None

    class _DB:
        def __init__(self, db_path: str | Path, schema_path: str | Path) -> None:
            return None

        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class _PromptLoader:
        def __init__(self, root_dir: str | Path) -> None:
            return None

    class _Orchestrator:
        def __init__(self, **kwargs) -> None:
            return None

    class _Cache:
        def __init__(self, **_kwargs) -> None:
            self.cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    class _Demo:
        def launch(self) -> None:
            calls.launched = True

    def _build_ui(_orchestrator, logo_dir: Path):
        calls.build_ui_logo_dir = logo_dir
        return _Demo()

    monkeypatch.setattr(app_module, "load_config", _load_config)
    monkeypatch.setattr(app_module, "WhisperASR", _ASR)
    monkeypatch.setattr(app_module, "MlxLmBackend", _Backend)
    monkeypatch.setattr(app_module, "QwenLLM", _LLM)
    monkeypatch.setattr(app_module, "KokoroTTS", _TTS)
    monkeypatch.setattr(app_module, "SessionAudioCache", _Cache)
    monkeypatch.setattr(app_module, "SQLiteWriter", _DB)
    monkeypatch.setattr(app_module, "PromptLoader", _PromptLoader)
    monkeypatch.setattr(app_module, "ConversationOrchestrator", _Orchestrator)
    monkeypatch.setattr(app_module, "build_ui", _build_ui)
    monkeypatch.setattr(app_module.atexit, "register", lambda *_args, **_kwargs: None)

    app_module.main(launch_ui=True)

    assert calls.build_ui_logo_dir == Path(tmp_path / "assets" / "logo")
    assert calls.launched is True
