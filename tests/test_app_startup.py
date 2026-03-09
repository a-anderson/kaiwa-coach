"""Tests for app startup wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kaiwacoach.app as app_module


def test_app_main_wires_components(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """main should call factory functions with the loaded config."""
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

    class _FakeModel:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    def _build_asr(cfg):
        calls.asr = cfg
        return _FakeModel(cfg.models.asr_id)

    def _build_llm(cfg):
        calls.llm = cfg
        return _FakeModel(cfg.models.llm_id)

    def _build_tts(cfg, cache):
        calls.tts = (cfg, cache)
        return _FakeModel(cfg.models.tts_id)

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
    monkeypatch.setattr(app_module, "build_asr", _build_asr)
    monkeypatch.setattr(app_module, "build_llm", _build_llm)
    monkeypatch.setattr(app_module, "build_tts", _build_tts)
    monkeypatch.setattr(app_module, "SessionAudioCache", _Cache)
    monkeypatch.setattr(app_module, "SQLiteWriter", _DB)
    monkeypatch.setattr(app_module, "PromptLoader", _PromptLoader)
    monkeypatch.setattr(app_module, "ConversationOrchestrator", _Orchestrator)
    monkeypatch.setattr(app_module.atexit, "register", lambda *_args, **_kwargs: None)

    app_module.main(launch_ui=False)

    assert calls.asr.models.asr_id == "asr"
    assert calls.asr.session.language == "ja"
    assert calls.llm.models.llm_id == "llm"
    assert calls.tts[0].models.tts_id == "tts"
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

    class _FakeModel:
        pass

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
    monkeypatch.setattr(app_module, "build_asr", lambda cfg: _FakeModel())
    monkeypatch.setattr(app_module, "build_llm", lambda cfg: _FakeModel())
    monkeypatch.setattr(app_module, "build_tts", lambda cfg, cache: _FakeModel())
    monkeypatch.setattr(app_module, "SessionAudioCache", _Cache)
    monkeypatch.setattr(app_module, "SQLiteWriter", _DB)
    monkeypatch.setattr(app_module, "PromptLoader", _PromptLoader)
    monkeypatch.setattr(app_module, "ConversationOrchestrator", _Orchestrator)
    monkeypatch.setattr(app_module, "build_ui", _build_ui)
    monkeypatch.setattr(app_module.atexit, "register", lambda *_args, **_kwargs: None)

    app_module.main(launch_ui=True)

    assert calls.build_ui_logo_dir == Path(tmp_path / "assets" / "logo")
    assert calls.launched is True
