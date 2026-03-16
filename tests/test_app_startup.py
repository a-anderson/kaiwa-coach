"""Tests for app startup wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kaiwacoach.app as app_module
from kaiwacoach.config.models import ASR_MODEL_ID, LLM_MODEL_ID, LLM_MODEL_ID_BF16, TTS_MODEL_ID
from kaiwacoach.settings import load_config


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


# --- startup model logging ---

def test_main_logs_model_ids_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    """main should log ASR, LLM, and TTS model IDs at INFO level before loading models.

    capfd is used instead of caplog because main() calls logging.basicConfig(force=True),
    which removes caplog's handler before it can capture anything.
    """
    def _load_config():
        return SimpleNamespace(
            models=SimpleNamespace(
                asr_id="mlx-community/whisper-large-v3-mlx",
                llm_id="mlx-community/Qwen3-14B-bf16",
                tts_id="mlx-community/Kokoro-82M-bf16",
            ),
            session=SimpleNamespace(language="ja"),
            llm=SimpleNamespace(max_context_tokens=10, role_max_new_tokens=SimpleNamespace(**{"role": 3})),
            storage=SimpleNamespace(root_dir=str(tmp_path / "storage"), expected_sample_rate=16000),
            tts=SimpleNamespace(voice="default", speed=1.0),
            logging=SimpleNamespace(timing_logs=True),
        )

    class _Noop:
        """Minimal stub satisfying the interface of any component main() constructs."""
        def __init__(self, *_args: object, **_kw: object) -> None: pass
        def start(self) -> None: pass
        def close(self) -> None: pass
        def cleanup(self) -> None: pass

    monkeypatch.setattr(app_module, "load_config", _load_config)
    monkeypatch.setattr(app_module, "build_asr", lambda cfg: _Noop())
    monkeypatch.setattr(app_module, "build_llm", lambda cfg: _Noop())
    monkeypatch.setattr(app_module, "build_tts", lambda cfg, cache: _Noop())
    monkeypatch.setattr(app_module, "SessionAudioCache", _Noop)
    monkeypatch.setattr(app_module, "SQLiteWriter", _Noop)
    monkeypatch.setattr(app_module, "PromptLoader", _Noop)
    monkeypatch.setattr(app_module, "ConversationOrchestrator", _Noop)
    monkeypatch.setattr(app_module.atexit, "register", lambda *_args, **_kwargs: None)

    app_module.main(launch_ui=False)
    stderr = capfd.readouterr().err

    assert "mlx-community/whisper-large-v3-mlx" in stderr
    assert "mlx-community/Qwen3-14B-bf16" in stderr
    assert "mlx-community/Kokoro-82M-bf16" in stderr


# --- load_config: LLM model ID overrides ---

def test_load_config_accepts_bf16_llm_id_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config should accept the bf16 model ID when set via KAIWACOACH_MODELS_LLM_ID."""
    monkeypatch.setenv("KAIWACOACH_MODELS_LLM_ID", LLM_MODEL_ID_BF16)

    config = load_config()

    assert config.models.llm_id == LLM_MODEL_ID_BF16


def test_load_config_bf16_llm_id_roundtrips_to_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bf16 model ID should survive the load_config → to_dict round-trip."""
    monkeypatch.setenv("KAIWACOACH_MODELS_LLM_ID", LLM_MODEL_ID_BF16)

    config = load_config()

    assert config.to_dict()["models"]["llm_id"] == LLM_MODEL_ID_BF16


# --- model ID validation ---

def test_load_config_rejects_unknown_llm_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config should raise ValueError for an unrecognised LLM model ID."""
    monkeypatch.setenv("KAIWACOACH_MODELS_LLM_ID", "mlx-community/unknown-model")

    with pytest.raises(ValueError, match="Unsupported models.llm_id"):
        load_config()


def test_load_config_rejects_unknown_asr_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config should raise ValueError for an unrecognised ASR model ID."""
    monkeypatch.setenv("KAIWACOACH_MODELS_ASR_ID", "mlx-community/unknown-asr")

    with pytest.raises(ValueError, match="Unsupported models.asr_id"):
        load_config()


def test_load_config_rejects_unknown_tts_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config should raise ValueError for an unrecognised TTS model ID."""
    monkeypatch.setenv("KAIWACOACH_MODELS_TTS_ID", "mlx-community/unknown-tts")

    with pytest.raises(ValueError, match="Unsupported models.tts_id"):
        load_config()


def test_load_config_error_message_lists_valid_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The validation error for an unknown LLM ID should list the valid alternatives."""
    monkeypatch.setenv("KAIWACOACH_MODELS_LLM_ID", "typo-model")

    with pytest.raises(ValueError, match=LLM_MODEL_ID):
        load_config()
