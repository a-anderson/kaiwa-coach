"""Tests for model factory functions."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaiwacoach.models.factory as factory_module
from kaiwacoach.config.models import ASR_MODEL_ID, LLM_MODEL_ID, LLM_MODEL_ID_BF16, TTS_MODEL_ID
from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.factory import build_asr, build_llm, build_tts
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.protocols import ASRProtocol, LLMProtocol, TTSProtocol
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.settings import AppConfig, LLMConfig, LLMRoleCaps, LoggingConfig, ModelsConfig, SessionConfig, StorageConfig, TTSConfig, UIConfig
from kaiwacoach.storage.blobs import SessionAudioCache


def _asr_config(asr_id: str = "test-asr-model", language: str = "ja") -> SimpleNamespace:
    """Minimal config stub sufficient for build_asr."""
    return SimpleNamespace(
        models=SimpleNamespace(asr_id=asr_id),
        session=SimpleNamespace(language=language),
    )


# --- build_asr (non-slow: WhisperASR does not load the model on init) ---

# Routing tests below (test_build_*_routes_to_*) document the *current* backend
# selection for each model type. They are expected to change when new backends
# are added and the routing logic in factory.py is updated.

def test_build_asr_routes_to_whisper_asr() -> None:
    """build_asr should route to WhisperASR for the current default config."""
    assert isinstance(build_asr(_asr_config()), WhisperASR)


def test_build_asr_wires_model_id() -> None:
    """build_asr should pass the configured model ID to the wrapper."""
    asr = build_asr(_asr_config(asr_id="mlx-community/whisper-large-v3-mlx"))
    assert asr.model_id == "mlx-community/whisper-large-v3-mlx"


def test_build_asr_wires_language() -> None:
    """build_asr should pass the configured session language to the wrapper."""
    asr = build_asr(_asr_config(language="fr"))
    assert asr.language == "fr"


def test_build_asr_satisfies_asr_protocol() -> None:
    """build_asr output should satisfy ASRProtocol."""
    assert isinstance(build_asr(_asr_config()), ASRProtocol)


# --- model ID constants ---

def test_llm_model_id_bf16_constant_value() -> None:
    """LLM_MODEL_ID_BF16 should point to the bf16 Qwen3-14B variant."""
    assert LLM_MODEL_ID_BF16 == "mlx-community/Qwen3-14B-bf16"


def test_llm_model_id_constants_are_distinct() -> None:
    """The 8-bit and bf16 constants must refer to different model IDs."""
    assert LLM_MODEL_ID != LLM_MODEL_ID_BF16


# --- build_llm (non-slow with stubbed backend) ---


class _StubBackend:
    """Stub MlxLmBackend — satisfies LLMBackend protocol without loading a model."""

    def __init__(self, model_id: str) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int, extra_eos_tokens: list[str] | None = None) -> str:
        return ""

    def count_tokens(self, text: str) -> int:
        return 0


def _llm_config(llm_id: str, tmp_path: Path) -> AppConfig:
    """Minimal AppConfig sufficient for build_llm."""
    return AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id=llm_id, tts_id=TTS_MODEL_ID),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )


def test_build_llm_wires_bf16_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should wire the bf16 model ID through to QwenLLM when configured."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    llm = build_llm(_llm_config(LLM_MODEL_ID_BF16, tmp_path))

    assert isinstance(llm, QwenLLM)
    assert llm.model_id == LLM_MODEL_ID_BF16
    assert isinstance(llm, LLMProtocol)


def test_build_llm_wires_8bit_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should wire the 8-bit model ID through to QwenLLM when configured."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    llm = build_llm(_llm_config(LLM_MODEL_ID, tmp_path))

    assert isinstance(llm, QwenLLM)
    assert llm.model_id == LLM_MODEL_ID


# --- build_llm (slow: MlxLmBackend loads the model on init) ---

@pytest.mark.slow
def test_build_llm_routes_to_qwen_llm() -> None:
    """build_llm should route to QwenLLM for the current default config."""
    from kaiwacoach.settings import load_config
    config = load_config()
    try:
        llm = build_llm(config)
    except RuntimeError as exc:
        pytest.skip(f"LLM backend unavailable: {exc}")
    try:
        assert isinstance(llm, QwenLLM)
        assert llm.model_id == config.models.llm_id
        assert llm.max_context_tokens == config.llm.max_context_tokens
        assert isinstance(llm, LLMProtocol)
    finally:
        del llm
        gc.collect()


# --- build_tts (slow: MlxAudioBackend loads the model on init) ---

@pytest.mark.slow
def test_build_tts_routes_to_kokoro_tts(tmp_path: Path) -> None:
    """build_tts should route to KokoroTTS for the current default config."""
    from kaiwacoach.settings import load_config
    config = load_config()
    cache = SessionAudioCache(root_dir=tmp_path)
    try:
        tts = build_tts(config, cache)
    except RuntimeError as exc:
        pytest.skip(f"TTS backend unavailable: {exc}")
    try:
        assert isinstance(tts, KokoroTTS)
        assert tts.model_id == config.models.tts_id
        assert isinstance(tts, TTSProtocol)
    finally:
        cache.cleanup()
        del tts
        gc.collect()
