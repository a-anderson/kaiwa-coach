"""Tests for model factory functions."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.factory import build_asr, build_llm, build_tts
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.protocols import ASRProtocol, LLMProtocol, TTSProtocol
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.storage.blobs import SessionAudioCache


def _asr_config(asr_id: str = "test-asr-model", language: str = "ja") -> SimpleNamespace:
    """Minimal config stub sufficient for build_asr."""
    return SimpleNamespace(
        models=SimpleNamespace(asr_id=asr_id),
        session=SimpleNamespace(language=language),
    )


# --- build_asr (non-slow: WhisperASR does not load the model on init) ---

def test_build_asr_returns_whisper_asr() -> None:
    """build_asr should return a WhisperASR instance."""
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


# --- build_llm (slow: MlxLmBackend loads the model on init) ---

@pytest.mark.slow
def test_build_llm_returns_qwen_llm() -> None:
    """build_llm should return a QwenLLM wired from real config."""
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
def test_build_tts_returns_kokoro_tts(tmp_path: Path) -> None:
    """build_tts should return a KokoroTTS wired from real config."""
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
