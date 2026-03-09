"""Factory functions for building model wrappers from application config.

Each function inspects the configured model ID and returns the appropriate
wrapper instance. Currently one backend per model type is supported; add
routing branches here as new backends are integrated.
"""

from __future__ import annotations

import dataclasses

from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.llm_qwen import MlxLmBackend, QwenLLM
from kaiwacoach.models.protocols import ASRProtocol, LLMProtocol, TTSProtocol
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.settings import AppConfig
from kaiwacoach.storage.blobs import SessionAudioCache


def build_asr(config: AppConfig) -> ASRProtocol:
    """Return an ASR wrapper configured from config.

    All ASR models currently use the MLX Whisper backend.
    Add routing branches here as new ASR backends are integrated.
    """
    return WhisperASR(
        model_id=config.models.asr_id,
        language=config.session.language,
    )


def build_llm(config: AppConfig) -> LLMProtocol:
    """Return an LLM wrapper configured from config.

    All LLM models currently use the MLX-LM backend.
    Add routing branches here as new LLM backends are integrated.
    """
    backend = MlxLmBackend(config.models.llm_id)
    return QwenLLM(
        model_id=config.models.llm_id,
        max_context_tokens=config.llm.max_context_tokens,
        role_max_new_tokens=dataclasses.asdict(config.llm.role_max_new_tokens),
        backend=backend,
        token_counter=backend.count_tokens,
    )


def build_tts(config: AppConfig, cache: SessionAudioCache) -> TTSProtocol:
    """Return a TTS wrapper configured from config.

    All TTS models currently use the MLX Audio (Kokoro) backend.
    Add routing branches here as new TTS backends are integrated.
    """
    return KokoroTTS(
        model_id=config.models.tts_id,
        cache=cache,
    )
