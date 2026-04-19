"""Factory functions for building model wrappers from application config.

Each function reads config, selects the appropriate backend and wrapper, and
returns a protocol-typed instance. Add new backend routing here as new backends
and model families are integrated.

LLM routing is two-step:
  1. Backend selection: config.models.llm_backend → MlxLmBackend or OllamaBackend
  2. Family detection: config.models.llm_id prefix → QwenLLM (or future GemmaLLM)
"""

from __future__ import annotations

import dataclasses

from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.llm_backends import MlxLmBackend, OllamaBackend
from kaiwacoach.models.llm_gemma import GemmaLLM
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.protocols import ASRProtocol, LLMProtocol, TTSProtocol
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.settings import AppConfig
from kaiwacoach.storage.blobs import SessionAudioCache

# Maps llm_id prefixes to model family names. First matching prefix wins.
# Extend this list when adding new model families (e.g. Gemma 4 in PR 2).
_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("mlx-community/Qwen3-", "qwen3"),
    ("qwen3:", "qwen3"),
    ("mlx-community/gemma-4-", "gemma4"),
    ("gemma4:", "gemma4"),
]


def _detect_family(llm_id: str) -> str:
    """Return the model family inferred from the llm_id prefix.

    Raises ValueError at startup if no prefix matches, so misconfigured IDs
    are caught before any model is loaded.
    """
    for prefix, family in _FAMILY_PREFIXES:
        if llm_id.startswith(prefix):
            return family
    raise ValueError(
        f"Cannot determine LLM model family for ID {llm_id!r}. "
        f"Supported prefixes: {[p for p, _ in _FAMILY_PREFIXES]}"
    )


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

    Selects backend by config.models.llm_backend, then routes to the correct
    wrapper by model family (detected from config.models.llm_id prefix).
    """
    llm_id = config.models.llm_id
    backend_name = config.models.llm_backend
    family = _detect_family(llm_id)

    if backend_name == "mlx":
        backend = MlxLmBackend(llm_id)
        token_counter = backend.count_tokens
    elif backend_name == "ollama":
        OllamaBackend.check_available()
        # Gemma 4 26B-A4B generates mandatory thought blocks that consume the
        # token budget before the actual answer. Suppressing thinking ensures
        # the full cap is available for the JSON answer across all roles.
        backend = OllamaBackend(llm_id, suppress_thinking=(family == "gemma4"))
        token_counter = None  # Ollama does not expose a token-counting endpoint
    else:
        raise ValueError(f"Unknown llm_backend: {backend_name!r}")

    if family == "qwen3":
        return QwenLLM(
            model_id=llm_id,
            max_context_tokens=config.llm.max_context_tokens,
            role_max_new_tokens=dataclasses.asdict(config.llm.role_max_new_tokens),
            backend=backend,
            token_counter=token_counter,
            conversation_temperature=config.llm.conversation_temperature,
        )

    if family == "gemma4":
        backend_label = "mlx_lm" if backend_name == "mlx" else "ollama"
        return GemmaLLM(
            model_id=llm_id,
            max_context_tokens=config.llm.max_context_tokens,
            role_max_new_tokens=dataclasses.asdict(config.llm.role_max_new_tokens),
            backend=backend,
            token_counter=token_counter,
            conversation_temperature=config.llm.conversation_temperature,
            backend_label=backend_label,
        )

    raise ValueError(f"No LLM wrapper implemented for family {family!r}")


def build_tts(config: AppConfig, cache: SessionAudioCache) -> TTSProtocol:
    """Return a TTS wrapper configured from config.

    All TTS models currently use the MLX Audio (Kokoro) backend.
    Add routing branches here as new TTS backends are integrated.
    """
    return KokoroTTS(
        model_id=config.models.tts_id,
        cache=cache,
    )
