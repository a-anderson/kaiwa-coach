"""Protocols for ASR, LLM, and TTS model wrappers.

These protocols define the interfaces the orchestrator depends on.
Concrete implementations (WhisperASR, QwenLLM, KokoroTTS) satisfy them
structurally — no explicit inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from kaiwacoach.models.asr_whisper import ASRResult
from kaiwacoach.models.json_enforcement import ParseResult
from kaiwacoach.models.llm_qwen import LLMResult
from kaiwacoach.models.tts_kokoro import TTSResult


@runtime_checkable
class ASRProtocol(Protocol):
    @property
    def model_id(self) -> str: ...

    def transcribe(self, audio_path: str | Path) -> ASRResult: ...

    def set_language(self, language: str) -> None: ...


@runtime_checkable
class LLMProtocol(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def max_context_tokens(self) -> int: ...

    def generate(self, prompt: str, role: str, max_new_tokens: Optional[int] = None) -> LLMResult: ...

    def generate_json(
        self,
        prompt: str,
        role: str,
        max_new_tokens: Optional[int] = None,
        repair_fn: Callable[[str], str] | None = None,
    ) -> ParseResult: ...

    def count_tokens(self, text: str) -> int | None: ...

    def clear_cache(self) -> None: ...


@runtime_checkable
class TTSProtocol(Protocol):
    @property
    def model_id(self) -> str: ...

    def synthesize(
        self,
        conversation_id: str,
        turn_id: str,
        text: str,
        voice: str | None,
        speed: float,
        lang_code: str | None = None,
        language: str | None = None,
    ) -> TTSResult: ...
