"""Protocols and shared result types for ASR, LLM, and TTS model wrappers.

Result types (ASRResult, LLMResult, TTSResult) are defined here so that the
protocols and any future concrete implementations share a common vocabulary
without creating circular dependencies.

Concrete implementations (WhisperASR, QwenLLM, KokoroTTS) import the result
types from this module and satisfy the protocols structurally — no explicit
inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from kaiwacoach.models.json_enforcement import ParseResult


@dataclass(frozen=True)
class ASRResult:
    text: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    text: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class TTSResult:
    audio_path: str
    meta: Dict[str, Any]


# @runtime_checkable is applied to all three protocols to enable isinstance
# checks in tests (e.g. assert isinstance(build_asr(config), ASRProtocol)).
# No production code performs isinstance checks against these protocols;
# structural compatibility is relied upon at runtime.

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
