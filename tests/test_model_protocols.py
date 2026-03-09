"""Tests that concrete model classes satisfy their protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.models.protocols import ASRProtocol, ASRResult, LLMProtocol, TTSProtocol
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache


# --- Stubs ---
# These stubs are intentionally minimal — they only satisfy the internal backend
# protocols (e.g. TTSBackend.synthesize) so the wrappers can be constructed
# without loading real ML models. They are distinct from the behavior-tracking
# stubs in individual model test files (e.g. test_tts_kokoro.py), which capture
# call arguments and simulate more complex interactions.

class _ASRTranscriber:
    def __call__(self, path: Path, language: str) -> Tuple[str, Dict[str, Any]]:
        return ("stub text", {})


class _LLMBackend:
    def generate(self, prompt: str, max_tokens: int, extra_eos_tokens=None) -> str:
        return '{"reply": "ok"}'


class _TTSBackend:
    def synthesize(
        self, text: str, voice: str, speed: float, lang_code: str
    ) -> Tuple[bytes, AudioMeta, Dict[str, Any]]:
        meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)
        return b"\x00\x00", meta, {}


# --- ASR ---

def test_whisper_asr_satisfies_asr_protocol() -> None:
    """WhisperASR should satisfy ASRProtocol."""
    asr = WhisperASR(model_id="test-model", language="ja", transcriber=_ASRTranscriber())
    assert isinstance(asr, ASRProtocol)


def test_asr_protocol_model_id() -> None:
    """WhisperASR.model_id should return the configured model ID."""
    asr = WhisperASR(model_id="test-model", language="ja", transcriber=_ASRTranscriber())
    assert asr.model_id == "test-model"


def test_asr_protocol_transcribe(tmp_path: Path) -> None:
    """WhisperASR.transcribe should return an ASRResult via the stub transcriber."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"\x00" * 64)
    asr = WhisperASR(model_id="test-model", language="ja", transcriber=_ASRTranscriber())
    result = asr.transcribe(audio_file)
    assert isinstance(result, ASRResult)
    assert result.text == "stub text"


def test_asr_protocol_set_language() -> None:
    """WhisperASR.set_language should update the active language."""
    asr = WhisperASR(model_id="test-model", language="ja", transcriber=_ASRTranscriber())
    asr.set_language("fr")
    assert asr.language == "fr"


# --- LLM ---

def _make_llm() -> QwenLLM:
    return QwenLLM(
        model_id="test-model",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 10},
        backend=_LLMBackend(),
    )


def test_qwen_llm_satisfies_llm_protocol() -> None:
    """QwenLLM should satisfy LLMProtocol."""
    assert isinstance(_make_llm(), LLMProtocol)


def test_llm_protocol_model_id() -> None:
    """QwenLLM.model_id should return the configured model ID."""
    assert _make_llm().model_id == "test-model"


def test_llm_protocol_max_context_tokens() -> None:
    """QwenLLM.max_context_tokens should return the configured value."""
    assert _make_llm().max_context_tokens == 100


# --- TTS ---

def _make_tts(tmp_path: Path) -> KokoroTTS:
    cache = SessionAudioCache(root_dir=tmp_path)
    return KokoroTTS(model_id="test-model", cache=cache, backend=_TTSBackend())


def test_kokoro_tts_satisfies_tts_protocol(tmp_path: Path) -> None:
    """KokoroTTS should satisfy TTSProtocol."""
    assert isinstance(_make_tts(tmp_path), TTSProtocol)


def test_tts_protocol_model_id(tmp_path: Path) -> None:
    """KokoroTTS.model_id should return the configured model ID."""
    assert _make_tts(tmp_path).model_id == "test-model"
