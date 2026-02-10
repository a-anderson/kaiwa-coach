"""Slow integration tests for real model wiring."""

from __future__ import annotations

from pathlib import Path
import gc

import pytest

from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.llm_qwen import MlxLmBackend, QwenLLM
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.settings import load_config
from kaiwacoach.storage.blobs import SessionAudioCache


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts"


@pytest.mark.slow
def test_asr_integration_real_model(tmp_path: Path) -> None:
    """ASR integration: model loads and returns a transcription."""
    config = load_config()
    audio_path = tmp_path / "silence.wav"

    import wave

    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    try:
        asr = WhisperASR(model_id=config.models.asr_id, language=config.session.language)
    except RuntimeError as exc:  # pragma: no cover
        pytest.skip(f"ASR backend unavailable: {exc}")

    try:
        result = asr.transcribe(audio_path)
        assert isinstance(result.text, str)
        assert "model_id" in result.meta
        assert result.meta.get("language") == config.session.language
    finally:
        del asr
        gc.collect()


@pytest.mark.slow
def test_llm_integration_real_model() -> None:
    """LLM integration: model loads and returns a JSON response."""
    config = load_config()
    prompt_loader = PromptLoader(_prompt_dir())
    prompt = prompt_loader.render(
        "conversation.md",
        {
            "language": config.session.language,
            "conversation_history": "",
            "user_text": "Hello",
        },
    )
    try:
        backend = MlxLmBackend(config.models.llm_id)
    except RuntimeError as exc:  # pragma: no cover
        pytest.skip(f"LLM backend unavailable: {exc}")

    llm = QwenLLM(
        model_id=config.models.llm_id,
        max_context_tokens=config.llm.max_context_tokens,
        role_max_new_tokens=config.llm.role_max_new_tokens.__dict__,
        backend=backend,
    )

    try:
        result = llm.generate_json(prompt=prompt.text, role="conversation")
        assert result.model is not None
        assert getattr(result.model, "reply", "") is not None
    finally:
        del llm
        del backend
        gc.collect()


@pytest.mark.slow
def test_tts_integration_real_model(tmp_path: Path) -> None:
    """TTS integration: model loads and produces an audio file."""
    config = load_config()
    cache = SessionAudioCache(root_dir=tmp_path)
    try:
        tts = KokoroTTS(model_id=config.models.tts_id, cache=cache)
    except RuntimeError as exc:  # pragma: no cover
        pytest.skip(f"TTS backend unavailable: {exc}")

    try:
        result = tts.synthesize(
            conversation_id="conv",
            turn_id="turn",
            text="Test",
            voice=None,
            speed=config.tts.speed,
            language=config.session.language,
        )
        assert Path(result.audio_path).exists()
        assert result.meta.get("cache_hit") is False
    finally:
        cache.cleanup()
        del tts
        gc.collect()
