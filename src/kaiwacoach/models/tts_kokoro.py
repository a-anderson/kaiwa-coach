"""Kokoro TTS wrapper with session-only caching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from kaiwacoach.constants import DEFAULT_VOICES
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache


@dataclass(frozen=True)
class TTSResult:
    audio_path: str
    meta: Dict[str, Any]


class TTSBackend(Protocol):
    """Protocol for TTS backend implementations."""

    def synthesize(
        self, text: str, voice: str, speed: float, lang_code: str
    ) -> Tuple[bytes, AudioMeta, Dict[str, Any]]: ...


class KokoroTTS:
    """TTS wrapper with session-only cache and hash-based reuse."""

    def __init__(
        self,
        model_id: str,
        cache: SessionAudioCache,
        backend: TTSBackend | None = None,
    ) -> None:
        """Initialize the TTS wrapper.

        Parameters
        ----------
        model_id : str
            TTS model identifier.
        cache : SessionAudioCache
            Session audio cache for storing generated WAV files.
        backend : TTSBackend | None
            Optional backend implementation; defaults to the MLX-Audio backend.
        """
        self._model_id = model_id
        self._cache = cache
        self._backend = backend or MlxAudioBackend(model_id)
        self._cache_index: Dict[str, str] = {}

    def synthesize(
        self,
        conversation_id: str,
        turn_id: str,
        text: str,
        voice: str | None,
        speed: float,
        lang_code: str | None = None,
        language: str | None = None,
    ) -> TTSResult:
        """Synthesize speech and cache the resulting audio.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        turn_id : str
            Turn identifier.
        text : str
            Text to synthesize.
        voice : str | None
            Voice identifier. If None, use the default voice for the language.
        speed : float
            Speech rate multiplier.
        lang_code : str | None
            Optional language code override (e.g., "j" for Japanese, "f" for French).
        language : str | None
            Optional language name (e.g., "ja", "fr", "en") used to infer defaults.

        Returns
        -------
        TTSResult
            Result containing the cached audio path and metadata.
        """
        if voice is None:
            if lang_code is None and language is None:
                raise ValueError("voice is None and neither lang_code nor language is provided.")
            if lang_code is None:
                lang_code = _language_to_lang_code(language)
            voice = DEFAULT_VOICES.get(_lang_code_to_language(lang_code))
            if voice is None:
                raise ValueError(f"No default voice for language code: {lang_code}")

        effective_lang = lang_code or _infer_lang_code_from_voice(voice)
        cache_key = self._hash_key(text=text, voice=voice, speed=speed, lang_code=effective_lang)
        cached_path = self._cache_index.get(cache_key)
        if cached_path:
            return TTSResult(
                audio_path=cached_path,
                meta={
                    "model_id": self._model_id,
                    "voice": voice,
                    "speed": speed,
                    "lang_code": effective_lang,
                    "cache_hit": True,
                    "cache_key": cache_key,
                },
            )

        pcm_bytes, meta, backend_meta = self._backend.synthesize(
            text=text, voice=voice, speed=speed, lang_code=effective_lang
        )
        audio_path = self._cache.save_audio(
            conversation_id=conversation_id,
            turn_id=turn_id,
            kind="tts",
            pcm_bytes=pcm_bytes,
            meta=meta,
        )
        audio_path_str = str(audio_path)
        self._cache_index[cache_key] = audio_path_str
        out_meta = {
            "model_id": self._model_id,
            "voice": voice,
            "speed": speed,
            "lang_code": effective_lang,
            "cache_hit": False,
            "cache_key": cache_key,
        }
        out_meta.update(backend_meta)
        return TTSResult(audio_path=audio_path_str, meta=out_meta)

    @staticmethod
    def _hash_key(text: str, voice: str, speed: float, lang_code: str) -> str:
        payload = f"{text}|{voice}|{speed}|{lang_code}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class _DefaultBackend:
    """Stub backend that instructs callers to provide a real implementation."""

    def synthesize(self, text: str, voice: str, speed: float, lang_code: str) -> Tuple[bytes, AudioMeta, Dict[str, Any]]:
        raise RuntimeError(
            "TTS backend not configured. Provide a TTSBackend implementation "
            "or integrate Kokoro MLX inference."
        )


class MlxAudioBackend:
    """MLX-Audio backend for Kokoro TTS."""

    def __init__(self, model_id: str) -> None:
        try:
            from mlx_audio.tts.utils import load_model  # type: ignore
            import mlx.core as mx  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-audio is not available. Install the TTS dependency to enable synthesis."
            ) from exc

        self._model_id = model_id
        self._model = load_model(model_id)
        self._mx = mx
        self._np = np

    def synthesize(self, text: str, voice: str, speed: float, lang_code: str) -> Tuple[bytes, AudioMeta, Dict[str, Any]]:
        audio_chunks = []
        sample_rate = None
        for result in self._model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code):
            if sample_rate is None:
                sample_rate = result.sample_rate
            audio_chunks.append(result.audio)

        if not audio_chunks or sample_rate is None:
            raise RuntimeError("mlx-audio returned no audio chunks.")

        full_audio = self._mx.concatenate(audio_chunks, axis=0)
        audio_np = self._np.array(full_audio)
        audio_np = self._np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(self._np.int16)

        pcm_bytes = audio_int16.tobytes()
        meta = AudioMeta(sample_rate=sample_rate, channels=1, sample_width=2)
        return pcm_bytes, meta, {"backend": "mlx_audio"}


def _infer_lang_code_from_voice(voice: str) -> str:
    if voice.startswith("jf_") or voice.startswith("jm_"):
        return "j"
    if voice.startswith("ff_"):
        return "f"
    if voice.startswith("bf_") or voice.startswith("bm_"):
        return "b"
    if voice.startswith("zf_") or voice.startswith("zm_"):
        return "z"
    if voice.startswith("ef_") or voice.startswith("em_"):
        return "e"
    return "a"


def _lang_code_to_language(lang_code: str) -> str:
    mapping = {
        "j": "ja",
        "f": "fr",
        "b": "en",
        "a": "en",
        "z": "en",
        "e": "en",
    }
    return mapping.get(lang_code, "en")


def _language_to_lang_code(language: str | None) -> str:
    if language in {"ja", "japanese"}:
        return "j"
    if language in {"fr", "french"}:
        return "f"
    if language in {"en", "english"}:
        return "b"
    return "a"
