"""Whisper ASR wrapper with session-level caching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from kaiwacoach.constants import SUPPORTED_LANGUAGES


@dataclass(frozen=True)
class ASRResult:
    text: str
    meta: Dict[str, Any]


class WhisperASR:
    """ASR wrapper with forced language and audio-hash caching."""

    # TODO: When integrating MLX Whisper, explicitly allow code-switching to preserve English
    # words in mixed-language utterances (avoid normalization that strips Latin tokens).

    def __init__(
        self,
        model_id: str,
        language: str,
        cache_enabled: bool = True,
        transcriber: Callable[[Path, str], Tuple[str, Dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize the ASR wrapper.

        Parameters
        ----------
        model_id : str
            Whisper model identifier.
        language : str
            Forced language code (e.g., "ja", "fr", etc.).
        cache_enabled : bool
            Whether to cache ASR results by audio hash.
        transcriber : Callable[[pathlib.Path, str], tuple[str, dict]] | None
            Optional transcriber callable for dependency injection and testing.
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}.")
        self._model_id = model_id
        self._language = language
        self._cache_enabled = cache_enabled
        self._cache: Dict[str, ASRResult] = {}
        self._transcriber = transcriber or self._default_transcriber

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        """Transcribe audio to text using a forced language setting.

        Parameters
        ----------
        audio_path : str | pathlib.Path
            Path to the audio file to transcribe.

        Returns
        -------
        ASRResult
            Transcription text and metadata.
        """
        path = Path(audio_path)
        audio_hash = self._hash_file(path)

        if self._cache_enabled and audio_hash in self._cache:
            cached = self._cache[audio_hash]
            meta = dict(cached.meta)
            meta["cache_hit"] = True
            return ASRResult(text=cached.text, meta=meta)

        text, meta = self._transcriber(path, self._language)
        meta = dict(meta)
        meta.update(
            {
                "model_id": self._model_id,
                "language": self._language,
                "audio_hash": audio_hash,
                "cache_hit": False,
            }
        )
        result = ASRResult(text=text, meta=meta)
        if self._cache_enabled:
            self._cache[audio_hash] = result
        return result

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _default_transcriber(self, audio_path: Path, language: str) -> Tuple[str, Dict[str, Any]]:
        """Default transcriber hook (requires MLX Whisper integration).

        Parameters
        ----------
        audio_path : pathlib.Path
            Path to the audio file to transcribe.
        language : str
            Forced language code for Whisper.

        Returns
        -------
        tuple[str, dict]
            Transcribed text and metadata.

        Raises
        ------
        RuntimeError
            If no ASR backend is configured.
        """
        try:
            import mlx_whisper  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-whisper is not available. Install the ASR dependency to enable transcription."
            ) from exc

        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=self._model_id)

        if not isinstance(result, dict) or "text" not in result:
            raise RuntimeError("mlx-whisper returned an unexpected result format.")

        meta: Dict[str, Any] = {
            "segments": result.get("segments"),
        }
        return str(result["text"]), meta
