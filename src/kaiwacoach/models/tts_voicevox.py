"""VoiceVox TTS backend, wrapper, and language-dispatch router."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.parse
import urllib.request
import wave

from kaiwacoach.models.protocols import TTSProtocol, TTSResult
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache
from kaiwacoach.utils import BoundedDict

_log = logging.getLogger(__name__)

_VOICEVOX_CACHE_MAX = 512
_VOICEVOX_REQUEST_TIMEOUT = 30


class VoiceVoxBackend:
    """HTTP client for the VoiceVox local TTS server."""

    @staticmethod
    def check_available(url: str) -> bool:
        """Return True if the VoiceVox server is reachable at url, False otherwise."""
        try:
            with urllib.request.urlopen(f"{url}/version", timeout=2):
                pass
            return True
        except OSError:
            return False

    @staticmethod
    def synthesize(text: str, speaker_id: int, speed: float, url: str) -> tuple[bytes, AudioMeta]:
        """Synthesize speech and return raw PCM bytes with audio metadata.

        Parameters
        ----------
        text : str
            Text to synthesize.
        speaker_id : int
            VoiceVox speaker ID.
        speed : float
            Speech rate; written to the AudioQuery speedScale field.
        url : str
            Base URL of the VoiceVox HTTP server.
        """
        encoded = urllib.parse.quote(text, safe="")
        query_req = urllib.request.Request(
            f"{url}/audio_query?text={encoded}&speaker={speaker_id}",
            method="POST",
        )
        with urllib.request.urlopen(query_req, timeout=_VOICEVOX_REQUEST_TIMEOUT) as resp:
            audio_query = json.loads(resp.read().decode("utf-8"))

        audio_query["speedScale"] = speed

        body = json.dumps(audio_query).encode("utf-8")
        synth_req = urllib.request.Request(
            f"{url}/synthesis?speaker={speaker_id}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(synth_req, timeout=_VOICEVOX_REQUEST_TIMEOUT) as resp:
            wav_bytes = resp.read()

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as w:
            sample_rate = w.getframerate()
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            num_frames = w.getnframes()
            pcm_bytes = w.readframes(num_frames)

        return pcm_bytes, AudioMeta(sample_rate=sample_rate, channels=channels, sample_width=sample_width)


class VoiceVoxTTS:
    """TTS wrapper for VoiceVox with session-only cache."""

    def __init__(
        self,
        speaker_id: int,
        url: str,
        speed: float,
        cache: SessionAudioCache,
    ) -> None:
        self._speaker_id = speaker_id
        self._url = url
        self._speed = speed
        self._cache = cache
        self._cache_index: BoundedDict[str, str] = BoundedDict(maxsize=_VOICEVOX_CACHE_MAX)

    @property
    def model_id(self) -> str:
        return f"voicevox:speaker={self._speaker_id}"

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
        """Synthesize via VoiceVox. voice, speed, lang_code, and language params are ignored.

        voice and lang_code are not used by VoiceVox (speaker_id selects the voice).
        speed is set at construction time from config.tts.voicevox.speed.
        language is handled by LanguageDispatchTTS before this method is called.
        """
        cache_key = self._hash_key(text)
        cached_path = self._cache_index.get(cache_key)
        if cached_path:
            return TTSResult(
                audio_path=cached_path,
                meta={"model_id": self.model_id, "cache_hit": True, "cache_key": cache_key},
            )

        pcm_bytes, meta = VoiceVoxBackend.synthesize(
            text=text,
            speaker_id=self._speaker_id,
            speed=self._speed,
            url=self._url,
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
        return TTSResult(
            audio_path=audio_path_str,
            meta={"model_id": self.model_id, "cache_hit": False, "cache_key": cache_key},
        )

    def _hash_key(self, text: str) -> str:
        payload = f"voicevox|{text}|{self._speaker_id}|{self._speed}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class LanguageDispatchTTS:
    """Routes Japanese TTS to VoiceVox and all other languages to the default backend."""

    def __init__(self, japanese_tts: TTSProtocol, default_tts: TTSProtocol) -> None:
        self._japanese_tts = japanese_tts
        self._default_tts = default_tts

    @property
    def model_id(self) -> str:
        return f"{self._japanese_tts.model_id} (ja) / {self._default_tts.model_id} (other)"

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
        if language == "ja":
            try:
                return self._japanese_tts.synthesize(
                    conversation_id, turn_id, text, voice, speed,
                    lang_code=lang_code, language=language,
                )
            except OSError as exc:
                _log.warning(
                    "VoiceVox unavailable (%s); falling back to Kokoro — language=%r text=%r",
                    exc,
                    language,
                    text[:40],
                )
        return self._default_tts.synthesize(
            conversation_id, turn_id, text, voice, speed,
            lang_code=lang_code, language=language,
        )
