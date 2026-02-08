"""Session-only audio cache utilities for WAV blobs."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AudioMeta:
    sample_rate: int
    channels: int
    sample_width: int
    num_frames: int | None = None
    duration_seconds: float | None = None


class SessionAudioCache:
    """Session-scoped WAV cache with deterministic paths.

    Notes
    -----
    If `max_cache_bytes` is set, the cache evicts the oldest WAV files by
    modification time until the size is under the limit. This is a
    best-effort policy and not a strict LRU implementation.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        expected_sample_rate: int | None = 16000,
        max_cache_bytes: int | None = None,
    ) -> None:
        """Initialize the session audio cache.

        Parameters
        ----------
        root_dir : str | pathlib.Path | None
            Root directory for the session cache. If None, a temp directory is created.
        expected_sample_rate : int | None
            Expected sample rate for all cached audio. If None, accept any rate.
        max_cache_bytes : int | None
            Optional maximum size for the session cache in bytes.

        Returns
        -------
        None
        """
        self._root_dir = Path(root_dir) if root_dir else Path(tempfile.mkdtemp(prefix="kaiwacoach-audio-"))
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._expected_sample_rate = expected_sample_rate
        self._max_cache_bytes = max_cache_bytes

    @property
    def root_dir(self) -> Path:
        """Return the root directory for this session cache.

        Parameters
        ----------
        None

        Returns
        -------
        pathlib.Path
            Root directory path.
        """
        return self._root_dir

    def cleanup(self) -> None:
        """Delete the session cache directory and all audio blobs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._root_dir.exists():
            shutil.rmtree(self._root_dir)

    def save_audio(
        self,
        conversation_id: str,
        turn_id: str,
        kind: str,
        pcm_bytes: bytes,
        meta: AudioMeta,
    ) -> Path:
        """Save PCM bytes as a WAV file in the session cache.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        turn_id : str
            Turn identifier.
        kind : str
            Audio kind (e.g., user, assistant, tts).
        pcm_bytes : bytes
            Raw PCM bytes to write.
        meta : AudioMeta
            Audio metadata to encode.

        Returns
        -------
        pathlib.Path
            Path to the cached WAV file.

        Raises
        ------
        ValueError
            If sample rate or PCM alignment validation fails.
        """
        self._validate_token("conversation_id", conversation_id)
        self._validate_token("turn_id", turn_id)
        self._validate_token("kind", kind)
        self._validate_sample_rate(meta.sample_rate, kind)
        self._validate_pcm(pcm_bytes, meta)
        blob_hash = self._hash_bytes(pcm_bytes)
        path = self._audio_path(conversation_id, turn_id, kind, blob_hash)
        self._write_wav_atomic(path, pcm_bytes, meta)
        self._enforce_cache_limit()
        return path

    def load_audio(self, path: str | Path) -> Tuple[bytes, AudioMeta]:
        """Load PCM bytes and metadata from a WAV file.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to the WAV file.

        Returns
        -------
        tuple[bytes, AudioMeta]
            Raw PCM bytes and associated metadata.
        """
        return self._read_wav(Path(path))

    def _validate_sample_rate(self, sample_rate: int, kind: str) -> None:
        """Validate that the sample rate matches the expected value for user audio.

        Parameters
        ----------
        sample_rate : int
            Sample rate to validate.
        kind : str
            Audio kind (e.g., user, assistant, tts).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the sample rate does not match the expected value for user audio.
        """
        if kind != "user":
            return
        if self._expected_sample_rate is None:
            return
        if sample_rate != self._expected_sample_rate:
            raise ValueError(
                f"Unexpected sample rate {sample_rate}. Expected {self._expected_sample_rate}."
            )

    def _validate_pcm(self, pcm_bytes: bytes, meta: AudioMeta) -> None:
        """Validate PCM byte alignment against channel/sample width metadata.

        Parameters
        ----------
        pcm_bytes : bytes
            Raw PCM bytes to validate.
        meta : AudioMeta
            Audio metadata describing channel count and sample width.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the PCM bytes are not aligned to the frame size.
        """
        frame_size = meta.channels * meta.sample_width
        if frame_size <= 0:
            raise ValueError("channels and sample_width must be positive.")
        if len(pcm_bytes) % frame_size != 0:
            raise ValueError("PCM byte length is not aligned to frame size.")

    def _audio_path(self, conversation_id: str, turn_id: str, kind: str, blob_hash: str) -> Path:
        """Build a deterministic cache path for an audio blob.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        turn_id : str
            Turn identifier.
        kind : str
            Audio kind (e.g., user, assistant, tts).
        blob_hash : str
            SHA-256 hash of the PCM bytes.

        Returns
        -------
        pathlib.Path
            Full path to the cached WAV file.
        """
        directory = self._root_dir / conversation_id / turn_id
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{kind}_{blob_hash}.wav"
        return directory / filename

    @staticmethod
    def _validate_token(label: str, value: str) -> None:
        """Validate identifier tokens to prevent unsafe paths.

        Parameters
        ----------
        label : str
            Field name for error reporting.
        value : str
            Token value to validate.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the token is empty or contains invalid characters.
        """
        if not value:
            raise ValueError(f"{label} must not be empty.")
        for char in value:
            if not (char.isalnum() or char in {"-", "_"}):
                raise ValueError(
                    f"{label} contains invalid character '{char}'. Use letters, numbers, '-' or '_'."
                )

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        """Compute a SHA-256 hash for the given bytes.

        Parameters
        ----------
        data : bytes
            Bytes to hash.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _write_wav(path: Path, pcm_bytes: bytes, meta: AudioMeta) -> None:
        """Write PCM bytes to a WAV file.

        Parameters
        ----------
        path : pathlib.Path
            Destination file path.
        pcm_bytes : bytes
            Raw PCM bytes to write.
        meta : AudioMeta
            Audio metadata to encode.

        Returns
        -------
        None
        """
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(meta.channels)
            handle.setsampwidth(meta.sample_width)
            handle.setframerate(meta.sample_rate)
            handle.writeframes(pcm_bytes)

    def _write_wav_atomic(self, path: Path, pcm_bytes: bytes, meta: AudioMeta) -> None:
        """Write PCM bytes to a WAV file atomically.

        Parameters
        ----------
        path : pathlib.Path
            Destination file path.
        pcm_bytes : bytes
            Raw PCM bytes to write.
        meta : AudioMeta
            Audio metadata to encode.

        Returns
        -------
        None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=path.stem,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            self._write_wav(temp_path, pcm_bytes, meta)
            temp_path.replace(path)
        finally:
            if temp_path.exists() and temp_path != path:
                temp_path.unlink(missing_ok=True)

    def _enforce_cache_limit(self) -> None:
        """Enforce the optional max cache size by evicting oldest files.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a single audio blob exceeds the max cache size.
        """
        if self._max_cache_bytes is None:
            return

        wav_files = list(self._root_dir.rglob("*.wav"))
        total_bytes = sum(path.stat().st_size for path in wav_files)
        if not wav_files:
            return
        largest_file = max(wav_files, key=lambda path: path.stat().st_size)
        if largest_file.stat().st_size > self._max_cache_bytes:
            largest_file.unlink(missing_ok=True)
            raise ValueError("Audio blob exceeds max_cache_bytes.")

        if total_bytes <= self._max_cache_bytes:
            return

        wav_files.sort(key=lambda path: (path.stat().st_mtime, path.name))
        for path in wav_files:
            if total_bytes <= self._max_cache_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total_bytes -= size

    @staticmethod
    def _read_wav(path: Path) -> Tuple[bytes, AudioMeta]:
        """Read PCM bytes and metadata from a WAV file.

        Parameters
        ----------
        path : pathlib.Path
            WAV file path.

        Returns
        -------
        tuple[bytes, AudioMeta]
            Raw PCM bytes and associated metadata.
        """
        with wave.open(str(path), "rb") as handle:
            num_frames = handle.getnframes()
            meta = AudioMeta(
                sample_rate=handle.getframerate(),
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                num_frames=num_frames,
                duration_seconds=num_frames / handle.getframerate() if handle.getframerate() else None,
            )
            pcm_bytes = handle.readframes(num_frames)
        return pcm_bytes, meta
