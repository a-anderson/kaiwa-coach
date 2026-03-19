"""WebM/Opus → raw PCM conversion via ffmpeg subprocess.

Requires ffmpeg on PATH (``brew install ffmpeg`` on macOS).
``soundfile`` and ``scipy`` cannot decode WebM/Opus, so ffmpeg is the only
practical option for browser-recorded audio.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

from kaiwacoach.storage.blobs import AudioMeta


def webm_to_pcm(webm_bytes: bytes, target_sample_rate: int = 16000) -> tuple[bytes, AudioMeta]:
    """Convert a WebM/Opus browser recording to raw PCM + AudioMeta.

    Parameters
    ----------
    webm_bytes:
        Raw bytes of the WebM file (as received from the browser).
    target_sample_rate:
        Desired output sample rate in Hz. Use ``orchestrator.expected_sample_rate``
        rather than hardcoding, so the value stays in sync with the ASR model.

    Returns
    -------
    tuple[bytes, AudioMeta]
        Raw 16-bit signed-integer mono PCM and its metadata.

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg exits non-zero (bad input, ffmpeg not installed, etc.).
    """
    with tempfile.TemporaryDirectory() as tmp:
        webm_path = Path(tmp) / "input.webm"
        wav_path = Path(tmp) / "output.wav"
        webm_path.write_bytes(webm_bytes)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(webm_path),
                "-ar", str(target_sample_rate),
                "-ac", "1",            # mono
                "-sample_fmt", "s16",  # 16-bit signed int
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            pcm_bytes = wf.readframes(num_frames)
        duration = num_frames / sample_rate if sample_rate > 0 else None
    return pcm_bytes, AudioMeta(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        num_frames=num_frames,
        duration_seconds=duration,
    )
