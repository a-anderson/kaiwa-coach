"""Unit tests for the WebM→PCM audio conversion utility.

These tests do not require ffmpeg or real WebM input — they verify the
WAV-parsing and AudioMeta assembly logic by feeding pre-built WAV files
into the conversion function via monkeypatching.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from kaiwacoach.api.audio_conversion import webm_to_pcm
from kaiwacoach.storage.blobs import AudioMeta


def _write_wav_bytes(
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    num_frames: int = 160,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * num_frames * channels * sample_width)
    return buf.getvalue()


def _fake_ffmpeg(wav_content: bytes):
    """Return a subprocess.run replacement that writes wav_content to the output path."""

    def fake_run(cmd, *, check, capture_output, **_kw):
        # The last positional arg in the ffmpeg command is the output path.
        out_path = Path(cmd[-1])
        out_path.write_bytes(wav_content)

    return fake_run


# ── Happy path ────────────────────────────────────────────────────────────────


def test_webm_to_pcm_returns_correct_sample_rate():
    wav = _write_wav_bytes(sample_rate=16000, num_frames=320)
    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=_fake_ffmpeg(wav)):
        pcm, meta = webm_to_pcm(b"fake-webm", target_sample_rate=16000)
    assert meta.sample_rate == 16000


def test_webm_to_pcm_returns_mono():
    wav = _write_wav_bytes(channels=1)
    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=_fake_ffmpeg(wav)):
        _, meta = webm_to_pcm(b"fake-webm")
    assert meta.channels == 1


def test_webm_to_pcm_returns_16bit():
    wav = _write_wav_bytes(sample_width=2)
    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=_fake_ffmpeg(wav)):
        _, meta = webm_to_pcm(b"fake-webm")
    assert meta.sample_width == 2


def test_webm_to_pcm_pcm_bytes_match_num_frames():
    num_frames = 480
    wav = _write_wav_bytes(num_frames=num_frames)
    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=_fake_ffmpeg(wav)):
        pcm, meta = webm_to_pcm(b"fake-webm")
    expected_bytes = num_frames * meta.channels * meta.sample_width
    assert len(pcm) == expected_bytes


def test_webm_to_pcm_duration_seconds_correct():
    wav = _write_wav_bytes(sample_rate=16000, num_frames=16000)  # 1 second
    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=_fake_ffmpeg(wav)):
        _, meta = webm_to_pcm(b"fake-webm")
    assert meta.duration_seconds == pytest.approx(1.0)


def test_webm_to_pcm_num_frames_populated():
    wav = _write_wav_bytes(num_frames=800)
    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=_fake_ffmpeg(wav)):
        _, meta = webm_to_pcm(b"fake-webm")
    assert meta.num_frames == 800


def test_webm_to_pcm_passes_target_sample_rate_to_ffmpeg():
    """ffmpeg must receive -ar <target_sample_rate> in the command."""
    wav = _write_wav_bytes(sample_rate=8000)
    captured_cmd = []

    def fake_run(cmd, *, check, capture_output, **_kw):
        captured_cmd.extend(cmd)
        Path(cmd[-1]).write_bytes(wav)

    with patch("kaiwacoach.api.audio_conversion.subprocess.run", side_effect=fake_run):
        webm_to_pcm(b"fake-webm", target_sample_rate=8000)

    assert "-ar" in captured_cmd
    ar_index = captured_cmd.index("-ar")
    assert captured_cmd[ar_index + 1] == "8000"


# ── Error path ────────────────────────────────────────────────────────────────


def test_webm_to_pcm_raises_on_ffmpeg_failure():
    """CalledProcessError from ffmpeg must propagate to the caller."""
    with patch(
        "kaiwacoach.api.audio_conversion.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            webm_to_pcm(b"garbage")
