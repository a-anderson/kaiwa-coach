"""Tests for session-only audio cache utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Audio saved to the cache should be readable with matching metadata."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    pcm_bytes = b"\x00\x01" * 320  # 640 bytes, aligned to 2-byte frames
    meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)

    path = cache.save_audio("conv_1", "turn_1", "assistant", pcm_bytes, meta)
    loaded_bytes, loaded_meta = cache.load_audio(path)

    assert loaded_bytes == pcm_bytes
    assert loaded_meta.sample_rate == meta.sample_rate
    assert loaded_meta.channels == meta.channels
    assert loaded_meta.sample_width == meta.sample_width
    assert loaded_meta.num_frames is not None
    assert loaded_meta.duration_seconds is not None


def test_deterministic_paths_use_hash(tmp_path: Path) -> None:
    """Identical audio bytes should map to the same deterministic filename."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    pcm_bytes = b"\x00\x01" * 100
    meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)

    path_a = cache.save_audio("conv_a", "turn_a", "assistant", pcm_bytes, meta)
    path_b = cache.save_audio("conv_a", "turn_a", "assistant", pcm_bytes, meta)

    assert path_a.name == path_b.name


def test_sample_rate_validation(tmp_path: Path) -> None:
    """Mismatched sample rates should raise."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    meta = AudioMeta(sample_rate=44100, channels=1, sample_width=2)

    with pytest.raises(ValueError, match="Unexpected sample rate"):
        cache.save_audio("conv_1", "turn_1", "user", b"\x00\x01" * 100, meta)


def test_sample_rate_validation_skipped_for_user_raw(tmp_path: Path) -> None:
    """User raw audio should bypass expected sample-rate validation."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    meta = AudioMeta(sample_rate=44100, channels=1, sample_width=2)

    path = cache.save_audio("conv_1", "turn_1", "user_raw", b"\x00\x01" * 100, meta)
    assert path.exists()


def test_sample_rate_validation_skipped_for_tts(tmp_path: Path) -> None:
    """Non-user audio should not enforce the expected sample rate."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    meta = AudioMeta(sample_rate=44100, channels=1, sample_width=2)

    path = cache.save_audio("conv_1", "turn_1", "tts", b"\x00\x01" * 100, meta)
    assert path.exists()


def test_pcm_alignment_validation(tmp_path: Path) -> None:
    """PCM byte length must align with channels * sample_width."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    meta = AudioMeta(sample_rate=16000, channels=2, sample_width=2)
    pcm_bytes = b"\x00\x01\x02"  # 3 bytes, not aligned to 4-byte frames

    with pytest.raises(ValueError, match="PCM byte length is not aligned"):
        cache.save_audio("conv_1", "turn_1", "user", pcm_bytes, meta)


def test_cleanup_removes_cache_dir(tmp_path: Path) -> None:
    """cleanup should remove the cache directory and its contents."""
    cache = SessionAudioCache(root_dir=tmp_path / "audio-cache")
    meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)
    cache.save_audio("conv_1", "turn_1", "tts", b"\x00\x01" * 100, meta)

    cache.cleanup()
    assert not cache.root_dir.exists()


def test_invalid_tokens_are_rejected(tmp_path: Path) -> None:
    """Invalid identifiers should raise to prevent path traversal."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000)
    meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)

    with pytest.raises(ValueError, match="conversation_id contains invalid character"):
        cache.save_audio("conv/1", "turn_1", "tts", b"\x00\x01" * 10, meta)

    with pytest.raises(ValueError, match="turn_id contains invalid character"):
        cache.save_audio("conv_1", "turn 1", "tts", b"\x00\x01" * 10, meta)

    with pytest.raises(ValueError, match="kind contains invalid character"):
        cache.save_audio("conv_1", "turn_1", "t ts", b"\x00\x01" * 10, meta)


def test_cache_limit_eviction(tmp_path: Path) -> None:
    """Cache should evict oldest files when size exceeds max_cache_bytes."""
    cache = SessionAudioCache(root_dir=tmp_path, expected_sample_rate=16000, max_cache_bytes=500)
    meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)

    first = cache.save_audio("conv_1", "turn_1", "tts", b"\x00\x01" * 200, meta)
    second = cache.save_audio("conv_1", "turn_2", "tts", b"\x00\x01" * 200, meta)

    remaining = [path for path in (first, second) if path.exists()]
    assert len(remaining) == 1
