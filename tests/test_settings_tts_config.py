"""Tests for nested TTS config structure (KokoroConfig + VoiceVoxConfig)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kaiwacoach.settings import load_config


# --- defaults ---


def test_tts_config_defaults(tmp_path: Path) -> None:
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.tts.kokoro.voice == "default"
    assert config.tts.kokoro.speed == 1.0
    assert config.tts.voicevox.url == "http://localhost:50021"
    assert config.tts.voicevox.speaker_id == 74
    assert config.tts.voicevox.speed == 1.0


# --- env var overrides ---


def test_kaiwacoach_tts_kokoro_voice_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_KOKORO_VOICE", "jf_alpha")
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.tts.kokoro.voice == "jf_alpha"


def test_kaiwacoach_tts_kokoro_speed_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_KOKORO_SPEED", "1.3")
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.tts.kokoro.speed == pytest.approx(1.3)


def test_kaiwacoach_tts_voicevox_url_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_VOICEVOX_URL", "http://localhost:9999")
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.tts.voicevox.url == "http://localhost:9999"


def test_kaiwacoach_tts_voicevox_speaker_id_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_VOICEVOX_SPEAKER_ID", "3")
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.tts.voicevox.speaker_id == 3


def test_kaiwacoach_tts_voicevox_speed_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_VOICEVOX_SPEED", "0.9")
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.tts.voicevox.speed == pytest.approx(0.9)


# --- validation ---


def test_kokoro_speed_zero_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_KOKORO_SPEED", "0")
    with pytest.raises(ValueError, match="tts.kokoro.speed"):
        load_config()


def test_kokoro_speed_negative_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_KOKORO_SPEED", "-0.5")
    with pytest.raises(ValueError, match="tts.kokoro.speed"):
        load_config()


def test_voicevox_speed_zero_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_VOICEVOX_SPEED", "0")
    with pytest.raises(ValueError, match="tts.voicevox.speed"):
        load_config()


def test_voicevox_speed_negative_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIWACOACH_TTS_VOICEVOX_SPEED", "-1.0")
    with pytest.raises(ValueError, match="tts.voicevox.speed"):
        load_config()


# --- to_dict round-trip ---


def test_tts_config_roundtrips_to_dict(tmp_path: Path) -> None:
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    d = config.to_dict()
    assert d["tts"]["kokoro"]["voice"] == "default"
    assert d["tts"]["kokoro"]["speed"] == 1.0
    assert d["tts"]["voicevox"]["url"] == "http://localhost:50021"
    assert d["tts"]["voicevox"]["speaker_id"] == 74
    assert d["tts"]["voicevox"]["speed"] == 1.0


# --- YAML file parsing ---


def test_tts_kokoro_speed_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tts:\n  kokoro:\n    speed: 1.5\n", encoding="utf-8")
    config = load_config(config_path=cfg_file)
    assert config.tts.kokoro.speed == pytest.approx(1.5)


def test_tts_kokoro_voice_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('tts:\n  kokoro:\n    voice: "jf_alpha"\n', encoding="utf-8")
    config = load_config(config_path=cfg_file)
    assert config.tts.kokoro.voice == "jf_alpha"


def test_tts_voicevox_speaker_id_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tts:\n  voicevox:\n    speaker_id: 3\n", encoding="utf-8")
    config = load_config(config_path=cfg_file)
    assert config.tts.voicevox.speaker_id == 3


def test_tts_voicevox_url_from_yaml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('tts:\n  voicevox:\n    url: "http://localhost:9999"\n', encoding="utf-8")
    config = load_config(config_path=cfg_file)
    assert config.tts.voicevox.url == "http://localhost:9999"


def test_tts_yaml_partial_override_preserves_other_defaults(tmp_path: Path) -> None:
    """Setting only tts.kokoro.speed in YAML should not clobber voicevox defaults."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("tts:\n  kokoro:\n    speed: 1.5\n", encoding="utf-8")
    config = load_config(config_path=cfg_file)
    assert config.tts.voicevox.speaker_id == 74
    assert config.tts.voicevox.url == "http://localhost:50021"


# --- stale key warnings ---


@pytest.mark.parametrize("stale_key", ["voice", "speed"])
def test_stale_flat_tts_key_emits_warning(
    stale_key: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Old flat tts.voice / tts.speed key in config.yaml should emit a deprecation warning."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"tts:\n  {stale_key}: value\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="kaiwacoach.settings"):
        load_config(config_path=cfg_file)
    assert stale_key in caplog.text


def test_stale_flat_tts_voice_key_does_not_override_kokoro_voice(tmp_path: Path) -> None:
    """Old flat tts.voice is silently ignored; tts.kokoro.voice stays at default."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('tts:\n  voice: "jf_alpha"\n', encoding="utf-8")
    config = load_config(config_path=cfg_file)
    assert config.tts.kokoro.voice == "default"


@pytest.mark.parametrize(
    "stale_var,replacement",
    [
        ("KAIWACOACH_TTS_VOICE", "KAIWACOACH_TTS_KOKORO_VOICE"),
        ("KAIWACOACH_TTS_SPEED", "KAIWACOACH_TTS_KOKORO_SPEED"),
    ],
)
def test_stale_tts_env_var_emits_warning(
    stale_var: str,
    replacement: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Old KAIWACOACH_TTS_VOICE / _TTS_SPEED env vars should emit a deprecation warning."""
    monkeypatch.setenv(stale_var, "value")
    with caplog.at_level(logging.WARNING, logger="kaiwacoach.settings"):
        load_config()
    assert stale_var in caplog.text
    assert replacement in caplog.text
