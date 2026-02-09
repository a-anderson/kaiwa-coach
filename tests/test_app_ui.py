"""Tests for the minimal UI scaffold."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

import kaiwacoach.ui.gradio_app as ui_module
from kaiwacoach.storage.blobs import AudioMeta


def test_format_conversation_history_formats_turns() -> None:
    dict_history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good."},
    ]
    tuple_history = [("Hi", "Hello"), ("How are you?", "Good.")]
    formatted_dict = ui_module._format_conversation_history(dict_history)
    formatted_tuple = ui_module._format_conversation_history(tuple_history)
    expected = "User: Hi\nAssistant: Hello\nUser: How are you?\nAssistant: Good."
    assert formatted_dict == expected
    assert formatted_tuple == expected


def test_build_ui_constructs_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Blocks:
        def __init__(self, *args, **kwargs):
            self._entered = False

        def __enter__(self):
            self._entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Component:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def click(self, *args, **kwargs):
            return self

        def then(self, *args, **kwargs):
            return self

        def submit(self, *args, **kwargs):
            return self

    fake_gr = SimpleNamespace(
        Blocks=_Blocks,
        Markdown=_Component,
        HTML=_Component,
        Chatbot=_Component,
        Textbox=_Component,
        Audio=_Component,
        Image=_Component,
        Button=_Component,
        Dataframe=_Component,
        Row=_Blocks,
        Column=_Blocks,
        State=_Component,
    )

    monkeypatch.setitem(__import__("sys").modules, "gradio", fake_gr)
    demo = ui_module.build_ui(orchestrator=SimpleNamespace())
    assert isinstance(demo, _Blocks)


def test_audio_to_pcm_from_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gradio supplies microphone audio as (sample_rate, numpy_array)."""
    import numpy as np

    audio = (16000, np.array([0.0, 0.5, -0.5], dtype=np.float32))
    pcm_bytes, meta = ui_module._audio_to_pcm(audio)

    assert isinstance(pcm_bytes, bytes)
    assert meta == AudioMeta(sample_rate=16000, channels=1, sample_width=2)


def test_audio_to_pcm_from_wave_file(tmp_path: Path) -> None:
    """Gradio supplies uploaded audio as a file path (str/Path)."""
    import wave

    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 10)

    pcm_bytes, meta = ui_module._audio_to_pcm(path)

    assert len(pcm_bytes) == 20
    assert meta == AudioMeta(sample_rate=16000, channels=1, sample_width=2)


def test_audio_to_pcm_rejects_missing_audio() -> None:
    with pytest.raises(ValueError, match="No audio input provided"):
        ui_module._audio_to_pcm(None)


def test_handle_text_turn_sets_assistant_audio_only(tmp_path: Path) -> None:
    class _Orchestrator:
        def __init__(self):
            self.created = False

        def create_conversation(self):
            self.created = True
            return "conv"

        def process_text_turn(self, conversation_id: str, user_text: str, conversation_history: str):
            return SimpleNamespace(reply_text="ok", tts_audio_path="/tmp/tts.wav", user_turn_id="turn-1")

        def get_latest_corrections(self, user_turn_id: str):
            return {"errors": ["err"], "corrected": "fixed", "native": "native", "explanation": "explain"}

    orch = _Orchestrator()
    (
        history,
        conversation_id,
        _,
        error_update,
        user_audio,
        assistant_audio,
        corrected,
        native,
        explanation,
    ) = ui_module._handle_text_turn(orch, "hello", [], None)

    assert conversation_id == "conv"
    assert user_audio is None
    assert assistant_audio == "/tmp/tts.wav"
    assert error_update == {"visible": False}
    assert history[-1]["role"] == "assistant"
    assert corrected == "fixed"
    assert native == "native"
    assert explanation == "explain"


def test_start_text_turn_sets_placeholder_and_states() -> None:
    class _Orchestrator:
        def create_conversation(self):
            return "conv"

        def persist_user_text_turn(
            self, conversation_id: str, user_text: str, timings: dict | None = None
        ):
            return "turn-1"

    orch = _Orchestrator()
    (
        history,
        conversation_id,
        _cleared_input,
        user_turn_id,
        display_history,
        conversation_history,
        user_text,
        error_update,
        skip_pipeline,
        timings,
    ) = ui_module._start_text_turn(orch, "hello", [], None)

    assert conversation_id == "conv"
    assert user_turn_id == "turn-1"
    assert user_text == "hello"
    assert history[-1]["content"] == "hello"
    assert display_history[-1]["content"] == "hello"
    assert "User: " not in conversation_history
    assert error_update == {"visible": False}
    assert skip_pipeline is False
    assert isinstance(timings, dict)


def test_handle_audio_turn_sets_both_audio_paths(tmp_path: Path) -> None:
    class _Orchestrator:
        def __init__(self):
            self.created = False

        def create_conversation(self):
            self.created = True
            return "conv"

        def process_audio_turn(self, conversation_id: str, pcm_bytes: bytes, audio_meta: AudioMeta, conversation_history: str):
            return SimpleNamespace(
                asr_text="konnichiwa",
                reply_text="hello",
                input_audio_path="/tmp/user.wav",
                tts_audio_path="/tmp/tts.wav",
                user_turn_id="turn-1",
            )

        def get_latest_corrections(self, user_turn_id: str):
            return {"errors": ["err"], "corrected": "fixed", "native": "native", "explanation": "explain"}
    orch = _Orchestrator()
    audio = (16000, [0.0, 0.5, -0.5])
    (
        history,
        conversation_id,
        _,
        error_update,
        user_audio,
        assistant_audio,
        corrected,
        native,
        explanation,
    ) = ui_module._handle_audio_turn(orch, audio, [], None)

    assert conversation_id == "conv"
    assert user_audio == "/tmp/user.wav"
    assert assistant_audio == "/tmp/tts.wav"
    assert error_update == {"visible": False}
    assert history[-1]["role"] == "assistant"
    assert corrected == "fixed"
    assert native == "native"
    assert explanation == "explain"


def test_start_audio_turn_returns_asr_text(tmp_path: Path) -> None:
    class _Orchestrator:
        def create_conversation(self):
            return "conv"

        def prepare_audio_turn(
            self,
            conversation_id: str,
            pcm_bytes: bytes,
            audio_meta: AudioMeta,
            timings: dict | None = None,
        ):
            return SimpleNamespace(
                user_turn_id="turn-1",
                input_audio_path="/tmp/user.wav",
                asr_text="konnichiwa",
            )

    orch = _Orchestrator()
    audio = (16000, [0.0, 0.5, -0.5])
    (
        history,
        conversation_id,
        _cleared_input,
        user_turn_id,
        _display_history,
        _conversation_history,
        user_text,
        error_update,
        user_audio_path,
        skip_pipeline,
        timings,
    ) = ui_module._start_audio_turn(orch, audio, [], None)

    assert conversation_id == "conv"
    assert user_turn_id == "turn-1"
    assert user_text == "konnichiwa"
    assert user_audio_path == "/tmp/user.wav"
    assert history[-1]["content"] == "konnichiwa"
    assert error_update == {"visible": False}
    assert skip_pipeline is False
    assert isinstance(timings, dict)


def test_run_llm_reply_updates_history() -> None:
    class _Orchestrator:
        def generate_reply(
            self,
            conversation_id: str,
            user_turn_id: str,
            user_text: str,
            conversation_history: str,
            timings: dict | None = None,
        ):
            return "assistant-1", "hello"

    orch = _Orchestrator()
    chat_history = [{"role": "assistant", "content": "(Thinking...)"}]
    updated, assistant_turn_id, reply_text, error_update = ui_module._run_llm_reply(
        orch, "conv", "turn-1", "hello", "", chat_history, False
    )

    assert updated[-1]["content"] == "hello"
    assert assistant_turn_id == "assistant-1"
    assert reply_text == "hello"
    assert error_update == {"visible": False}


def test_run_corrections_noop_on_skip() -> None:
    class _Orchestrator:
        def run_corrections(self, user_turn_id: str, user_text: str, assistant_turn_id: str | None = None):
            return {"errors": ["x"], "corrected": "c", "native": "n", "explanation": "e"}

    orch = _Orchestrator()
    result = ui_module._run_corrections(orch, None, "hello", None, True)
    assert result == ("", "", "")


def test_run_tts_noop_on_skip() -> None:
    class _Orchestrator:
        def run_tts(self, conversation_id: str, assistant_turn_id: str, reply_text: str):
            return SimpleNamespace(audio_path="/tmp/tts.wav")

    orch = _Orchestrator()
    assert ui_module._run_tts(orch, None, None, "ok", True) is None


def test_handle_reset_clears_state() -> None:
    class _Orchestrator:
        def __init__(self):
            self.reset_called = False

        def reset_session(self):
            self.reset_called = True

    orch = _Orchestrator()
    result = ui_module._handle_reset(orch)
    assert orch.reset_called is True
    assert result[0] == []
    assert result[1] is None
