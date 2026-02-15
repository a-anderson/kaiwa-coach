"""Gradio UI wiring and handlers.

Notes
-----
- UI callbacks call the orchestrator only. No direct DB access happens here.
- The orchestrator uses SQLiteWriter's single-writer queue plus short-lived read
  connections, which is safe for Gradio's concurrent callback execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import time
from datetime import datetime, timezone
from string import Template

from kaiwacoach.constants import SUPPORTED_LANGUAGES
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.storage.blobs import AudioMeta

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextTurnStartResult:
    chat_history: list[dict[str, str]] | list[tuple[str, str]]
    conversation_id: str | None
    cleared_input: str
    user_turn_id: str | None
    history_state: list[dict[str, str]] | list[tuple[str, str]]
    conversation_history: str
    user_text: str
    error_update: dict
    skip_pipeline: bool
    timings: dict


@dataclass(frozen=True)
class AudioTurnStartResult:
    chat_history: list[dict[str, str]] | list[tuple[str, str]]
    conversation_id: str | None
    cleared_input: str
    user_turn_id: str | None
    history_state: list[dict[str, str]] | list[tuple[str, str]]
    conversation_history: str
    user_text: str
    error_update: dict
    user_audio_path: str | None
    skip_pipeline: bool
    timings: dict


@dataclass(frozen=True)
class LlmReplyResult:
    chat_history: list[dict[str, str]]
    assistant_turn_id: str | None
    reply_text: str
    error_update: dict


_LANGUAGE_THEMES: dict[str, dict[str, str]] = {
    "ja": {
        "primary": "#3a3a3a",
        "user": "#fce8e8",
        "bot": "#f1f1f1",
        "checkbox": "#ce2037",
    },
    "fr": {
        "primary": "#264db6",
        "user": "#e9eef9",
        "bot": "#fce8e8",
        "checkbox": "#264db6",
    },
    "it": {
        "primary": "#166534",
        "user": "#e9f6ee",
        "bot": "#fce8e8",
        "checkbox": "#166534",
    },
    "es": {
        "primary": "#e27a2b",
        "user": "#fff2e0",
        "bot": "#fce8e8",
        "checkbox": "#e27a2b",
    },
    "pt-br": {
        "primary": "#166534",
        "user": "#e7f5ec",
        "bot": "#e9eef9",
        "checkbox": "#166534",
    },
    "en": {
        "primary": "#264db6",
        "user": "#e9eef9",
        "bot": "#fce8e8",
        "checkbox": "#264db6",
    },
}

_THEME_STYLE_TEMPLATE = Template(
    """
<style id="kc-theme-style">
:root, body, .gradio-container {
  --kc-primary: $primary;
  --kc-checkbox: $checkbox;
  --kc-user-bg: $user_bg;
  --kc-bot-bg: $bot_bg;
  --kc-audio-active: #3a3a3a;
  --kc-audio-inactive: #9ca3af;
  --color-accent: $checkbox !important;
  --color-accent-soft: $checkbox !important;
  --color-accent-subtle: $checkbox !important;
  --color-accent-text: #ffffff !important;
  --color-accent-hover: $checkbox !important;
}
.gradio-container button,
.gradio-container .gr-button {
  border-color: var(--kc-user-bg) !important;
}
.gradio-container button.primary,
.gradio-container .gr-button.gr-button-primary {
  background: var(--kc-user-bg) !important;
  color: #111827 !important;
  box-shadow: none !important;
}
.gradio-container .message.user {
  background: var(--kc-user-bg) !important;
  color: #111827 !important;
  border: 1px solid var(--kc-user-bg) !important;
}
.gradio-container .message.bot {
  background: var(--kc-bot-bg) !important;
  color: #111827 !important;
  border: 1px solid var(--kc-bot-bg) !important;
}
#mic-input,
#last-user-audio,
#last-assistant-audio {
  --color-accent: var(--kc-audio-active) !important;
  --color-accent-soft: var(--kc-audio-active) !important;
  --color-accent-subtle: var(--kc-audio-active) !important;
  --waveform-color: var(--kc-audio-inactive);
  --waveform-progress-color: var(--kc-audio-active);
}
#corrections-toggle input[data-testid="checkbox"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  width: 1rem !important;
  height: 1rem !important;
  margin: 0 !important;
  border: 2px solid var(--kc-checkbox) !important;
  border-radius: 0.25rem !important;
  background: #ffffff !important;
  display: inline-grid !important;
  place-content: center !important;
}
#corrections-toggle input[data-testid="checkbox"]::after {
  content: "" !important;
  width: 0.28rem !important;
  height: 0.52rem !important;
  border: solid #ffffff !important;
  border-width: 0 2px 2px 0 !important;
  transform: rotate(45deg) scale(0) !important;
  transform-origin: center !important;
}
#corrections-toggle input[data-testid="checkbox"]:checked {
  background: var(--kc-checkbox) !important;
  border-color: var(--kc-checkbox) !important;
}
#corrections-toggle input[data-testid="checkbox"]:checked::after {
  transform: rotate(45deg) scale(1) !important;
}
.gradio-container .message.user,
.gradio-container .message.bot {
  box-shadow: none !important;
}
.gradio-container button.secondary,
.gradio-container .gr-button.gr-button-secondary {
  color: #111827 !important;
  background: var(--kc-user-bg) !important;
  border-color: var(--kc-user-bg) !important;
  box-shadow: none !important;
}
</style>
"""
)


def _theme_config(language: str) -> dict[str, str]:
    return _LANGUAGE_THEMES.get(language, _LANGUAGE_THEMES["en"])


def _waveform_options(language: str) -> dict[str, str]:
    return {
        "waveform_color": "#9ca3af",
        "waveform_progress_color": "#3a3a3a",
        "trim_region_color": "#3a3a3a",
    }


def _theme_updates_for_language(language: str):
    import gradio as gr  # type: ignore

    opts = _waveform_options(language)
    return (
        _theme_html(language),
        gr.Audio(
            value=None,
            waveform_options=opts,
            elem_id="mic-input",
            key=f"mic-input-{language}",
            sources=["microphone"],
            label="Microphone",
        ),
        gr.Audio(
            value=None,
            waveform_options=opts,
            elem_id="last-user-audio",
            key=f"user-audio-{language}",
            interactive=False,
            label="Last user audio",
        ),
        gr.Audio(
            value=None,
            waveform_options=opts,
            elem_id="last-assistant-audio",
            key=f"assistant-audio-{language}",
            interactive=False,
            autoplay=True,
            label="Last assistant audio",
        ),
    )


def _format_conversation_history(chat_history: list[dict[str, str]] | list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for message in chat_history:
        if isinstance(message, tuple) and len(message) == 2:
            user_text, assistant_text = message
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {assistant_text}")
            continue
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def _audio_to_pcm(audio) -> tuple[bytes, AudioMeta]:
    """Normalize Gradio audio input into PCM bytes + metadata.

    Notes
    -----
    Gradio may supply audio as either:
    - a tuple of ``(sample_rate, numpy_array)`` when using microphone input
    - a file path (str/Path) when using upload or cached audio
    """
    if audio is None:
        raise ValueError("No audio input provided.")
    if isinstance(audio, dict):
        if "path" in audio:
            return _audio_to_pcm(audio["path"])
        if "sample_rate" in audio and "data" in audio:
            return _audio_to_pcm((audio["sample_rate"], audio["data"]))
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise ValueError(f"Audio file not found: {path}")
        import wave

        with wave.open(str(path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            meta = AudioMeta(
                sample_rate=wav.getframerate(),
                channels=wav.getnchannels(),
                sample_width=wav.getsampwidth(),
            )
        return frames, meta
    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, data = audio
        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ValueError("Numpy is required to process array audio inputs.") from exc
        arr = np.asarray(data)
        if arr.ndim == 1:
            channels = 1
        else:
            channels = arr.shape[1]
            arr = arr.reshape(-1)
        if arr.dtype != np.int16:
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767).astype(np.int16)
        meta = AudioMeta(sample_rate=int(sample_rate), channels=channels, sample_width=2)
        return arr.tobytes(), meta
    raise ValueError("Unsupported audio input type.")


def _resample_pcm_bytes(
    pcm_bytes: bytes,
    meta: AudioMeta,
    target_sample_rate: int,
) -> tuple[bytes, AudioMeta]:
    # NOTE: This is a simple linear resampler for MVP correctness. It is not
    # band-limited and may introduce minor artifacts compared to a higher
    # quality resampling library.
    if meta.sample_rate == target_sample_rate:
        return pcm_bytes, meta
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError("Numpy is required to resample audio inputs.") from exc

    frame_size = meta.channels * meta.sample_width
    if frame_size <= 0:
        raise ValueError("channels and sample_width must be positive.")
    frame_count = len(pcm_bytes) // frame_size
    if frame_count == 0:
        raise ValueError("Audio input contains no frames.")

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if meta.channels > 1:
        samples = samples.reshape(frame_count, meta.channels)

    new_length = int(round(frame_count * (target_sample_rate / meta.sample_rate)))
    if new_length <= 0:
        raise ValueError("Resampled audio would have no frames.")

    x_old = np.linspace(0, frame_count - 1, num=frame_count)
    x_new = np.linspace(0, frame_count - 1, num=new_length)

    if meta.channels > 1:
        resampled_channels = [
            np.interp(x_new, x_old, samples[:, idx]).astype(np.float32)
            for idx in range(meta.channels)
        ]
        resampled = np.stack(resampled_channels, axis=1)
    else:
        resampled = np.interp(x_new, x_old, samples).astype(np.float32)

    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    return resampled.tobytes(), AudioMeta(
        sample_rate=target_sample_rate,
        channels=meta.channels,
        sample_width=meta.sample_width,
    )


def _audio_to_pcm_with_raw(
    audio,
    target_sample_rate: int | None,
) -> tuple[bytes, AudioMeta, bytes | None, AudioMeta | None]:
    pcm_bytes, meta = _audio_to_pcm(audio)
    if target_sample_rate is None or meta.sample_rate == target_sample_rate:
        return pcm_bytes, meta, None, None

    raw_pcm_bytes, raw_meta = pcm_bytes, meta
    resampled_bytes, resampled_meta = _resample_pcm_bytes(raw_pcm_bytes, raw_meta, target_sample_rate)
    _logger.warning(
        "audio_turn.resampled_input: %sHz -> %sHz",
        raw_meta.sample_rate,
        resampled_meta.sample_rate,
    )
    return resampled_bytes, resampled_meta, raw_pcm_bytes, raw_meta


def _normalize_history(chat_history: list[dict[str, str]] | list[tuple[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in chat_history:
        if isinstance(message, dict):
            normalized.append({"role": message.get("role", ""), "content": message.get("content", "")})
        elif isinstance(message, tuple) and len(message) == 2:
            normalized.append({"role": "user", "content": message[0]})
            normalized.append({"role": "assistant", "content": message[1]})
    return normalized


def _replace_last_assistant(
    history: list[dict[str, str]],
    reply_text: str,
) -> list[dict[str, str]]:
    if not history:
        return history
    updated = list(history)
    if updated[-1].get("role") == "assistant":
        updated[-1] = {"role": "assistant", "content": reply_text}
    else:
        updated.append({"role": "assistant", "content": reply_text})
    return updated


def _format_turns_to_chat(turns: list[dict[str, str | None]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for turn in turns:
        user_text = turn.get("input_text") or turn.get("asr_text") or ""
        if user_text:
            history.append({"role": "user", "content": str(user_text)})
        reply_text = turn.get("reply_text") or ""
        if reply_text:
            history.append({"role": "assistant", "content": str(reply_text)})
    return history


def _conversation_label(row: dict[str, str]) -> str:
    title = row.get("title") or "Untitled"
    preview = row.get("preview_text") or ""
    language = row.get("language") or "unknown"
    updated_at = row.get("updated_at") or ""
    emoji = _language_emoji(language)
    summary = _truncate_preview(preview) if preview else title
    if updated_at:
        return f"{emoji} {summary} · {_format_local_time(updated_at)}"
    return f"{emoji} {summary}"


def _truncate_preview(text: str, max_chars: int = 60) -> str:
    trimmed = text.strip()
    if len(trimmed) <= max_chars:
        return trimmed
    return f"{trimmed[: max_chars - 1].rstrip()}…"


def _language_emoji(language: str) -> str:
    return {
        "ja": "🇯🇵",
        "fr": "🇫🇷",
        "en": "🇺🇸",
        "es": "🇪🇸",
        "it": "🇮🇹",
        "pt-br": "🇧🇷",
    }.get(language, "🏳️")


def _format_local_time(timestamp: str) -> str:
    """Format timestamps as HH:mm in the local timezone."""
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        try:
            parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return timestamp
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone()
    tz_name = local.tzname() or ""
    suffix = f" {tz_name}" if tz_name else ""
    return local.strftime("%H:%M") + suffix


def _theme_html(language: str) -> str:
    cfg = _theme_config(language)
    return _THEME_STYLE_TEMPLATE.substitute(
        primary=cfg["primary"],
        checkbox=cfg["checkbox"],
        user_bg=cfg["user"],
        bot_bg=cfg["bot"],
    )


def _load_conversation_options(orchestrator: ConversationOrchestrator) -> list[tuple[str, str]]:
    list_fn = getattr(orchestrator, "list_conversations", None)
    if not callable(list_fn):
        return []
    rows = list_fn()
    return [(_conversation_label(row), row["id"]) for row in rows]


def _refresh_conversation_options(orchestrator: ConversationOrchestrator):
    import gradio as gr  # type: ignore

    choices = _load_conversation_options(orchestrator)
    return (
        gr.update(choices=choices, value=None),
        gr.update(interactive=bool(choices)),
        gr.update(visible=not bool(choices)),
    )


def _delete_conversation_and_refresh(
    orchestrator: ConversationOrchestrator,
    conversation_id: str | None,
):
    if conversation_id:
        orchestrator.delete_conversation(conversation_id)
    reset = _handle_reset(orchestrator)
    convo_updates = _refresh_conversation_options(orchestrator)
    return reset + convo_updates


def _delete_all_conversations_and_refresh(
    orchestrator: ConversationOrchestrator,
):
    orchestrator.delete_all_conversations()
    reset = _handle_reset(orchestrator)
    convo_updates = _refresh_conversation_options(orchestrator)
    return reset + convo_updates


def _confirm_row_show_updates():
    import gradio as gr  # type: ignore

    return (gr.update(visible=True), gr.update(visible=True))


def _confirm_row_hide_updates():
    import gradio as gr  # type: ignore

    return (gr.update(visible=False), gr.update(visible=False))


def _load_conversation(
    orchestrator: ConversationOrchestrator,
    conversation_id: str | None,
):
    if not conversation_id:
        return (
            [],
            None,
            [],
            "",
            "",
            "",
            None,
            None,
            False,
            {},
            "",
            "",
            "",
            None,
            None,
            "",
            None,
            True,
        )
    data = orchestrator.get_conversation(conversation_id)
    language = data.get("language")
    if language:
        orchestrator.set_language(language)
    history = _format_turns_to_chat(data["turns"])
    conversation_history = _format_conversation_history(history)
    return (
        history,
        conversation_id,
        history,
        conversation_history,
        "",
        "",
        None,
        None,
        False,
        {},
        "",
        "",
        "",
        None,
        None,
        "",
        language,
        True,
    )


def _start_text_turn(
    orchestrator: ConversationOrchestrator,
    user_text: str,
    chat_history: list[dict[str, str]] | list[tuple[str, str]],
    conversation_id: str | None,
    timings: dict | None = None,
)-> TextTurnStartResult:
    if not user_text:
        return TextTurnStartResult(
            chat_history=chat_history,
            conversation_id=conversation_id,
            cleared_input="",
            user_turn_id=None,
            history_state=chat_history,
            conversation_history="",
            user_text="",
            error_update={"visible": False},
            skip_pipeline=False,
            timings={},
        )
    if conversation_id is None:
        conversation_id = orchestrator.create_conversation()
    timings = dict(timings or {})
    user_turn_id = orchestrator.persist_user_text_turn(
        conversation_id,
        user_text,
        timings=timings,
    )

    normalized_history = _normalize_history(chat_history)
    display_history = list(normalized_history)
    display_history.append({"role": "user", "content": user_text})
    conversation_history = _format_conversation_history(normalized_history)
    return TextTurnStartResult(
        chat_history=display_history,
        conversation_id=conversation_id,
        cleared_input="",
        user_turn_id=user_turn_id,
        history_state=display_history,
        conversation_history=conversation_history,
        user_text=user_text,
        error_update={"visible": False},
        skip_pipeline=False,
        timings=timings,
    )


def _start_audio_turn(
    orchestrator: ConversationOrchestrator,
    audio,
    chat_history: list[dict[str, str]] | list[tuple[str, str]],
    conversation_id: str | None,
    timings: dict | None = None,
)-> AudioTurnStartResult:
    try:
        target_sample_rate = getattr(orchestrator, "expected_sample_rate", None)
        start = time.perf_counter()
        pcm_bytes, meta, raw_pcm_bytes, raw_meta = _audio_to_pcm_with_raw(audio, target_sample_rate)
        timings = dict(timings or {})
        timings["audio_preprocess_seconds"] = time.perf_counter() - start
        if conversation_id is None:
            conversation_id = orchestrator.create_conversation()
        result = orchestrator.prepare_audio_turn(
            conversation_id=conversation_id,
            pcm_bytes=pcm_bytes,
            audio_meta=meta,
            timings=timings,
        )
        if raw_pcm_bytes is not None and raw_meta is not None:
            try:
                orchestrator.persist_input_audio(
                    conversation_id=conversation_id,
                    turn_id=result.user_turn_id,
                    pcm_bytes=raw_pcm_bytes,
                    audio_meta=raw_meta,
                    kind_suffix="raw",
                )
            except Exception as exc:  # pragma: no cover - non-fatal for UI flow
                _logger.warning("audio_turn.raw_store_failed: %s", exc)
    except Exception as exc:
        _logger.warning("audio_turn.start_failed: %s", exc)
        return AudioTurnStartResult(
            chat_history=chat_history,
            conversation_id=conversation_id,
            cleared_input="",
            user_turn_id=None,
            history_state=chat_history,
            conversation_history="",
            user_text="",
            error_update={"visible": False},
            user_audio_path=None,
            skip_pipeline=True,
            timings=timings or {},
        )

    normalized_history = _normalize_history(chat_history)
    display_history = list(normalized_history)
    display_history.append({"role": "user", "content": result.asr_text})
    conversation_history = _format_conversation_history(normalized_history)
    return AudioTurnStartResult(
        chat_history=display_history,
        conversation_id=conversation_id,
        cleared_input="",
        user_turn_id=result.user_turn_id,
        history_state=display_history,
        conversation_history=conversation_history,
        user_text=result.asr_text,
        error_update={"visible": False},
        user_audio_path=result.input_audio_path,
        skip_pipeline=False,
        timings=timings,
    )


def _run_llm_reply(
    orchestrator: ConversationOrchestrator,
    conversation_id: str | None,
    user_turn_id: str | None,
    user_text: str,
    conversation_history: str,
    chat_history: list[dict[str, str]],
    skip_pipeline: bool,
    timings: dict | None = None,
)-> LlmReplyResult:
    if skip_pipeline:
        return LlmReplyResult(chat_history=chat_history, assistant_turn_id=None, reply_text="", error_update={"visible": False})
    if conversation_id is None or user_turn_id is None:
        return LlmReplyResult(chat_history=chat_history, assistant_turn_id=None, reply_text="", error_update={"visible": False})
    if timings is None:
        timings = {}
    assistant_turn_id, reply_text = orchestrator.generate_reply(
        conversation_id=conversation_id,
        user_turn_id=user_turn_id,
        user_text=user_text,
        conversation_history=conversation_history,
        timings=timings,
    )
    if not reply_text:
        reply_text = "(No reply - invalid LLM response)"
        _logger.warning("llm_reply.invalid_json")
        error_update = {"visible": False}
    else:
        error_update = {"visible": False}
    updated_history = _replace_last_assistant(chat_history, reply_text)
    return LlmReplyResult(
        chat_history=updated_history,
        assistant_turn_id=assistant_turn_id,
        reply_text=reply_text,
        error_update=error_update,
    )


def _run_corrections(
    orchestrator: ConversationOrchestrator,
    user_turn_id: str | None,
    user_text: str,
    assistant_turn_id: str | None,
    skip_pipeline: bool,
    corrections_enabled: bool,
    timings: dict | None = None,
):
    if skip_pipeline or user_turn_id is None or not corrections_enabled:
        return ("", "", "")
    if timings is None:
        timings = {}
    corrections = orchestrator.run_corrections(
        user_turn_id,
        user_text,
        assistant_turn_id=assistant_turn_id,
        timings=timings,
    )
    return (
        corrections["corrected"],
        corrections["native"],
        corrections["explanation"],
    )


def _run_tts(
    orchestrator: ConversationOrchestrator,
    conversation_id: str | None,
    assistant_turn_id: str | None,
    reply_text: str,
    skip_pipeline: bool,
    timings: dict | None = None,
):
    if skip_pipeline or conversation_id is None or assistant_turn_id is None:
        return None
    if timings is None:
        timings = {}
    tts_result = orchestrator.run_tts(
        conversation_id,
        assistant_turn_id,
        reply_text,
        timings=timings,
    )
    return tts_result.audio_path if tts_result is not None else None


def _handle_reset(orchestrator: ConversationOrchestrator):
    orchestrator.reset_session()
    return (
        [],
        None,
        "",
        None,
        "",
        None,
        None,
        "",
        "",
        "",
        None,
        [],
        "",
        "",
        "",
        None,
        False,
        {},
    )


def _handle_language_change(
    orchestrator: ConversationOrchestrator,
    language: str,
    suppress_language_change: bool,
):
    if suppress_language_change:
        import gradio as gr  # type: ignore

        no_change = gr.update()
        return (
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
            no_change,
        )
    orchestrator.set_language(language)
    return _handle_reset(orchestrator)


def build_ui(orchestrator: ConversationOrchestrator):
    import gradio as gr  # type: ignore
    language_choices = [
        ("🇯🇵 Japanese", "ja"),
        ("🇫🇷 French", "fr"),
        ("🇺🇸 English", "en"),
        ("🇪🇸 Spanish", "es"),
        ("🇮🇹 Italian", "it"),
        ("🇧🇷 Portuguese (Brazil)", "pt-br"),
    ]
    for _, value in language_choices:
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language in UI choices: {value}")
    default_language = getattr(orchestrator, "language", "ja")
    default_waveform_options = _waveform_options(default_language)
    conversation_choices = _load_conversation_options(orchestrator)
    load_enabled = bool(conversation_choices)

    with gr.Blocks(
        css="""
#input-grid {align-items: stretch;}
#input-grid > div {flex: 1 1 0;}
#text-input, #audio-input {min-height: 200px;}
#text-input textarea {min-height: 140px;}
#audio-input .wrap {min-height: 140px;}
#header-row {align-items: center; justify-content: space-between; gap: 0;}
#header-row h1 {margin: 0; text-align: left;}
#header-left {display: flex; align-items: center;}
#header-right {display: flex; justify-content: flex-end;}
#delete-confirm-row, #delete-all-confirm-row,
#delete-confirm-buttons, #delete-all-confirm-buttons {
  background: #ffecec;
  border-radius: 6px;
  padding: 6px;
}
"""
    ) as demo:
        theme_html = gr.HTML(_theme_html(default_language))
        with gr.Row(elem_id="header-row"):
            with gr.Column(elem_id="header-left"):
                gr.Markdown("# KaiwaCoach")
            with gr.Column(elem_id="header-right"):
                language_dropdown = gr.Dropdown(
                    choices=language_choices,
                    value=default_language,
                    interactive=True,
                    show_label=False,
                )
        error_output = gr.Markdown(visible=False)
        with gr.Row():
            with gr.Column(scale=2, min_width=360):
                chat = gr.Chatbot(label="Conversation")
                with gr.Row(elem_id="input-grid"):
                    with gr.Column(scale=1, min_width=260, elem_id="text-input"):
                        user_input = gr.Textbox(label="Your message", lines=3)
                        send_btn = gr.Button("Send")
                    with gr.Column(scale=1, min_width=260, elem_id="audio-input"):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            label="Microphone",
                            elem_id="mic-input",
                            waveform_options=default_waveform_options,
                        )
                        audio_btn = gr.Button("Send Audio")
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Conversations")
                empty_conversations_note = gr.Markdown(
                    "No saved conversations yet.",
                    visible=not bool(conversation_choices),
                )
                conversation_dropdown = gr.Dropdown(
                    choices=conversation_choices,
                    value=None,
                    label="",
                    show_label=False,
                    interactive=True,
                )
                with gr.Row():
                    refresh_conversations_btn = gr.Button("Refresh")
                    load_conversation_btn = gr.Button("Load", interactive=load_enabled)
                    delete_conversation_btn = gr.Button("Delete")
                with gr.Row(visible=False, elem_id="delete-confirm-row") as delete_confirm_row:
                    gr.Markdown("⚠️ Delete this conversation? This cannot be undone. ⚠️")
                with gr.Row(visible=False, elem_id="delete-confirm-buttons") as delete_confirm_buttons_row:
                    delete_confirm_btn = gr.Button("Confirm Delete")
                    delete_cancel_btn = gr.Button("Cancel")
                user_audio_output = gr.Audio(
                    label="Last user audio",
                    interactive=False,
                    elem_id="last-user-audio",
                    waveform_options=default_waveform_options,
                )
                assistant_audio_output = gr.Audio(
                    label="Last assistant audio",
                    autoplay=True,
                    interactive=False,
                    elem_id="last-assistant-audio",
                    waveform_options=default_waveform_options,
                )
                corrections_toggle = gr.Checkbox(value=True, label="Corrections", elem_id="corrections-toggle")
                corrected_output = gr.Textbox(label="Corrected")
                native_output = gr.Textbox(label="Native")
                explanation_output = gr.Textbox(label="Explanation")
        with gr.Row():
            reset_btn = gr.Button("Reset Session")
            delete_all_btn = gr.Button("Delete All History")
        with gr.Row(visible=False, elem_id="delete-all-confirm-row") as delete_all_confirm_row:
            gr.Markdown("⚠️ Delete all conversations and history? This cannot be undone. ⚠️")
        with gr.Row(visible=False, elem_id="delete-all-confirm-buttons") as delete_all_confirm_buttons_row:
            delete_all_confirm_btn = gr.Button("Confirm Delete All")
            delete_all_cancel_btn = gr.Button("Cancel")

        conversation_id_state = gr.State(None)
        history_state = gr.State([])
        user_turn_id_state = gr.State(None)
        assistant_turn_id_state = gr.State(None)
        conversation_history_state = gr.State("")
        user_text_state = gr.State("")
        reply_text_state = gr.State("")
        skip_pipeline_state = gr.State(False)
        timings_state = gr.State({})
        suppress_language_change_state = gr.State(False)
        loaded_language_state = gr.State(None)
        theme_audio_outputs = [theme_html, audio_input, user_audio_output, assistant_audio_output]
        reset_outputs = [
            chat,
            conversation_id_state,
            user_input,
            audio_input,
            error_output,
            user_audio_output,
            assistant_audio_output,
            corrected_output,
            native_output,
            explanation_output,
            user_turn_id_state,
            history_state,
            conversation_history_state,
            user_text_state,
            reply_text_state,
            assistant_turn_id_state,
            skip_pipeline_state,
            timings_state,
        ]
        conversation_picker_outputs = [conversation_dropdown, load_conversation_btn, empty_conversations_note]
        delete_refresh_outputs = [*reset_outputs, *conversation_picker_outputs]
        confirm_delete_outputs = [delete_confirm_row, delete_confirm_buttons_row]
        confirm_delete_all_outputs = [delete_all_confirm_row, delete_all_confirm_buttons_row]
        load_conversation_outputs = [
            chat,
            conversation_id_state,
            history_state,
            conversation_history_state,
            user_text_state,
            reply_text_state,
            user_turn_id_state,
            assistant_turn_id_state,
            skip_pipeline_state,
            timings_state,
            corrected_output,
            native_output,
            explanation_output,
            user_audio_output,
            assistant_audio_output,
            error_output,
            loaded_language_state,
            suppress_language_change_state,
        ]

        def _apply_loaded_language(language: str | None):
            if not language:
                return gr.update()
            return gr.update(value=language)

        def _wire_language_and_conversation_handlers() -> None:
            language_dropdown.change(
                lambda language, suppress: _handle_language_change(orchestrator, language, suppress),
                inputs=[language_dropdown, suppress_language_change_state],
                outputs=reset_outputs,
            ).then(
                _theme_updates_for_language,
                inputs=[language_dropdown],
                outputs=theme_audio_outputs,
            )

            refresh_conversations_btn.click(
                lambda: _refresh_conversation_options(orchestrator),
                inputs=[],
                outputs=conversation_picker_outputs,
            )

            load_conversation_btn.click(
                lambda: True,
                inputs=[],
                outputs=[suppress_language_change_state],
            ).then(
                lambda cid: _load_conversation(orchestrator, cid),
                inputs=[conversation_dropdown],
                outputs=load_conversation_outputs,
            ).then(
                _apply_loaded_language,
                inputs=[loaded_language_state],
                outputs=[language_dropdown],
            ).then(
                _theme_updates_for_language,
                inputs=[language_dropdown],
                outputs=theme_audio_outputs,
            ).then(
                lambda: False,
                inputs=[],
                outputs=[suppress_language_change_state],
            )

        def _wire_delete_handlers() -> None:
            delete_conversation_btn.click(
                _confirm_row_show_updates,
                inputs=[],
                outputs=confirm_delete_outputs,
            )
            delete_cancel_btn.click(
                _confirm_row_hide_updates,
                inputs=[],
                outputs=confirm_delete_outputs,
            )
            delete_confirm_btn.click(
                lambda cid: _delete_conversation_and_refresh(orchestrator, cid),
                inputs=[conversation_id_state],
                outputs=delete_refresh_outputs,
            ).then(
                _confirm_row_hide_updates,
                inputs=[],
                outputs=confirm_delete_outputs,
            )

            delete_all_btn.click(
                _confirm_row_show_updates,
                inputs=[],
                outputs=confirm_delete_all_outputs,
            )
            delete_all_cancel_btn.click(
                _confirm_row_hide_updates,
                inputs=[],
                outputs=confirm_delete_all_outputs,
            )
            delete_all_confirm_btn.click(
                lambda: _delete_all_conversations_and_refresh(orchestrator),
                inputs=[],
                outputs=delete_refresh_outputs,
            ).then(
                _confirm_row_hide_updates,
                inputs=[],
                outputs=confirm_delete_all_outputs,
            )

        def _wire_turn_handlers() -> None:
            def _on_send(
                user_text: str,
                chat_history: list[dict[str, str]] | list[tuple[str, str]],
                conversation_id: str | None,
                timings: dict,
            ):
                result = _start_text_turn(orchestrator, user_text, chat_history, conversation_id, timings)
                error_update = gr.update(**result.error_update)
                return (
                    result.chat_history,
                    result.conversation_id,
                    result.cleared_input,
                    result.user_turn_id,
                    result.history_state,
                    result.conversation_history,
                    result.user_text,
                    error_update,
                    result.skip_pipeline,
                    result.timings,
                )

            def _on_audio(
                audio,
                chat_history: list[dict[str, str]] | list[tuple[str, str]],
                conversation_id: str | None,
                timings: dict,
            ):
                result = _start_audio_turn(orchestrator, audio, chat_history, conversation_id, timings)
                error_update = gr.update(**result.error_update)
                return (
                    result.chat_history,
                    result.conversation_id,
                    result.cleared_input,
                    result.user_turn_id,
                    result.history_state,
                    result.conversation_history,
                    result.user_text,
                    error_update,
                    result.user_audio_path,
                    result.skip_pipeline,
                    result.timings,
                )

            def _llm_reply_wrapper(
                conversation_id: str | None,
                user_turn_id: str | None,
                user_text: str,
                conversation_history: str,
                chat_history: list[dict[str, str]],
                skip_pipeline: bool,
                timings: dict,
            ):
                timings = dict(timings or {})
                result = _run_llm_reply(
                    orchestrator,
                    conversation_id,
                    user_turn_id,
                    user_text,
                    conversation_history,
                    chat_history,
                    skip_pipeline,
                    timings=timings,
                )
                error_update = gr.update(**result.error_update)
                return (
                    result.chat_history,
                    result.assistant_turn_id,
                    result.reply_text,
                    error_update,
                    timings,
                )

            def _corrections_wrapper(
                user_turn_id: str | None,
                user_text: str,
                assistant_turn_id: str | None,
                skip_pipeline: bool,
                corrections_enabled: bool,
                timings: dict,
            ):
                timings = dict(timings or {})
                result = _run_corrections(
                    orchestrator,
                    user_turn_id,
                    user_text,
                    assistant_turn_id,
                    skip_pipeline,
                    corrections_enabled,
                    timings=timings,
                )
                return result + (timings,)

            def _tts_wrapper(
                conversation_id: str | None,
                assistant_turn_id: str | None,
                reply_text: str,
                skip_pipeline: bool,
                timings: dict,
            ):
                timings = dict(timings or {})
                result = _run_tts(
                    orchestrator,
                    conversation_id,
                    assistant_turn_id,
                    reply_text,
                    skip_pipeline,
                    timings=timings,
                )
                return result, timings

            def _wire_turn_pipeline(start_event, turn_name: str):
                return (
                    start_event.then(
                        _llm_reply_wrapper,
                        inputs=[
                            conversation_id_state,
                            user_turn_id_state,
                            user_text_state,
                            conversation_history_state,
                            history_state,
                            skip_pipeline_state,
                            timings_state,
                        ],
                        outputs=[
                            chat,
                            assistant_turn_id_state,
                            reply_text_state,
                            error_output,
                            timings_state,
                        ],
                    )
                    .then(
                        _tts_wrapper,
                        inputs=[
                            conversation_id_state,
                            assistant_turn_id_state,
                            reply_text_state,
                            skip_pipeline_state,
                            timings_state,
                        ],
                        outputs=[assistant_audio_output, timings_state],
                    )
                    .then(
                        _corrections_wrapper,
                        inputs=[
                            user_turn_id_state,
                            user_text_state,
                            assistant_turn_id_state,
                            skip_pipeline_state,
                            corrections_toggle,
                            timings_state,
                        ],
                        outputs=[corrected_output, native_output, explanation_output, timings_state],
                    )
                    .then(
                        lambda h, t: (h, t),
                        inputs=[chat, timings_state],
                        outputs=[history_state, timings_state],
                    )
                    .then(
                        lambda t: orchestrator.finalize_and_log_timings(turn_name, t),
                        inputs=[timings_state],
                        outputs=[],
                    )
                )

            text_start_inputs = [user_input, history_state, conversation_id_state, timings_state]
            text_start_outputs = [
                chat,
                conversation_id_state,
                user_input,
                user_turn_id_state,
                history_state,
                conversation_history_state,
                user_text_state,
                error_output,
                skip_pipeline_state,
                timings_state,
            ]
            audio_start_inputs = [audio_input, history_state, conversation_id_state, timings_state]
            audio_start_outputs = [
                chat,
                conversation_id_state,
                user_input,
                user_turn_id_state,
                history_state,
                conversation_history_state,
                user_text_state,
                error_output,
                user_audio_output,
                skip_pipeline_state,
                timings_state,
            ]

            _wire_turn_pipeline(
                send_btn.click(
                    _on_send,
                    inputs=text_start_inputs,
                    outputs=text_start_outputs,
                ),
                "text_turn",
            )
            _wire_turn_pipeline(
                user_input.submit(
                    _on_send,
                    inputs=text_start_inputs,
                    outputs=text_start_outputs,
                ),
                "text_turn",
            )
            _wire_turn_pipeline(
                audio_btn.click(
                    _on_audio,
                    inputs=audio_start_inputs,
                    outputs=audio_start_outputs,
                ),
                "audio_turn",
            )

        def _wire_reset_handler() -> None:
            reset_btn.click(
                lambda: _handle_reset(orchestrator),
                inputs=[],
                outputs=reset_outputs,
            ).then(
                lambda h: h,
                inputs=chat,
                outputs=history_state,
            )

        _wire_language_and_conversation_handlers()
        _wire_delete_handlers()
        _wire_turn_handlers()
        _wire_reset_handler()

    return demo
