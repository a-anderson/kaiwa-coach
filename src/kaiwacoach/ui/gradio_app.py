"""Gradio UI wiring and handlers.

Notes
-----
- UI callbacks call the orchestrator only. No direct DB access happens here.
- The orchestrator uses SQLiteWriter's single-writer queue plus short-lived read
  connections, which is safe for Gradio's concurrent callback execution.
"""

from __future__ import annotations

from pathlib import Path
import logging
import time

from kaiwacoach.constants import SUPPORTED_LANGUAGES
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.storage.blobs import AudioMeta

_logger = logging.getLogger(__name__)

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


def _start_text_turn(
    orchestrator: ConversationOrchestrator,
    user_text: str,
    chat_history: list[dict[str, str]] | list[tuple[str, str]],
    conversation_id: str | None,
    timings: dict | None = None,
):
    if not user_text:
        return (
            chat_history,
            conversation_id,
            "",
            None,
            chat_history,
            "",
            "",
            "",
            {"visible": False},
            False,
            {},
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
    return (
        display_history,
        conversation_id,
        "",
        user_turn_id,
        display_history,
        conversation_history,
        user_text,
        {"visible": False},
        False,
        timings,
    )


def _start_audio_turn(
    orchestrator: ConversationOrchestrator,
    audio,
    chat_history: list[dict[str, str]] | list[tuple[str, str]],
    conversation_id: str | None,
    timings: dict | None = None,
):
    if conversation_id is None:
        conversation_id = orchestrator.create_conversation()
    try:
        target_sample_rate = getattr(orchestrator, "expected_sample_rate", None)
        start = time.perf_counter()
        pcm_bytes, meta, raw_pcm_bytes, raw_meta = _audio_to_pcm_with_raw(audio, target_sample_rate)
        timings = dict(timings or {})
        timings["audio_preprocess_seconds"] = time.perf_counter() - start
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
        return (
            chat_history,
            conversation_id,
            "",
            None,
            chat_history,
            "",
            "",
            {"visible": False},
            None,
            True,
            timings or {},
        )

    normalized_history = _normalize_history(chat_history)
    display_history = list(normalized_history)
    display_history.append({"role": "user", "content": result.asr_text})
    conversation_history = _format_conversation_history(normalized_history)
    return (
        display_history,
        conversation_id,
        "",
        result.user_turn_id,
        display_history,
        conversation_history,
        result.asr_text,
        {"visible": False},
        result.input_audio_path,
        False,
        timings,
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
):
    if skip_pipeline:
        return (
            chat_history,
            None,
            "",
            {"visible": False},
        )
    if conversation_id is None or user_turn_id is None:
        return (
            chat_history,
            None,
            "",
            {"visible": False},
        )
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
    return (
        updated_history,
        assistant_turn_id,
        reply_text,
        error_update,
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


def _handle_text_turn(
    orchestrator: ConversationOrchestrator,
    user_text: str,
    chat_history: list[dict[str, str]] | list[tuple[str, str]],
    conversation_id: str | None,
):
    if not user_text:
        return (
            chat_history,
            conversation_id,
            "",
            {"visible": False},
            None,
            None,
            [],
            "",
            "",
            "",
        )
    if conversation_id is None:
        conversation_id = orchestrator.create_conversation()
    conversation_history = _format_conversation_history(chat_history)
    result = orchestrator.process_text_turn(
        conversation_id=conversation_id,
        user_text=user_text,
        conversation_history=conversation_history,
    )
    corrections = orchestrator.get_latest_corrections(result.user_turn_id)
    reply_text = result.reply_text
    error_update = {"visible": False}
    if not reply_text:
        reply_text = "(No reply - invalid LLM response)"
        error_update = {
            "value": "LLM response was invalid JSON. See logs for details.",
            "visible": True,
        }

    normalized_history = _normalize_history(chat_history)
    normalized_history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply_text},
        ]
    )
    return (
        normalized_history,
        conversation_id,
        "",
        error_update,
        None,
        result.tts_audio_path,
        corrections["corrected"],
        corrections["native"],
        corrections["explanation"],
    )


def _handle_audio_turn(
    orchestrator: ConversationOrchestrator,
    audio,
    chat_history: list[dict[str, str]] | list[tuple[str, str]],
    conversation_id: str | None,
):
    if conversation_id is None:
        conversation_id = orchestrator.create_conversation()
    try:
        pcm_bytes, meta = _audio_to_pcm(audio)
        conversation_history = _format_conversation_history(chat_history)
        result = orchestrator.process_audio_turn(
            conversation_id=conversation_id,
            pcm_bytes=pcm_bytes,
            audio_meta=meta,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        return (
            chat_history,
            conversation_id,
            "",
            {"value": str(exc), "visible": True},
            None,
            None,
            [],
            "",
            "",
            "",
        )

    normalized_history = _normalize_history(chat_history)
    reply_text = result.reply_text
    error_update = {"visible": False}
    if not reply_text:
        reply_text = "(No reply - invalid LLM response)"
        error_update = {
            "value": "LLM response was invalid JSON. See logs for details.",
            "visible": True,
        }
    normalized_history.extend(
        [
            {"role": "user", "content": result.asr_text},
            {"role": "assistant", "content": reply_text},
        ]
    )
    corrections = orchestrator.get_latest_corrections(result.user_turn_id)
    return (
        normalized_history,
        conversation_id,
        None,
        error_update,
        result.input_audio_path,
        result.tts_audio_path,
        corrections["corrected"],
        corrections["native"],
        corrections["explanation"],
    )


def _handle_reset(orchestrator: ConversationOrchestrator):
    orchestrator.reset_session()
    return (
        [],
        None,
        "",
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
    orchestrator: ConversationOrchestrator, language: str, conversation_id: str | None
):
    orchestrator.set_language(language)
    if conversation_id is not None:
        orchestrator.update_conversation_language(conversation_id, language)
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
"""
    ) as demo:
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
                        audio_input = gr.Audio(sources=["microphone"], label="Microphone")
                        audio_btn = gr.Button("Send Audio")
            with gr.Column(scale=1, min_width=280):
                user_audio_output = gr.Audio(label="Last user audio", interactive=False)
                assistant_audio_output = gr.Audio(label="Last assistant audio", autoplay=True, interactive=False)
                corrections_toggle = gr.Checkbox(value=True, label="Corrections")
                corrected_output = gr.Textbox(label="Corrected")
                native_output = gr.Textbox(label="Native")
                explanation_output = gr.Textbox(label="Explanation")
        reset_btn = gr.Button("Reset Session")

        conversation_id_state = gr.State(None)
        history_state = gr.State([])
        user_turn_id_state = gr.State(None)
        assistant_turn_id_state = gr.State(None)
        conversation_history_state = gr.State("")
        user_text_state = gr.State("")
        reply_text_state = gr.State("")
        skip_pipeline_state = gr.State(False)
        timings_state = gr.State({})
        language_dropdown.change(
            lambda lang, cid: _handle_language_change(orchestrator, lang, cid),
            inputs=[language_dropdown, conversation_id_state],
            outputs=[
                chat,
                conversation_id_state,
                user_input,
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
            ],
        )

        def _on_send(
            user_text: str,
            chat_history: list[dict[str, str]] | list[tuple[str, str]],
            conversation_id: str | None,
            timings: dict,
        ):
            result = _start_text_turn(orchestrator, user_text, chat_history, conversation_id, timings)
            error_update = result[7]
            if isinstance(error_update, dict):
                error_update = gr.update(**error_update)
            return result[:7] + (error_update,) + result[8:]

        def _on_audio(
            audio,
            chat_history: list[dict[str, str]] | list[tuple[str, str]],
            conversation_id: str | None,
            timings: dict,
        ):
            result = _start_audio_turn(orchestrator, audio, chat_history, conversation_id, timings)
            error_update = result[7]
            if isinstance(error_update, dict):
                error_update = gr.update(**error_update)
            return result[:7] + (error_update,) + result[8:]

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
            error_update = result[3]
            if isinstance(error_update, dict):
                error_update = gr.update(**error_update)
            return result[:3] + (error_update,) + (timings,)

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

        send_btn.click(
            _on_send,
            inputs=[user_input, history_state, conversation_id_state, timings_state],
            outputs=[
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
            ],
        ).then(
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
        ).then(
            _tts_wrapper,
            inputs=[
                conversation_id_state,
                assistant_turn_id_state,
                reply_text_state,
                skip_pipeline_state,
                timings_state,
            ],
            outputs=[assistant_audio_output, timings_state],
        ).then(
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
        ).then(
            lambda h, t: (h, t),
            inputs=[chat, timings_state],
            outputs=[history_state, timings_state],
        ).then(
            lambda t: orchestrator.finalize_and_log_timings("text_turn", t),
            inputs=[timings_state],
            outputs=[],
        )

        user_input.submit(
            _on_send,
            inputs=[user_input, history_state, conversation_id_state, timings_state],
            outputs=[
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
            ],
        ).then(
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
        ).then(
            _tts_wrapper,
            inputs=[
                conversation_id_state,
                assistant_turn_id_state,
                reply_text_state,
                skip_pipeline_state,
                timings_state,
            ],
            outputs=[assistant_audio_output, timings_state],
        ).then(
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
        ).then(
            lambda h, t: (h, t),
            inputs=[chat, timings_state],
            outputs=[history_state, timings_state],
        ).then(
            lambda t: orchestrator.finalize_and_log_timings("text_turn", t),
            inputs=[timings_state],
            outputs=[],
        )

        audio_btn.click(
            _on_audio,
            inputs=[audio_input, history_state, conversation_id_state, timings_state],
            outputs=[
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
            ],
        ).then(
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
        ).then(
            _tts_wrapper,
            inputs=[
                conversation_id_state,
                assistant_turn_id_state,
                reply_text_state,
                skip_pipeline_state,
                timings_state,
            ],
            outputs=[assistant_audio_output, timings_state],
        ).then(
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
        ).then(
            lambda h, t: (h, t),
            inputs=[chat, timings_state],
            outputs=[history_state, timings_state],
        ).then(
            lambda t: orchestrator.finalize_and_log_timings("audio_turn", t),
            inputs=[timings_state],
            outputs=[],
        )

        reset_btn.click(
            lambda: _handle_reset(orchestrator),
            inputs=[],
            outputs=[
                chat,
                conversation_id_state,
                user_input,
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
            ],
        ).then(
            lambda h: h,
            inputs=chat,
            outputs=history_state,
        )

    return demo
