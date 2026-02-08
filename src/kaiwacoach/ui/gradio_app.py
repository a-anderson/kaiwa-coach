"""Gradio UI wiring and handlers.

Notes
-----
- UI callbacks call the orchestrator only. No direct DB access happens here.
- The orchestrator uses SQLiteWriter's single-writer queue plus short-lived read
  connections, which is safe for Gradio's concurrent callback execution.
"""

from __future__ import annotations

from pathlib import Path

from kaiwacoach.models.asr_whisper import ASRResult
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.storage.blobs import AudioMeta


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
    if audio is None:
        raise ValueError("No audio input provided.")
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


def _normalize_history(chat_history: list[dict[str, str]] | list[tuple[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in chat_history:
        if isinstance(message, dict):
            normalized.append({"role": message.get("role", ""), "content": message.get("content", "")})
        elif isinstance(message, tuple) and len(message) == 2:
            normalized.append({"role": "user", "content": message[0]})
            normalized.append({"role": "assistant", "content": message[1]})
    return normalized


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
        corrections["errors"],
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
        corrections["errors"],
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
        [],
        "",
        "",
        "",
    )


def build_ui(orchestrator: ConversationOrchestrator):
    import gradio as gr  # type: ignore

    with gr.Blocks(
        css="""
#input-grid {align-items: stretch;}
#input-grid > div {flex: 1 1 0;}
#text-input, #audio-input {min-height: 200px;}
#text-input textarea {min-height: 140px;}
#audio-input .wrap {min-height: 140px;}
#header-row {align-items: center; justify-content: flex-start; gap: 0;}
#header-row > div:last-child {flex: 0 0 auto;}
#header-row h1 {margin: 0; text-align: left;}
"""
    ) as demo:
        with gr.Row(elem_id="header-row"):
            gr.Markdown("# KaiwaCoach")
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
                errors_output = gr.Dataframe(
                    headers=["Errors"],
                    datatype=["str"],
                    row_count=0,
                    column_count=1,
                    label="Errors",
                )
                corrected_output = gr.Textbox(label="Corrected")
                native_output = gr.Textbox(label="Native")
                explanation_output = gr.Textbox(label="Explanation")
                reset_btn = gr.Button("Reset Session")

        conversation_id_state = gr.State(None)
        history_state = gr.State([])

        def _on_send(
            user_text: str,
            chat_history: list[dict[str, str]] | list[tuple[str, str]],
            conversation_id: str | None,
        ):
            result = _handle_text_turn(orchestrator, user_text, chat_history, conversation_id)
            error_update = result[3]
            if isinstance(error_update, dict):
                error_update = gr.update(**error_update)
            return result[:3] + (error_update,) + result[4:]

        def _on_audio(
            audio,
            chat_history: list[dict[str, str]] | list[tuple[str, str]],
            conversation_id: str | None,
        ):
            result = _handle_audio_turn(orchestrator, audio, chat_history, conversation_id)
            error_update = result[3]
            if isinstance(error_update, dict):
                error_update = gr.update(**error_update)
            return result[:3] + (error_update,) + result[4:]

        send_btn.click(
            _on_send,
            inputs=[user_input, history_state, conversation_id_state],
            outputs=[
                chat,
                conversation_id_state,
                user_input,
                error_output,
                user_audio_output,
                assistant_audio_output,
                errors_output,
                corrected_output,
                native_output,
                explanation_output,
            ],
        ).then(
            lambda h: h,
            inputs=chat,
            outputs=history_state,
        )

        user_input.submit(
            _on_send,
            inputs=[user_input, history_state, conversation_id_state],
            outputs=[
                chat,
                conversation_id_state,
                user_input,
                error_output,
                user_audio_output,
                assistant_audio_output,
                errors_output,
                corrected_output,
                native_output,
                explanation_output,
            ],
        ).then(
            lambda h: h,
            inputs=chat,
            outputs=history_state,
        )

        audio_btn.click(
            _on_audio,
            inputs=[audio_input, history_state, conversation_id_state],
            outputs=[
                chat,
                conversation_id_state,
                user_input,
                error_output,
                user_audio_output,
                assistant_audio_output,
                errors_output,
                corrected_output,
                native_output,
                explanation_output,
            ],
        ).then(
            lambda h: h,
            inputs=chat,
            outputs=history_state,
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
                errors_output,
                corrected_output,
                native_output,
                explanation_output,
            ],
        ).then(
            lambda h: h,
            inputs=chat,
            outputs=history_state,
        )

    return demo
