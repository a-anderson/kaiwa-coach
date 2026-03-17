"""SSE turn streaming routes.

POST /api/turns/text  — submit a text turn, stream stage events + complete
POST /api/turns/audio — submit a WebM audio turn, stream stage events + complete
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from kaiwacoach.api.audio_conversion import webm_to_pcm
from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.schemas.turn import TurnTextRequest
from kaiwacoach.orchestrator import AudioTurnResult, ConversationOrchestrator, TextTurnResult

_logger = logging.getLogger(__name__)

router = APIRouter()

# One turn at a time — the orchestrator holds ML models and is not safe for
# concurrent processing. max_workers=1 serialises requests naturally.
_executor = ThreadPoolExecutor(max_workers=1)


def _audio_path_to_url(audio_path: str | None, cache_root: Path) -> str | None:
    """Return a serveable /api/audio URL for *audio_path*, or None if missing."""
    if not audio_path:
        return None
    p = Path(audio_path)
    if not p.exists():
        return None
    try:
        rel = p.relative_to(cache_root)
        return f"/api/audio/{rel}"
    except ValueError:
        return None


def _build_sse_generator(
    orc: ConversationOrchestrator,
    cache_root: Path,
    run_fn,
):
    """Return an async generator that offloads *run_fn* to a thread and yields SSE dicts.

    *run_fn* must accept a single ``on_stage`` callback and return a
    ``TextTurnResult`` or ``AudioTurnResult``.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_stage(stage: str, status: str, data: dict) -> None:
        # Called from background thread — bridge to the async event loop.
        if stage == "tts" and status == "complete":
            audio_url = _audio_path_to_url(data.get("audio_path"), cache_root)
            payload = {"stage": stage, "status": status, "audio_url": audio_url}
        else:
            payload = {"stage": stage, "status": status, **data}
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def run_sync() -> None:
        try:
            result = run_fn(on_stage)
            loop.call_soon_threadsafe(queue.put_nowait, {"_done": True, "result": result})
        except Exception as exc:  # noqa: BLE001
            _logger.exception("Turn processing failed")
            loop.call_soon_threadsafe(queue.put_nowait, {"_done": True, "error": str(exc)})

    loop.run_in_executor(_executor, run_sync)

    async def event_generator():
        while True:
            msg = await queue.get()
            if msg.get("_done"):
                if "error" in msg:
                    yield {"event": "error", "data": json.dumps({"message": msg["error"]})}
                else:
                    result = msg["result"]
                    tts_url = _audio_path_to_url(result.tts_audio_path, cache_root)
                    complete_data: dict = {
                        "conversation_id": result.conversation_id,
                        "user_turn_id": result.user_turn_id,
                        "assistant_turn_id": result.assistant_turn_id,
                        "reply_text": result.reply_text,
                        "audio_url": tts_url,
                    }
                    if isinstance(result, AudioTurnResult):
                        complete_data["asr_text"] = result.asr_text
                    yield {"event": "complete", "data": json.dumps(complete_data)}
                break
            else:
                yield {"event": "stage", "data": json.dumps(msg)}

    return event_generator()


@router.post("/turns/text")
async def submit_text_turn(body: TurnTextRequest, request: Request) -> EventSourceResponse:
    """Submit a text turn and stream pipeline progress as SSE.

    Emits ``stage`` events for each pipeline step, then a final ``complete``
    event with the turn IDs and audio URL.
    """
    orc: ConversationOrchestrator = get_orchestrator(request)
    cache_root: Path = request.app.state.audio_cache.root_dir

    if body.language:
        orc.set_language(body.language)

    conversation_id = body.conversation_id or orc.create_conversation()

    def run_fn(on_stage):
        return orc.process_text_turn(
            conversation_id=conversation_id,
            user_text=body.text,
            conversation_history=body.conversation_history,
            corrections_enabled=body.corrections_enabled,
            on_stage=on_stage,
        )

    return EventSourceResponse(_build_sse_generator(orc, cache_root, run_fn))


@router.post("/turns/audio")
async def submit_audio_turn(
    request: Request,
    audio: UploadFile,
    conversation_id: str = Form(default=""),
    language: str = Form(default="ja"),
    conversation_history: str = Form(default=""),
    corrections_enabled: bool = Form(default=True),
) -> EventSourceResponse:
    """Submit an audio turn (WebM/Opus) and stream pipeline progress as SSE.

    The audio field must be a WebM file as recorded by the browser's
    MediaRecorder API. Conversion to PCM is handled server-side via ffmpeg.
    """
    orc: ConversationOrchestrator = get_orchestrator(request)
    cache_root: Path = request.app.state.audio_cache.root_dir

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    if language:
        orc.set_language(language)

    conv_id = conversation_id or orc.create_conversation()

    target_sr = orc.expected_sample_rate or 16000
    try:
        pcm_bytes, audio_meta = webm_to_pcm(audio_bytes, target_sample_rate=target_sr)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {exc}") from exc

    def run_fn(on_stage):
        return orc.process_audio_turn(
            conversation_id=conv_id,
            pcm_bytes=pcm_bytes,
            audio_meta=audio_meta,
            conversation_history=conversation_history,
            corrections_enabled=corrections_enabled,
            on_stage=on_stage,
        )

    return EventSourceResponse(_build_sse_generator(orc, cache_root, run_fn))
