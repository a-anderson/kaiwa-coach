"""SSE monologue routes.

POST /api/turns/monologue/text  — submit a text monologue turn, stream stage events + complete
POST /api/turns/monologue/audio — submit a WebM audio monologue turn, stream stage events + complete
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from kaiwacoach.api.audio_conversion import webm_to_pcm
from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.utils import _ML_EXECUTOR
from kaiwacoach.orchestrator import ConversationOrchestrator, MonologueTurnResult

_logger = logging.getLogger(__name__)

router = APIRouter()


class MonologueTextRequest(BaseModel):
    conversation_id: str
    text: str


def _build_monologue_sse_generator(orc: ConversationOrchestrator, run_fn):
    """Return an async generator that offloads *run_fn* and yields SSE dicts.

    *run_fn* must accept a single ``on_stage`` callback and return a
    ``MonologueTurnResult``. Unlike the turn SSE generator there is no TTS
    stage, so no audio path conversion is needed.
    """
    request_id = uuid.uuid4().hex

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_stage(stage: str, status: str, data: dict) -> None:
        payload = {"stage": stage, "status": status, **data}
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def run_sync() -> None:
        try:
            result = run_fn(on_stage)
            loop.call_soon_threadsafe(queue.put_nowait, {"_done": True, "result": result})
        except Exception as exc:  # noqa: BLE001 — background thread; all errors must be funnelled to the SSE queue
            _logger.exception("Monologue processing failed request_id=%s", request_id)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"_done": True, "error": str(exc), "request_id": request_id},
            )

    loop.run_in_executor(_ML_EXECUTOR, run_sync)

    async def event_generator():
        while True:
            msg = await queue.get()
            if msg.get("_done"):
                if "error" in msg:
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "message": msg["error"],
                            "request_id": msg["request_id"],
                        }),
                    }
                else:
                    result: MonologueTurnResult = msg["result"]
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "conversation_id": result.conversation_id,
                            "user_turn_id": result.user_turn_id,
                            "input_text": result.input_text,
                            "asr_text": result.asr_text,
                            "corrections": result.corrections,
                            "summary": result.summary,
                        }),
                    }
                break
            else:
                yield {"event": "stage", "data": json.dumps(msg)}

    return event_generator()


@router.post("/turns/monologue/text")
async def submit_monologue_text(
    body: MonologueTextRequest,
    request: Request,
) -> EventSourceResponse:
    """Submit a text monologue turn and stream pipeline progress as SSE."""
    orc: ConversationOrchestrator = get_orchestrator(request)

    def run_fn(on_stage):
        return orc.process_monologue_turn(
            conversation_id=body.conversation_id,
            text=body.text,
            on_stage=on_stage,
        )

    return EventSourceResponse(_build_monologue_sse_generator(orc, run_fn))


@router.post("/turns/monologue/audio")
async def submit_monologue_audio(
    request: Request,
    audio: UploadFile,
    conversation_id: str = Form(),
) -> EventSourceResponse:
    """Submit a WebM audio monologue turn and stream pipeline progress as SSE."""
    orc: ConversationOrchestrator = get_orchestrator(request)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    target_sr = orc.expected_sample_rate or 16000
    try:
        pcm_bytes, audio_meta = webm_to_pcm(audio_bytes, target_sample_rate=target_sr)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {exc}") from exc

    def run_fn(on_stage):
        return orc.process_monologue_turn(
            conversation_id=conversation_id,
            pcm_bytes=pcm_bytes,
            audio_meta=audio_meta,
            on_stage=on_stage,
        )

    return EventSourceResponse(_build_monologue_sse_generator(orc, run_fn))
