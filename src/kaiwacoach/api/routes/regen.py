"""Audio regeneration routes.

POST /api/turns/{assistant_turn_id}/regen-audio        — regen one turn
POST /api/conversations/{conversation_id}/regen-audio  — regen all turns (SSE)
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.utils import audio_path_to_url
from kaiwacoach.orchestrator import ConversationOrchestrator

_logger = logging.getLogger(__name__)

router = APIRouter()

# Shared with turns router — serialises all ML work through one thread.
_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/turns/{assistant_turn_id}/regen-audio")
async def regen_turn_audio(assistant_turn_id: str, request: Request) -> dict:
    """Regenerate TTS for a single assistant turn.

    Returns ``{ "audio_url": "<url>" }`` on success.
    Raises 404 if the turn does not exist, 422 if TTS is not configured.
    """
    orc: ConversationOrchestrator = get_orchestrator(request)
    cache_root: Path = request.app.state.audio_cache.root_dir

    loop = asyncio.get_running_loop()

    def run_sync():
        return orc.regenerate_turn_audio(assistant_turn_id)

    try:
        result = await loop.run_in_executor(_executor, run_sync)
    except ValueError as exc:
        status = 404 if "Unknown assistant_turn_id" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"audio_url": audio_path_to_url(result.audio_path, cache_root)}


@router.post("/conversations/{conversation_id}/regen-audio")
async def regen_conversation_audio(conversation_id: str, request: Request) -> EventSourceResponse:
    """Regenerate TTS for all assistant turns in a conversation.

    Streams SSE events::

        { "event": "turn_done", "data": {"assistant_turn_id": "...", "audio_url": "..."} }
        ...
        { "event": "complete", "data": {} }

    or on error::

        { "event": "error", "data": {"message": "..."} }
    """
    orc: ConversationOrchestrator = get_orchestrator(request)
    cache_root: Path = request.app.state.audio_cache.root_dir

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_turn(turn_id: str, audio_path: str | None) -> None:
        payload = {
            "assistant_turn_id": turn_id,
            "audio_url": audio_path_to_url(audio_path, cache_root),
        }
        loop.call_soon_threadsafe(queue.put_nowait, {"_turn": payload})

    def run_sync() -> None:
        try:
            orc.regenerate_conversation_audio(conversation_id, on_turn=on_turn)
            loop.call_soon_threadsafe(queue.put_nowait, {"_done": True})
        except Exception as exc:  # noqa: BLE001
            _logger.exception("Conversation audio regen failed")
            loop.call_soon_threadsafe(queue.put_nowait, {"_done": True, "error": str(exc)})

    loop.run_in_executor(_executor, run_sync)

    async def event_generator():
        while True:
            msg = await queue.get()
            if msg.get("_done"):
                if "error" in msg:
                    yield {"event": "error", "data": json.dumps({"message": msg["error"]})}
                else:
                    yield {"event": "complete", "data": "{}"}
                break
            elif "_turn" in msg:
                yield {"event": "turn_done", "data": json.dumps(msg["_turn"])}

    return EventSourceResponse(event_generator())
