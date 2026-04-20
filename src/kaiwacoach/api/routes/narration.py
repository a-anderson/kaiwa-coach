"""Narration route.

POST /api/narrate — synthesise arbitrary text to audio using the session TTS model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.utils import _ML_EXECUTOR, audio_path_to_url
from kaiwacoach.orchestrator import ConversationOrchestrator

router = APIRouter()


class NarrationRequest(BaseModel):
    text: str


@router.post("/narrate")
async def narrate(body: NarrationRequest, request: Request) -> dict:
    """Synthesise text to audio and return a playable URL.

    Returns ``{"audio_url": "<url>"}`` on success.
    Raises 400 on empty text, 422 if TTS is not configured.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Narration text is empty")

    orc: ConversationOrchestrator = get_orchestrator(request)
    cache_root: Path = request.app.state.audio_cache.root_dir
    loop = asyncio.get_running_loop()

    try:
        audio_path = await loop.run_in_executor(_ML_EXECUTOR, orc.generate_narration, body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="TTS not configured") from exc

    return {"audio_url": audio_path_to_url(audio_path, cache_root)}
