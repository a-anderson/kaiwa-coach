"""Translation route.

POST /api/turns/{assistant_turn_id}/translate — translate an assistant reply on demand.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.utils import _ML_EXECUTOR
from kaiwacoach.orchestrator import ConversationOrchestrator

router = APIRouter()

_DEFAULT_TARGET_LANGUAGE = "English"


class TranslateRequest(BaseModel):
    target_language: str = _DEFAULT_TARGET_LANGUAGE


@router.post("/turns/{assistant_turn_id}/translate")
async def translate_turn(
    assistant_turn_id: str,
    body: TranslateRequest,
    request: Request,
) -> dict:
    """Translate the reply text of an assistant turn.

    Returns ``{"translation": "<translated text>"}`` on success.
    Raises 404 if the turn does not exist, 422 if translation fails.
    """
    orc: ConversationOrchestrator = get_orchestrator(request)
    loop = asyncio.get_running_loop()

    def run_sync() -> str:
        return orc.translate_assistant_turn(
            assistant_turn_id=assistant_turn_id,
            target_language=body.target_language,
        )

    try:
        translation = await loop.run_in_executor(_ML_EXECUTOR, run_sync)
    except ValueError as exc:
        msg = str(exc)
        if "Unknown assistant_turn_id" in msg:
            raise HTTPException(status_code=404, detail="Turn not found") from exc
        raise HTTPException(status_code=422, detail=f"Translation failed: {exc}") from exc

    return {"translation": translation}
