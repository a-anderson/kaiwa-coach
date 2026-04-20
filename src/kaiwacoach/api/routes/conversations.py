"""Conversation CRUD routes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.schemas.conversation import (
    ConversationDetail,
    ConversationSummary,
    CorrectionData,
    CreateConversationRequest,
    TurnRecord,
)
from kaiwacoach.api.utils import audio_path_to_url
from kaiwacoach.orchestrator import ConversationOrchestrator

router = APIRouter()


def _build_turn_record(
    turn: dict,
    orc: ConversationOrchestrator,
    cache_root: Path,
) -> TurnRecord:
    """Enrich a raw turn dict with corrections and audio URL resolution."""
    raw_corrections = orc.get_latest_corrections(turn["user_turn_id"])
    has_corrections = any([
        raw_corrections.get("errors"),
        raw_corrections.get("corrected"),
        raw_corrections.get("native"),
        raw_corrections.get("explanation"),
    ])
    correction = CorrectionData(**raw_corrections) if has_corrections else None

    # For text turns input_text is set; for audio turns asr_text is set.
    user_text = turn.get("input_text") or turn.get("asr_text")

    # Resolve audio URLs first; derive has_* from whether the file exists on disk.
    user_audio_url = audio_path_to_url(turn.get("user_audio_path"), cache_root)
    assistant_audio_url = audio_path_to_url(turn.get("assistant_audio_path"), cache_root)

    return TurnRecord(
        user_turn_id=turn["user_turn_id"],
        assistant_turn_id=turn.get("assistant_turn_id"),
        user_text=user_text,
        asr_text=turn.get("asr_text"),
        reply_text=turn.get("reply_text"),
        correction=correction,
        has_user_audio=user_audio_url is not None,
        has_assistant_audio=assistant_audio_url is not None,
        user_audio_url=user_audio_url,
        assistant_audio_url=assistant_audio_url,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    type: Optional[str] = Query(default=None),
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> list[ConversationSummary]:
    return [ConversationSummary(**c) for c in orc.list_conversations(conversation_type=type)]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    request: Request,
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> ConversationDetail:
    try:
        convo = orc.get_conversation(conversation_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    cache_root: Path = request.app.state.audio_cache.root_dir
    turns = [_build_turn_record(t, orc, cache_root) for t in convo["turns"]]
    return ConversationDetail(
        id=convo["id"],
        title=convo.get("title"),
        language=convo["language"],
        created_at=convo.get("created_at"),
        updated_at=convo.get("updated_at"),
        turns=turns,
        conversation_type=convo.get("conversation_type", "chat"),
    )


@router.post("/conversations/monologue", status_code=201)
async def create_monologue_conversation(
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> dict:
    conversation_id = orc.create_monologue_conversation()
    return {"conversation_id": conversation_id}


@router.post("/conversations", response_model=ConversationSummary, status_code=201)
async def create_conversation(
    body: CreateConversationRequest,
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> ConversationSummary:
    if body.language:
        orc.set_language(body.language)
    conv_id = orc.create_conversation(title=body.title)
    # Fetch the summary for the newly created conversation.
    all_convos = orc.list_conversations()
    created = next((c for c in all_convos if c["id"] == conv_id), None)
    if created is None:
        raise HTTPException(status_code=500, detail="Created conversation not found")
    return ConversationSummary(**created)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> None:
    orc.delete_conversation(conversation_id)


@router.delete("/conversations", status_code=204)
async def delete_all_conversations(
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> None:
    orc.delete_all_conversations()
