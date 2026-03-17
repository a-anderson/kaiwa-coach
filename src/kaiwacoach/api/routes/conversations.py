"""Conversation CRUD routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.api.schemas.conversation import (
    ConversationDetail,
    ConversationSummary,
    CorrectionData,
    CreateConversationRequest,
    TurnRecord,
)
from kaiwacoach.orchestrator import ConversationOrchestrator

router = APIRouter()


def _audio_path_to_url(abs_path: str | None, cache_root: Path) -> str | None:
    """Convert an absolute audio path to a serveable API URL, or None if missing."""
    if not abs_path:
        return None
    p = Path(abs_path)
    if not p.exists():
        return None
    try:
        rel = p.relative_to(cache_root)
        return f"/api/audio/{rel}"
    except ValueError:
        return None


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

    return TurnRecord(
        user_turn_id=turn["user_turn_id"],
        assistant_turn_id=turn.get("assistant_turn_id"),
        user_text=user_text,
        asr_text=turn.get("asr_text"),
        reply_text=turn.get("reply_text"),
        correction=correction,
        # Audio availability: Phase 2 sets real paths; for now resolve what exists.
        has_user_audio=bool(turn.get("user_audio_path")),
        has_assistant_audio=bool(turn.get("assistant_audio_path")),
        user_audio_url=_audio_path_to_url(turn.get("user_audio_path"), cache_root),
        assistant_audio_url=_audio_path_to_url(turn.get("assistant_audio_path"), cache_root),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> list[ConversationSummary]:
    return [ConversationSummary(**c) for c in orc.list_conversations()]


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
    )


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
        raise HTTPException(status_code=500, detail="Failed to retrieve created conversation")
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
