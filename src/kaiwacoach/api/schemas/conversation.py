"""Pydantic schemas for conversation-related API request and response bodies."""

from __future__ import annotations

from pydantic import BaseModel


class CorrectionData(BaseModel):
    errors: list[str]
    corrected: str
    native: str
    explanation: str


class TurnRecord(BaseModel):
    user_turn_id: str
    assistant_turn_id: str | None
    user_text: str | None       # input_text (typed) or asr_text for audio turns
    asr_text: str | None        # only set for audio turns
    reply_text: str | None
    correction: CorrectionData | None
    has_user_audio: bool
    has_assistant_audio: bool
    user_audio_url: str | None
    assistant_audio_url: str | None


class ConversationSummary(BaseModel):
    id: str
    title: str | None
    language: str
    updated_at: str | None
    preview_text: str | None
    conversation_type: str = "chat"


class ConversationDetail(BaseModel):
    id: str
    title: str | None
    language: str
    created_at: str | None
    updated_at: str | None
    turns: list[TurnRecord]
    conversation_type: str = "chat"


class CreateConversationRequest(BaseModel):
    language: str | None = None
    title: str | None = None


class SetLanguageRequest(BaseModel):
    language: str
