"""Pydantic schemas for turn submission (Phase 2)."""

from __future__ import annotations

from pydantic import BaseModel


class TurnTextRequest(BaseModel):
    conversation_id: str | None = None
    language: str = "ja"
    text: str
    conversation_history: str = ""
    corrections_enabled: bool = True


class TurnAudioRequest(BaseModel):
    """Metadata fields sent alongside a multipart audio upload."""
    conversation_id: str | None = None
    language: str = "ja"
    conversation_history: str = ""
    corrections_enabled: bool = True
