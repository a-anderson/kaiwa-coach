"""User profile settings routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from kaiwacoach.api.deps import get_orchestrator
from kaiwacoach.constants import VALID_PROFICIENCY_LEVELS
from kaiwacoach.orchestrator import ConversationOrchestrator

router = APIRouter()


class UserProfileRequest(BaseModel):
    user_name: str | None = None
    language_proficiency: dict[str, str] = {}


@router.get("/settings/profile")
async def get_profile(
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> dict:
    return orc.get_user_profile()


@router.post("/settings/profile", status_code=204)
async def set_profile(
    body: UserProfileRequest,
    orc: ConversationOrchestrator = Depends(get_orchestrator),
) -> None:
    for language, level in body.language_proficiency.items():
        valid_levels = VALID_PROFICIENCY_LEVELS.get(language)
        if valid_levels is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown language '{language}'",
            )
        if level not in valid_levels:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid level '{level}' for language '{language}'",
            )
    orc.set_user_profile(
        user_name=body.user_name,
        language_proficiency=body.language_proficiency,
    )
