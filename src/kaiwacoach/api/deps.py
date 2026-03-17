"""FastAPI dependency injection helpers."""

from __future__ import annotations

from fastapi import Request

from kaiwacoach.orchestrator import ConversationOrchestrator


def get_orchestrator(request: Request) -> ConversationOrchestrator:
    """Return the orchestrator singleton from app state."""
    return request.app.state.orchestrator
