"""FastAPI application factory and Uvicorn launcher."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from kaiwacoach.api.routes.audio import router as audio_router
from kaiwacoach.api.routes.conversations import router as conversations_router
from kaiwacoach.api.routes.monologue import router as monologue_router
from kaiwacoach.api.routes.regen import router as regen_router
from kaiwacoach.api.routes.narration import router as narration_router
from kaiwacoach.api.routes.settings import router as settings_router
from kaiwacoach.api.routes.translate import router as translate_router
from kaiwacoach.api.routes.turns import router as turns_router
from kaiwacoach.api.schemas.conversation import SetLanguageRequest
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.storage.blobs import SessionAudioCache
from kaiwacoach.storage.db import SQLiteWriter

_logger = logging.getLogger(__name__)

# Resolved at import time; valid once the repo layout is in place.
_STATIC_DIR = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def create_app(
    orchestrator: ConversationOrchestrator,
    audio_cache: SessionAudioCache,
    db: SQLiteWriter,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    orchestrator:
        The fully constructed ConversationOrchestrator singleton.
    audio_cache:
        Session audio cache; cleaned up on shutdown.
    db:
        SQLiteWriter; closed on shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = orchestrator
        app.state.audio_cache = audio_cache
        app.state.db = db
        _logger.info("KaiwaCoach API started")
        yield
        _logger.info("KaiwaCoach API shutting down")
        db.close()
        audio_cache.cleanup()

    app = FastAPI(title="KaiwaCoach", version="2.0.0", lifespan=lifespan)

    # ── API routers ──────────────────────────────────────────────────────
    app.include_router(conversations_router, prefix="/api")
    app.include_router(audio_router, prefix="/api")
    app.include_router(turns_router, prefix="/api")
    app.include_router(monologue_router, prefix="/api")
    app.include_router(regen_router, prefix="/api")
    app.include_router(narration_router, prefix="/api")
    app.include_router(translate_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")

    # ── Session / settings endpoints ─────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings(request: Request) -> dict:
        orc: ConversationOrchestrator = request.app.state.orchestrator
        return {"language": orc.language}

    @app.post("/api/session/language", status_code=204)
    async def set_language(body: SetLanguageRequest, request: Request) -> None:
        orc: ConversationOrchestrator = request.app.state.orchestrator
        orc.set_language(body.language)

    @app.post("/api/session/reset", status_code=204)
    async def reset_session(request: Request) -> None:
        orc: ConversationOrchestrator = request.app.state.orchestrator
        orc.reset_session()

    # ── Static frontend ───────────────────────────────────────────────────
    if _STATIC_DIR.exists() and any(_STATIC_DIR.iterdir()):
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    else:
        @app.get("/")
        async def index() -> JSONResponse:
            return JSONResponse({
                "status": "KaiwaCoach API running",
                "note": "Frontend not built — run `cd frontend && npm run build`",
                "api_docs": "/docs",
            })

    return app


def run(app: FastAPI, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Uvicorn server."""
    uvicorn.run(app, host=host, port=port)
