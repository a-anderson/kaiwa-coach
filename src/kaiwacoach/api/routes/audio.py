"""Audio file serving route."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/audio/{audio_path:path}")
async def serve_audio(audio_path: str, request: Request) -> FileResponse:
    """Serve a WAV file from the session audio cache.

    The path is relative to the cache root (e.g. ``conv_id/turn_id/tts_hash.wav``).
    Absolute paths and traversal attempts are rejected.
    """
    cache_root: Path = request.app.state.audio_cache.root_dir.resolve()
    full_path = (cache_root / audio_path).resolve()

    # Reject any path that escapes the cache root.
    try:
        full_path.relative_to(cache_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid audio path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(full_path), media_type="audio/wav")
