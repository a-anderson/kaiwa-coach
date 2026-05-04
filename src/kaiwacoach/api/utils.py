"""Shared utilities for API route handlers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# One thread for all ML work. The orchestrator holds model state and is not
# safe for concurrent processing — a single shared executor serialises all
# turn and regen requests naturally.
ML_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def audio_path_to_url(audio_path: str | None, cache_root: Path) -> str | None:
    """Return a serveable /api/audio URL for *audio_path*, or None if missing."""
    if not audio_path:
        return None
    p = Path(audio_path)
    if not p.exists():
        return None
    try:
        rel = p.relative_to(cache_root)
        return f"/api/audio/{rel}"
    except ValueError:
        return None
