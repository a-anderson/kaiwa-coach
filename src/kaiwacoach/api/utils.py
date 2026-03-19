"""Shared utilities for API route handlers."""

from __future__ import annotations

from pathlib import Path


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
