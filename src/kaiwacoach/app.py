"""KaiwaCoach application entrypoint."""

from __future__ import annotations

import atexit

from kaiwacoach.config import load_config
from kaiwacoach.storage.blobs import SessionAudioCache


def main() -> None:
    """Load and validate configuration, then start the application."""
    _ = load_config()
    audio_cache = SessionAudioCache()
    atexit.register(audio_cache.cleanup)
    # TODO: initialize UI and orchestrator.


if __name__ == "__main__":
    main()
