"""KaiwaCoach application entrypoint."""

from __future__ import annotations

import atexit

from kaiwacoach.settings import load_config
from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.storage.blobs import SessionAudioCache


def main() -> None:
    """Load and validate configuration, then start the application."""
    config = load_config()
    audio_cache = SessionAudioCache()
    _ = WhisperASR(
        model_id=config.models.asr_id,
        language=config.session.language,
    )
    atexit.register(audio_cache.cleanup)
    # TODO: initialize UI and orchestrator.


if __name__ == "__main__":
    main()
