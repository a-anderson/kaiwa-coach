"""KaiwaCoach application entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from kaiwacoach.models.factory import build_asr, build_llm, build_tts
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.settings import load_config
from kaiwacoach.storage.blobs import SessionAudioCache
from kaiwacoach.storage.db import SQLiteWriter

log = logging.getLogger(__name__)


def main(launch: bool = True, configure_logging: bool = True) -> None:
    """Load and validate configuration, then start the FastAPI application.

    Parameters
    ----------
    launch:
        When True (default), start the Uvicorn server. Set to False in tests
        to exercise component wiring without binding a port.
    configure_logging:
        When True (default), call ``logging.basicConfig``. Set to False when
        the caller has already configured logging.
    """
    if configure_logging:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            force=True,
        )

    config = load_config()
    log.info(
        "Models: ASR=%s | LLM=%s | TTS=%s",
        config.models.asr_id,
        config.models.llm_id,
        config.models.tts_id,
    )

    storage_root = Path(config.storage.root_dir)
    storage_root.mkdir(parents=True, exist_ok=True)
    db_path = storage_root / "kaiwacoach.sqlite"
    schema_path = Path(__file__).resolve().parent / "storage" / "schema.sql"

    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()

    audio_cache = SessionAudioCache(expected_sample_rate=config.storage.expected_sample_rate)
    asr = build_asr(config)
    llm = build_llm(config)
    tts = build_tts(config, audio_cache)
    prompt_loader = PromptLoader(Path(__file__).resolve().parent / "prompts")

    orchestrator = ConversationOrchestrator(
        db=db,
        llm=llm,
        prompt_loader=prompt_loader,
        language=config.session.language,
        tts=tts,
        tts_voice=config.tts.voice,
        tts_speed=config.tts.speed,
        asr=asr,
        audio_cache=audio_cache,
        timing_logs_enabled=config.logging.timing_logs,
    )

    if launch:
        from kaiwacoach.api.server import create_app, run
        app = create_app(orchestrator=orchestrator, audio_cache=audio_cache, db=db)
        run(app)


if __name__ == "__main__":
    main()
