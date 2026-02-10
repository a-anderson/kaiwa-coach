"""KaiwaCoach application entrypoint."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path

from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.llm_qwen import MlxLmBackend, QwenLLM
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.settings import load_config
from kaiwacoach.storage.blobs import SessionAudioCache
from kaiwacoach.storage.db import SQLiteWriter
from kaiwacoach.ui.gradio_app import build_ui


def main(launch_ui: bool = True) -> None:
    """Load and validate configuration, then start the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    config = load_config()
    storage_root = Path(config.storage.root_dir)
    storage_root.mkdir(parents=True, exist_ok=True)
    db_path = storage_root / "kaiwacoach.sqlite"
    schema_path = Path(__file__).resolve().parent / "storage" / "schema.sql"
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    atexit.register(db.close)

    audio_cache = SessionAudioCache(expected_sample_rate=config.storage.expected_sample_rate)
    asr = WhisperASR(
        model_id=config.models.asr_id,
        language=config.session.language,
    )
    llm_backend = MlxLmBackend(config.models.llm_id)
    llm = QwenLLM(
        model_id=config.models.llm_id,
        max_context_tokens=config.llm.max_context_tokens,
        role_max_new_tokens=config.llm.role_max_new_tokens.__dict__,
        backend=llm_backend,
        token_counter=llm_backend.count_tokens,
    )
    tts = KokoroTTS(
        model_id=config.models.tts_id,
        cache=audio_cache,
    )
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
    atexit.register(audio_cache.cleanup)
    if launch_ui:
        demo = build_ui(orchestrator)
        demo.launch()


if __name__ == "__main__":
    main()
