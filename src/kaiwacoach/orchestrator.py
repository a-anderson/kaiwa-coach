"""Conversation orchestration for text/audio turns with persistence.

Notes
-----
This module only stores schema-validation metadata in `assistant_turns.llm_meta_json`.
We intentionally omit backend-specific generation metadata for now to keep persistence
minimal; this trades off debugging/observability for simpler storage.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from kaiwacoach.models.asr_whisper import ASRResult, WhisperASR
from kaiwacoach.models.json_enforcement import ParseResult
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.tts_kokoro import KokoroTTS, TTSResult
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache
from kaiwacoach.storage.db import SQLiteWriter
from kaiwacoach.textnorm.invariants import enforce_japanese_invariant
from kaiwacoach.textnorm.jp_katakana import normalise_katakana
from kaiwacoach.textnorm.tts_punctuation import normalize_for_tts


@dataclass(frozen=True)
class TextTurnResult:
    conversation_id: str
    user_turn_id: str
    assistant_turn_id: str
    reply_text: str
    tts_audio_path: str | None = None


@dataclass(frozen=True)
class AudioTurnResult:
    conversation_id: str
    user_turn_id: str
    assistant_turn_id: str
    reply_text: str
    input_audio_path: str
    asr_text: str
    asr_meta: Dict[str, Any]
    tts_audio_path: str | None = None


class ConversationOrchestrator:
    """Orchestrator for text/audio turns, corrections, and TTS with persistence."""

    def __init__(
        self,
        db: SQLiteWriter,
        llm: QwenLLM,
        prompt_loader: PromptLoader,
        language: str,
        tts: KokoroTTS | None = None,
        tts_voice: str | None = None,
        tts_speed: float = 1.0,
        asr: WhisperASR | None = None,
        audio_cache: SessionAudioCache | None = None,
    ) -> None:
        self._db = db
        self._llm = llm
        self._prompt_loader = prompt_loader
        self._language = language
        self._tts = tts
        self._tts_voice = tts_voice
        self._tts_speed = tts_speed
        self._asr = asr
        self._audio_cache = audio_cache

    def create_conversation(self, title: Optional[str] = None) -> str:
        conversation_id = str(uuid.uuid4())

        def _insert(conn) -> None:
            conn.execute(
                """
                INSERT INTO conversations (id, title, language, model_metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, title, self._language, "{}"),
            )
            conn.commit()

        self._db.run_write(_insert)
        return conversation_id

    def process_text_turn(
        self,
        conversation_id: str,
        user_text: str,
        conversation_history: str = "",
    ) -> TextTurnResult:
        user_turn_id = str(uuid.uuid4())

        def _insert_user(conn) -> None:
            conn.execute(
                """
                INSERT INTO user_turns (id, conversation_id, input_text)
                VALUES (?, ?, ?)
                """,
                (user_turn_id, conversation_id, user_text),
            )
            conn.commit()

        self._db.run_write(_insert_user)

        assistant_turn_id, reply_text, tts_result = self._process_text_flow(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            user_text=user_text,
            conversation_history=conversation_history,
        )

        return TextTurnResult(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
            tts_audio_path=tts_result.audio_path if tts_result is not None else None,
        )

    def process_audio_turn(
        self,
        conversation_id: str,
        pcm_bytes: bytes,
        audio_meta: AudioMeta,
        conversation_history: str = "",
    ) -> AudioTurnResult:
        if self._asr is None or self._audio_cache is None:
            raise ValueError("ASR and audio_cache must be configured for audio turns.")

        user_turn_id = str(uuid.uuid4())
        input_audio_path = self._audio_cache.save_audio(
            conversation_id=conversation_id,
            turn_id=user_turn_id,
            kind="user",
            pcm_bytes=pcm_bytes,
            meta=audio_meta,
        )
        try:
            asr_result = self._asr.transcribe(input_audio_path)
            asr_meta = dict(asr_result.meta)
            asr_meta["audio_path"] = str(input_audio_path)
            asr_meta_json = json.dumps(asr_meta, ensure_ascii=False)

            def _insert_user(conn) -> None:
                conn.execute(
                    """
                    INSERT INTO user_turns (id, conversation_id, input_text, asr_text, asr_meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_turn_id, conversation_id, None, asr_result.text, asr_meta_json),
                )
                conn.commit()

            self._db.run_write(_insert_user)

            assistant_turn_id, reply_text, tts_result = self._process_text_flow(
                conversation_id=conversation_id,
                user_turn_id=user_turn_id,
                user_text=asr_result.text,
                conversation_history=conversation_history,
            )

            return AudioTurnResult(
                conversation_id=conversation_id,
                user_turn_id=user_turn_id,
                assistant_turn_id=assistant_turn_id,
                reply_text=reply_text,
                input_audio_path=str(input_audio_path),
                asr_text=asr_result.text,
                asr_meta=asr_meta,
                tts_audio_path=tts_result.audio_path if tts_result is not None else None,
            )
        except Exception as exc:  # pragma: no cover - exercised in tests via fake ASR
            asr_meta = {
                "audio_path": str(input_audio_path),
                "error": str(exc),
            }
            asr_meta_json = json.dumps(asr_meta, ensure_ascii=False)

            def _insert_user_failed(conn) -> None:
                conn.execute(
                    """
                    INSERT INTO user_turns (id, conversation_id, input_text, asr_text, asr_meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_turn_id, conversation_id, None, None, asr_meta_json),
                )
                conn.commit()

            self._db.run_write(_insert_user_failed)

            return AudioTurnResult(
                conversation_id=conversation_id,
                user_turn_id=user_turn_id,
                assistant_turn_id="",
                reply_text="",
                input_audio_path=str(input_audio_path),
                asr_text="",
                asr_meta=asr_meta,
                tts_audio_path=None,
            )

    def _process_text_flow(
        self,
        conversation_id: str,
        user_turn_id: str,
        user_text: str,
        conversation_history: str,
    ) -> tuple[str, str, TTSResult | None]:
        prompt = self._prompt_loader.render(
            "conversation.md",
            {
                "language": self._language,
                "conversation_history": conversation_history,
                "user_text": user_text,
            },
        )
        parsed = self._safe_generate_json(prompt=prompt.text, role="conversation")
        reply_text = ""
        if parsed.model is not None:
            reply_text = getattr(parsed.model, "reply", "")
        llm_meta: Dict[str, Any] = {
            "role": "conversation",
            "schema_valid": parsed.model is not None,
        }
        if parsed.error:
            llm_meta["error"] = parsed.error
        if parsed.repaired:
            llm_meta["repaired"] = True
        llm_meta["status"] = "assistant_persisted"

        assistant_turn_id = str(uuid.uuid4())
        llm_meta["prompt_hash"] = prompt.sha256
        llm_meta_json = json.dumps(llm_meta, ensure_ascii=False)

        def _insert_assistant(conn) -> None:
            conn.execute(
                """
                INSERT INTO assistant_turns (id, user_turn_id, conversation_id, reply_text, llm_meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (assistant_turn_id, user_turn_id, conversation_id, reply_text, llm_meta_json),
            )
            conn.commit()

        self._db.run_write(_insert_assistant)

        self._process_corrections(user_turn_id=user_turn_id, user_text=user_text)
        self._update_assistant_meta(
            assistant_turn_id,
            {"corrections_persisted": True},
        )
        tts_result = self._process_tts(
            conversation_id=conversation_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
        )
        return assistant_turn_id, reply_text, tts_result

    def _process_corrections(self, user_turn_id: str, user_text: str) -> None:
        detect_prompt = self._prompt_loader.render(
            "detect_errors.md",
            {"language": self._language, "user_text": user_text},
        )
        detect_result = self._safe_generate_json(prompt=detect_prompt.text, role="error_detection")
        errors = []
        if detect_result.model is not None:
            errors = list(getattr(detect_result.model, "errors", []))

        correction_prompt = self._prompt_loader.render(
            "correct_sentence.md",
            {"language": self._language, "user_text": user_text},
        )
        correction_result = self._safe_generate_json(prompt=correction_prompt.text, role="correction")
        corrected_text = user_text
        if correction_result.model is not None:
            corrected_text = getattr(correction_result.model, "corrected", user_text)

        explain_prompt = self._prompt_loader.render(
            "explain.md",
            {"language": self._language, "user_text": user_text, "corrected_text": corrected_text},
        )
        explain_result = self._safe_generate_json(prompt=explain_prompt.text, role="explanation")
        explanation_text = ""
        if explain_result.model is not None:
            explanation_text = getattr(explain_result.model, "explanation", "")

        native_prompt = self._prompt_loader.render(
            "native_rewrite.md",
            {"language": self._language, "user_text": user_text},
        )
        native_result = self._safe_generate_json(prompt=native_prompt.text, role="native_reformulation")
        native_text = None
        if native_result.model is not None:
            native_text = getattr(native_result.model, "native", None)

        correction_id = str(uuid.uuid4())
        errors_json = json.dumps(errors, ensure_ascii=False)
        prompt_hash = correction_prompt.sha256

        def _insert_correction(conn) -> None:
            conn.execute(
                """
                INSERT INTO corrections (id, user_turn_id, errors_json, corrected_text, native_text, explanation_text, prompt_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    correction_id,
                    user_turn_id,
                    errors_json,
                    corrected_text,
                    native_text,
                    explanation_text,
                    prompt_hash,
                ),
            )
            conn.commit()

        self._db.run_write(_insert_correction)

    def _process_tts(
        self,
        conversation_id: str,
        assistant_turn_id: str,
        reply_text: str,
    ) -> TTSResult | None:
        if self._tts is None:
            return None

        try:
            normalised_text = self._normalise_for_tts(reply_text)
            voice = self._tts_voice
            if voice == "default":
                voice = None

            return self._tts.synthesize(
                conversation_id=conversation_id,
                turn_id=assistant_turn_id,
                text=normalised_text,
                voice=voice,
                speed=self._tts_speed,
                language=self._language,
            )
        except Exception:
            return None

    def regenerate_turn_audio(self, assistant_turn_id: str) -> TTSResult:
        """Regenerate TTS audio for a single assistant turn."""
        if self._tts is None:
            raise ValueError("TTS must be configured to regenerate audio.")

        def _fetch(conn) -> tuple[str, str]:
            row = conn.execute(
                "SELECT conversation_id, reply_text FROM assistant_turns WHERE id = ?",
                (assistant_turn_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown assistant_turn_id: {assistant_turn_id}")
            return row[0], row[1]

        with self._db.read_connection() as conn:
            conversation_id, reply_text = _fetch(conn)
        result = self._process_tts(
            conversation_id=conversation_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
        )
        if result is None:
            raise RuntimeError("TTS regeneration failed.")
        return result

    def regenerate_conversation_audio(self, conversation_id: str) -> list[TTSResult]:
        """Regenerate TTS audio for every assistant turn in a conversation."""
        if self._tts is None:
            raise ValueError("TTS must be configured to regenerate audio.")

        def _fetch(conn) -> list[tuple[str, str]]:
            rows = conn.execute(
                """
                SELECT id, reply_text
                FROM assistant_turns
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
            return [(row[0], row[1]) for row in rows]

        with self._db.read_connection() as conn:
            turns = _fetch(conn)
        results: list[TTSResult] = []
        for turn_id, reply_text in turns:
            result = self._process_tts(
                conversation_id=conversation_id,
                assistant_turn_id=turn_id,
                reply_text=reply_text,
            )
            if result is None:
                raise RuntimeError("TTS regeneration failed.")
            results.append(result)
        return results

    def _normalise_for_tts(self, text: str) -> str:
        if self._language != "ja":
            return text

        def _llm_rewrite(content: str) -> str:
            prompt = self._prompt_loader.render("jp_tts_normalise.md", {"original_text": content})
            result = self._safe_generate_json(prompt=prompt.text, role="jp_tts_normalisation")
            if result.model is None:
                return content
            return getattr(result.model, "text", content)

        katakana_result = normalise_katakana(text, _llm_rewrite)
        candidate, _ = enforce_japanese_invariant(text, katakana_result.text)
        return normalize_for_tts(candidate)

    def _safe_generate_json(self, prompt: str, role: str) -> ParseResult:
        try:
            return self._llm.generate_json(prompt=prompt, role=role)
        except Exception as exc:
            return ParseResult(model=None, raw_json=None, error=str(exc), repaired=False)

    def _update_assistant_meta(self, assistant_turn_id: str, updates: Dict[str, Any]) -> None:
        def _update(conn) -> None:
            row = conn.execute(
                "SELECT llm_meta_json FROM assistant_turns WHERE id = ?",
                (assistant_turn_id,),
            ).fetchone()
            if row is None:
                return
            meta = {}
            if row[0]:
                meta = json.loads(row[0])
            meta.update(updates)
            conn.execute(
                """
                UPDATE assistant_turns
                SET llm_meta_json = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (json.dumps(meta, ensure_ascii=False), assistant_turn_id),
            )
            conn.commit()

        self._db.run_write(_update)
