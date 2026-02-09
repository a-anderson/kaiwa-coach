"""Conversation orchestration for text/audio turns with persistence.

Notes
-----
This module only stores schema-validation metadata in `assistant_turns.llm_meta_json`.
We intentionally omit backend-specific generation metadata for now to keep persistence
minimal; this trades off debugging/observability for simpler storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from kaiwacoach.models.asr_whisper import ASRResult, WhisperASR
from kaiwacoach.models.json_enforcement import ParseResult, parse_with_schema
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
        timing_logs_enabled: bool = True,
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
        self._asr_cache: Dict[tuple[str, str, str], ASRResult] = {}
        self._logger = logging.getLogger(__name__)
        self._timing_logs_enabled = timing_logs_enabled

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
        timings: Dict[str, float] = {}
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

        start = time.perf_counter()
        self._db.run_write(_insert_user)
        timings["user_insert_seconds"] = time.perf_counter() - start

        assistant_turn_id, reply_text = self.generate_reply(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            user_text=user_text,
            conversation_history=conversation_history,
            timings=timings,
        )
        corrections = self.run_corrections(
            user_turn_id,
            user_text,
            assistant_turn_id=assistant_turn_id,
            timings=timings,
        )
        tts_result = self.run_tts(
            conversation_id=conversation_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
            timings=timings,
        )
        self._finalize_timings(timings)
        self._log_timings("text_turn", timings)

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

        timings: Dict[str, float] = {}
        user_turn_id = str(uuid.uuid4())
        audio_hash = hashlib.sha256(pcm_bytes).hexdigest()
        start = time.perf_counter()
        input_audio_path = self._audio_cache.save_audio(
            conversation_id=conversation_id,
            turn_id=user_turn_id,
            kind="user",
            pcm_bytes=pcm_bytes,
            meta=audio_meta,
        )
        timings["audio_save_seconds"] = time.perf_counter() - start
        try:
            model_id = getattr(self._asr, "model_id", None) or getattr(self._asr, "_model_id", "unknown")
            language = getattr(self._asr, "language", None) or getattr(self._asr, "_language", self._language)
            cache_key = (audio_hash, str(model_id), str(language))
            cached = self._asr_cache.get(cache_key)
            if cached is not None:
                timings["asr_cache_seconds"] = 0.0
                asr_meta = dict(cached.meta)
                asr_meta.update(
                    {
                        "audio_path": str(input_audio_path),
                        "audio_hash": audio_hash,
                        "cache_hit": True,
                    }
                )
                asr_result = ASRResult(text=cached.text, meta=asr_meta)
            else:
                start = time.perf_counter()
                asr_result = self._asr.transcribe(input_audio_path)
                timings["asr_transcribe_seconds"] = time.perf_counter() - start
                asr_meta = dict(asr_result.meta)
                asr_meta.setdefault("audio_hash", audio_hash)
                asr_meta.setdefault("cache_hit", False)
                cached_meta = dict(asr_meta)
                cached_meta.pop("audio_path", None)
                self._asr_cache[cache_key] = ASRResult(text=asr_result.text, meta=cached_meta)
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

            start = time.perf_counter()
            self._db.run_write(_insert_user)
            timings["user_insert_seconds"] = time.perf_counter() - start

            assistant_turn_id, reply_text = self.generate_reply(
                conversation_id=conversation_id,
                user_turn_id=user_turn_id,
                user_text=asr_result.text,
                conversation_history=conversation_history,
                timings=timings,
            )
            corrections = self.run_corrections(
                user_turn_id,
                asr_result.text,
                assistant_turn_id=assistant_turn_id,
                timings=timings,
            )
            tts_result = self.run_tts(
                conversation_id=conversation_id,
                assistant_turn_id=assistant_turn_id,
                reply_text=reply_text,
                timings=timings,
            )
            self._finalize_timings(timings)
            self._log_timings("audio_turn", timings)

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

    def persist_user_text_turn(
        self,
        conversation_id: str,
        user_text: str,
        timings: Dict[str, float] | None = None,
    ) -> str:
        """Persist a text-only user turn and return its ID.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_text : str
            Raw user input text.
        timings : dict[str, float] | None
            Optional timings accumulator.

        Returns
        -------
        str
            Generated user turn ID.
        """
        if timings is None:
            timings = {}
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

        start = time.perf_counter()
        self._db.run_write(_insert_user)
        timings["user_insert_seconds"] = time.perf_counter() - start
        return user_turn_id

    def prepare_audio_turn(
        self,
        conversation_id: str,
        pcm_bytes: bytes,
        audio_meta: AudioMeta,
        timings: Dict[str, float] | None = None,
    ) -> AudioTurnResult:
        """Persist audio input, run ASR, and store user turn metadata.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        pcm_bytes : bytes
            Raw PCM audio bytes.
        audio_meta : AudioMeta
            Audio metadata describing the PCM bytes.
        timings : dict[str, float] | None
            Optional timings accumulator.

        Returns
        -------
        AudioTurnResult
            Partial audio turn result containing ASR output and audio path.
        """
        if self._asr is None or self._audio_cache is None:
            raise ValueError("ASR and audio_cache must be configured for audio turns.")
        if timings is None:
            timings = {}
        user_turn_id = str(uuid.uuid4())
        audio_hash = hashlib.sha256(pcm_bytes).hexdigest()
        start = time.perf_counter()
        input_audio_path = self._audio_cache.save_audio(
            conversation_id=conversation_id,
            turn_id=user_turn_id,
            kind="user",
            pcm_bytes=pcm_bytes,
            meta=audio_meta,
        )
        timings["audio_save_seconds"] = time.perf_counter() - start
        try:
            model_id = getattr(self._asr, "model_id", None) or getattr(self._asr, "_model_id", "unknown")
            language = getattr(self._asr, "language", None) or getattr(self._asr, "_language", self._language)
            cache_key = (audio_hash, str(model_id), str(language))
            cached = self._asr_cache.get(cache_key)
            if cached is not None:
                timings["asr_cache_seconds"] = 0.0
                asr_meta = dict(cached.meta)
                asr_meta.update(
                    {
                        "audio_path": str(input_audio_path),
                        "audio_hash": audio_hash,
                        "cache_hit": True,
                    }
                )
                asr_result = ASRResult(text=cached.text, meta=asr_meta)
            else:
                start = time.perf_counter()
                asr_result = self._asr.transcribe(input_audio_path)
                timings["asr_transcribe_seconds"] = time.perf_counter() - start
                asr_meta = dict(asr_result.meta)
                asr_meta.setdefault("audio_hash", audio_hash)
                asr_meta.setdefault("cache_hit", False)
                cached_meta = dict(asr_meta)
                cached_meta.pop("audio_path", None)
                self._asr_cache[cache_key] = ASRResult(text=asr_result.text, meta=cached_meta)
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

            start = time.perf_counter()
            self._db.run_write(_insert_user)
            timings["user_insert_seconds"] = time.perf_counter() - start
            return AudioTurnResult(
                conversation_id=conversation_id,
                user_turn_id=user_turn_id,
                assistant_turn_id="",
                reply_text="",
                input_audio_path=str(input_audio_path),
                asr_text=asr_result.text,
                asr_meta=asr_meta,
                tts_audio_path=None,
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

    def generate_reply(
        self,
        conversation_id: str,
        user_turn_id: str,
        user_text: str,
        conversation_history: str,
        timings: Dict[str, float] | None = None,
    ) -> tuple[str, str]:
        """Generate and persist the assistant reply.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        user_turn_id : str
            User turn identifier.
        user_text : str
            Text to respond to.
        conversation_history : str
            Prior conversation history.
        timings : dict[str, float] | None
            Optional timings accumulator.

        Returns
        -------
        tuple[str, str]
            Assistant turn ID and reply text.
        """
        if timings is None:
            timings = {}
        start = time.perf_counter()
        prompt = self._prompt_loader.render(
            "conversation.md",
            {
                "language": self._language,
                "conversation_history": conversation_history,
                "user_text": user_text,
            },
        )
        timings["prompt_render_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        raw_text, parsed, fallback_used = self._generate_with_repair(
            prompt=prompt.text,
            role="conversation",
            repair_schema='{"reply": "<assistant reply>"}',
        )
        timings["llm_generate_seconds"] = time.perf_counter() - start
        if parsed.model is None and raw_text:
            parsed = self._extract_last_valid_reply(raw_text) or parsed
        salvage_used = False
        if parsed.model is None and raw_text:
            salvaged = self._salvage_reply_from_text(raw_text)
            if salvaged is not None:
                parsed = salvaged
                salvage_used = True
        reply_text = ""
        if parsed.model is not None:
            reply_text = getattr(parsed.model, "reply", "")
        llm_meta: Dict[str, Any] = {
            "role": "conversation",
            "schema_valid": parsed.model is not None,
            "raw_output": raw_text,
        }
        if fallback_used:
            llm_meta["fallback_used"] = True
        if salvage_used:
            llm_meta["salvage_used"] = True
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

        start = time.perf_counter()
        self._db.run_write(_insert_assistant)
        timings["assistant_insert_seconds"] = time.perf_counter() - start
        return assistant_turn_id, reply_text

    def run_corrections(
        self,
        user_turn_id: str,
        user_text: str,
        assistant_turn_id: str | None = None,
        timings: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Generate and persist correction artifacts for a user turn.

        Parameters
        ----------
        user_turn_id : str
            User turn identifier.
        user_text : str
            Raw user text.
        assistant_turn_id : str | None
            Assistant turn to annotate once corrections persist.
        timings : dict[str, float] | None
            Optional timings accumulator.

        Returns
        -------
        dict[str, Any]
            Correction payload (errors, corrected, native, explanation).
        """
        if timings is None:
            timings = {}
        start_total = time.perf_counter()
        detect_prompt = self._prompt_loader.render(
            "detect_errors.md",
            {"language": self._language, "user_text": user_text},
        )
        start = time.perf_counter()
        detect_result = self._safe_generate_json(prompt=detect_prompt.text, role="error_detection")
        timings["corrections_detect_seconds"] = time.perf_counter() - start
        errors = []
        if detect_result.model is not None:
            errors = list(getattr(detect_result.model, "errors", []))

        correction_prompt = self._prompt_loader.render(
            "correct_sentence.md",
            {"language": self._language, "user_text": user_text},
        )
        start = time.perf_counter()
        correction_result = self._safe_generate_json(prompt=correction_prompt.text, role="correction")
        timings["corrections_correct_seconds"] = time.perf_counter() - start
        corrected_text = user_text
        if correction_result.model is not None:
            corrected_text = getattr(correction_result.model, "corrected", user_text)

        explain_prompt = self._prompt_loader.render(
            "explain.md",
            {"language": self._language, "user_text": user_text, "corrected_text": corrected_text},
        )
        start = time.perf_counter()
        explain_result = self._safe_generate_json(prompt=explain_prompt.text, role="explanation")
        timings["corrections_explain_seconds"] = time.perf_counter() - start
        explanation_text = ""
        if explain_result.model is not None:
            explanation_text = getattr(explain_result.model, "explanation", "")

        native_prompt = self._prompt_loader.render(
            "native_rewrite.md",
            {"language": self._language, "user_text": user_text},
        )
        start = time.perf_counter()
        native_result = self._safe_generate_json(prompt=native_prompt.text, role="native_reformulation")
        timings["corrections_native_seconds"] = time.perf_counter() - start
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

        start = time.perf_counter()
        self._db.run_write(_insert_correction)
        timings["corrections_insert_seconds"] = time.perf_counter() - start
        timings["corrections_total_seconds"] = time.perf_counter() - start_total
        if assistant_turn_id is not None:
            self._update_assistant_meta(
                assistant_turn_id,
                {"corrections_persisted": True},
            )
        return {
            "errors": errors,
            "corrected": corrected_text,
            "native": native_text or "",
            "explanation": explanation_text,
        }

    def run_tts(
        self,
        conversation_id: str,
        assistant_turn_id: str,
        reply_text: str,
        timings: Dict[str, float] | None = None,
    ) -> TTSResult | None:
        """Generate and cache TTS audio for an assistant reply.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        assistant_turn_id : str
            Assistant turn identifier.
        reply_text : str
            Reply text to synthesize.
        timings : dict[str, float] | None
            Optional timings accumulator.

        Returns
        -------
        TTSResult | None
            TTS result if synthesis succeeded, otherwise None.
        """
        if self._tts is None:
            return None

        try:
            start_total = time.perf_counter()
            start = time.perf_counter()
            normalised_text = self._normalise_for_tts(reply_text)
            if timings is not None:
                timings["tts_normalise_seconds"] = time.perf_counter() - start
            voice = self._tts_voice
            if voice == "default":
                voice = None

            start = time.perf_counter()
            result = self._tts.synthesize(
                conversation_id=conversation_id,
                turn_id=assistant_turn_id,
                text=normalised_text,
                voice=voice,
                speed=self._tts_speed,
                language=self._language,
            )
            if timings is not None:
                timings["tts_synthesize_seconds"] = time.perf_counter() - start
                timings["tts_total_seconds"] = time.perf_counter() - start_total
            return result
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
        result = self.run_tts(
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
            result = self.run_tts(
                conversation_id=conversation_id,
                assistant_turn_id=turn_id,
                reply_text=reply_text,
            )
            if result is None:
                raise RuntimeError("TTS regeneration failed.")
            results.append(result)
        return results

    def get_latest_corrections(self, user_turn_id: str) -> Dict[str, Any]:
        """Fetch corrections for a user turn."""
        def _fetch(conn) -> Dict[str, Any]:
            row = conn.execute(
                """
                SELECT errors_json, corrected_text, native_text, explanation_text
                FROM corrections
                WHERE user_turn_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_turn_id,),
            ).fetchone()
            if row is None:
                return {"errors": [], "corrected": "", "native": "", "explanation": ""}
            errors_json, corrected, native, explanation = row
            errors = []
            if errors_json:
                try:
                    errors = json.loads(errors_json)
                except Exception:
                    errors = []
            return {
                "errors": errors,
                "corrected": corrected or "",
                "native": native or "",
                "explanation": explanation or "",
            }

        with self._db.read_connection() as conn:
            return _fetch(conn)

    def reset_session(self) -> None:
        """Reset session-scoped state such as audio cache."""
        if self._audio_cache is not None:
            self._audio_cache.cleanup()
        self._asr_cache.clear()
        self._logger.info("session_reset")

    def _log_timings(self, label: str, timings: Dict[str, float]) -> None:
        if not self._timing_logs_enabled:
            return
        if not timings:
            return
        self._finalize_timings(timings)
        ordered = ", ".join(f"{key}={timings[key]:.4f}s" for key in sorted(timings.keys()))
        self._logger.info("timings.%s %s", label, ordered)

    @staticmethod
    def _finalize_timings(timings: Dict[str, float]) -> None:
        if "total_seconds" in timings:
            return
        total = sum(
            value
            for key, value in timings.items()
            if not key.endswith("_total_seconds") and key != "total_seconds"
        )
        timings["total_seconds"] = total

    def finalize_and_log_timings(self, label: str, timings: Dict[str, float]) -> None:
        """Finalize and log timings for async UI pipelines."""
        self._finalize_timings(timings)
        self._log_timings(label, timings)

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

    def _safe_generate_json(
        self,
        prompt: str,
        role: str,
        repair_schema: str | None = None,
    ) -> ParseResult:
        try:
            if repair_schema is None:
                return self._llm.generate_json(prompt=prompt, role=role)

            def _repair(raw_output: str) -> str:
                return self._repair_json_output(role=role, schema=repair_schema, raw_output=raw_output)

            return self._llm.generate_json(prompt=prompt, role=role, repair_fn=_repair)
        except Exception as exc:
            return ParseResult(model=None, raw_json=None, error=str(exc), repaired=False)

    def _generate_with_repair(
        self,
        prompt: str,
        role: str,
        repair_schema: str | None = None,
    ) -> tuple[str, ParseResult, bool]:
        try:
            llm_result = self._llm.generate(prompt=prompt, role=role)
            raw_text = llm_result.text
            if raw_text.strip() == "":
                llm_result = self._llm.generate(prompt=prompt, role=role)
                raw_text = llm_result.text
                if raw_text.strip() == "":
                    return raw_text, ParseResult(model=None, raw_json=None, error="empty_response", repaired=False), False
            repair_fn = None
            if repair_schema is not None:
                repair_fn = lambda raw: self._repair_json_output(role=role, schema=repair_schema, raw_output=raw)
            parsed = parse_with_schema(role=role, text=raw_text, repair_fn=repair_fn)
            trimmed_used = False
            if parsed.model is None and raw_text:
                idx = raw_text.find("{")
                if idx != -1:
                    trimmed = raw_text[idx:]
                    parsed_trimmed = parse_with_schema(role=role, text=trimmed)
                    if parsed_trimmed.model is not None:
                        parsed = parsed_trimmed
                        trimmed_used = True
            fallback_used = parsed.repaired or trimmed_used
            return raw_text, parsed, fallback_used
        except Exception as exc:
            return "", ParseResult(model=None, raw_json=None, error=str(exc), repaired=False), False

    def _extract_last_valid_reply(self, text: str) -> ParseResult | None:
        decoder = json.JSONDecoder()
        idx = 0
        best: ParseResult | None = None
        while idx < len(text):
            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(obj, dict) and "reply" in obj:
                candidate = parse_with_schema(role="conversation", text=json.dumps(obj, ensure_ascii=False))
                if candidate.model is not None:
                    reply = getattr(candidate.model, "reply", "")
                    if isinstance(reply, str) and reply.strip():
                        best = candidate
            idx = end
        return best

    def _salvage_reply_from_text(self, text: str) -> ParseResult | None:
        marker_idx = text.rfind('"reply"')
        if marker_idx == -1:
            return None
        tail = text[marker_idx + len('"reply"') :]
        match = re.search(r':\s*"', tail)
        if match is None:
            return None
        start = marker_idx + len('"reply"') + match.end()
        snippet = text[start:]
        for token in ("<|endoftext|>", "\nHuman:", "\n\nHuman:", "\nSystem:", "</think>"):
            pos = snippet.find(token)
            if pos != -1:
                snippet = snippet[:pos]
        end_quote = snippet.find('"')
        if end_quote != -1:
            reply = snippet[:end_quote]
        else:
            reply = snippet.strip()
        reply = reply.strip()
        if not reply:
            return None
        manual_json = json.dumps({"reply": reply}, ensure_ascii=False)
        parsed = parse_with_schema(role="conversation", text=manual_json)
        if parsed.model is not None:
            return parsed
        return None

    def _repair_json_output(self, role: str, schema: str, raw_output: str) -> str:
        repair_prompt = self._prompt_loader.render(
            "repair_json.md",
            {"json_schema": schema, "raw_output": raw_output},
        )
        result = self._llm.generate(prompt=repair_prompt.text, role=role)
        return result.text

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
