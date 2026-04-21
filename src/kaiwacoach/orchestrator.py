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
from typing import Any, Callable

from kaiwacoach.constants import SUPPORTED_LANGUAGES, VALID_PROFICIENCY_LEVELS
from kaiwacoach.models.asr_whisper import ASRResult
from kaiwacoach.models.json_enforcement import ParseResult, parse_with_schema
from kaiwacoach.models.protocols import ASRProtocol, LLMProtocol, TTSProtocol
from kaiwacoach.models.tts_kokoro import TTSResult
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache
from kaiwacoach.storage.db import SQLiteWriter
from kaiwacoach.textnorm.invariants import enforce_japanese_invariant
from kaiwacoach.textnorm.jp_katakana import normalise_katakana
from kaiwacoach.textnorm.tts_punctuation import normalize_for_tts
from kaiwacoach.utils import BoundedDict

_logger = logging.getLogger(__name__)

# Maximum number of unique (audio_hash, model_id, language) entries held in the
# in-process ASR cache. Oldest entries are evicted when the limit is reached.
_ASR_CACHE_MAX_SIZE = 128


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
    asr_meta: dict[str, Any]
    tts_audio_path: str | None = None


@dataclass(frozen=True)
class MonologueTurnResult:
    conversation_id: str
    user_turn_id: str
    input_text: str
    asr_text: str | None
    asr_meta: dict | None
    corrections: dict[str, Any]
    summary: dict[str, Any]


class ConversationOrchestrator:
    """Orchestrator for text/audio turns, corrections, and TTS with persistence."""

    def __init__(
        self,
        db: SQLiteWriter,
        llm: LLMProtocol,
        prompt_loader: PromptLoader,
        language: str,
        tts: TTSProtocol | None = None,
        tts_voice: str | None = None,
        tts_speed: float = 1.0,
        asr: ASRProtocol | None = None,
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
        self._asr_model_id = self._resolve_model_id(asr)
        self._llm_model_id = self._resolve_model_id(llm)
        self._tts_model_id = self._resolve_model_id(tts)
        self._expected_sample_rate = (
            getattr(audio_cache, "expected_sample_rate", None) if audio_cache is not None else None
        )
        self._asr_cache: BoundedDict = BoundedDict(maxsize=_ASR_CACHE_MAX_SIZE)
        self._logger = logging.getLogger(__name__)
        self._timing_logs_enabled = timing_logs_enabled

    @property
    def expected_sample_rate(self) -> int | None:
        """Expected sample rate for user audio input validation."""
        return self._expected_sample_rate

    def create_conversation(self, title: str | None = None) -> str:
        conversation_id = str(uuid.uuid4())

        def _insert(conn) -> None:
            conn.execute(
                """
                INSERT INTO conversations (
                    id, title, language, asr_model_id, llm_model_id, tts_model_id, model_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    self._language,
                    self._asr_model_id,
                    self._llm_model_id,
                    self._tts_model_id,
                    "{}",
                ),
            )
            conn.commit()

        self._db.run_write(_insert)
        return conversation_id

    def process_text_turn(
        self,
        conversation_id: str,
        user_text: str,
        conversation_history: str = "",
        corrections_enabled: bool = True,
        on_stage: Callable[[str, str, dict], None] | None = None,
    ) -> TextTurnResult:
        timings: dict[str, float] = {}
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
        self._maybe_set_auto_title(conversation_id, user_text)

        if on_stage:
            on_stage("llm", "running", {})
        assistant_turn_id, reply_text = self.generate_reply(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            user_text=user_text,
            conversation_history=conversation_history,
            timings=timings,
        )
        if on_stage:
            on_stage("llm", "complete", {"reply": reply_text})
        if on_stage:
            on_stage("tts", "running", {})
        tts_result = self.run_tts(
            conversation_id=conversation_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
            timings=timings,
        )
        if on_stage:
            on_stage("tts", "complete", {"audio_path": tts_result.audio_path if tts_result else None})
        if corrections_enabled:
            if on_stage:
                on_stage("corrections", "running", {})
            corrections = self.run_corrections(
                user_turn_id,
                user_text,
                assistant_turn_id=assistant_turn_id,
                timings=timings,
            )
            if on_stage:
                on_stage("corrections", "complete", {"data": corrections})
        self._finalize_timings(timings)
        self._log_timings("text_turn", timings)

        return TextTurnResult(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
            tts_audio_path=tts_result.audio_path if tts_result is not None else None,
        )

    def _run_asr_with_caching(
        self,
        pcm_bytes: bytes,
        audio_meta: AudioMeta,
        conversation_id: str,
        timings: dict[str, float],
    ) -> tuple[str, str, ASRResult, dict[str, Any]]:
        """Save audio, run ASR with in-process caching, persist user turn, return results.

        Returns
        -------
        tuple[str, str, ASRResult, dict[str, Any]]
            (user_turn_id, input_audio_path, asr_result, asr_meta)
        """
        if self._audio_cache is None or self._asr is None:
            raise ValueError("audio_cache and asr must be configured for audio turns")
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

        # ASR: persist a failure record if transcription throws, then re-raise.
        try:
            model_id, language = self._asr_cache_key()
            cache_key = (audio_hash, model_id, language)
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
        except Exception as exc:
            # Persist the failed user turn so the audio file is tracked in storage.
            failed_meta = {"audio_path": str(input_audio_path), "error": str(exc)}
            failed_meta_json = json.dumps(failed_meta, ensure_ascii=False)

            def _insert_user_failed(conn) -> None:
                conn.execute(
                    """
                    INSERT INTO user_turns (id, conversation_id, input_text, asr_text, asr_meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_turn_id, conversation_id, None, None, failed_meta_json),
                )
                conn.commit()

            self._db.run_write(_insert_user_failed)
            raise

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
        return user_turn_id, str(input_audio_path), asr_result, asr_meta

    def process_audio_turn(
        self,
        conversation_id: str,
        pcm_bytes: bytes,
        audio_meta: AudioMeta,
        conversation_history: str = "",
        corrections_enabled: bool = True,
        on_stage: Callable[[str, str, dict], None] | None = None,
    ) -> AudioTurnResult:
        if self._asr is None or self._audio_cache is None:
            raise ValueError("ASR and audio_cache must be configured for audio turns.")

        timings: dict[str, float] = {}

        # ASR: errors from later stages (LLM, corrections, TTS) propagate directly.
        if on_stage:
            on_stage("asr", "running", {})
        user_turn_id, input_audio_path, asr_result, asr_meta = self._run_asr_with_caching(
            pcm_bytes=pcm_bytes,
            audio_meta=audio_meta,
            conversation_id=conversation_id,
            timings=timings,
        )
        if on_stage:
            on_stage("asr", "complete", {"transcript": asr_result.text})
        self._maybe_set_auto_title(conversation_id, asr_result.text)

        if on_stage:
            on_stage("llm", "running", {})
        assistant_turn_id, reply_text = self.generate_reply(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            user_text=asr_result.text,
            conversation_history=conversation_history,
            timings=timings,
        )
        if on_stage:
            on_stage("llm", "complete", {"reply": reply_text})
        if on_stage:
            on_stage("tts", "running", {})
        tts_result = self.run_tts(
            conversation_id=conversation_id,
            assistant_turn_id=assistant_turn_id,
            reply_text=reply_text,
            timings=timings,
        )
        if on_stage:
            on_stage("tts", "complete", {"audio_path": tts_result.audio_path if tts_result else None})
        if corrections_enabled:
            if on_stage:
                on_stage("corrections", "running", {})
            corrections = self.run_corrections(
                user_turn_id,
                asr_result.text,
                assistant_turn_id=assistant_turn_id,
                timings=timings,
            )
            if on_stage:
                on_stage("corrections", "complete", {"data": corrections})
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

    def persist_user_text_turn(
        self,
        conversation_id: str,
        user_text: str,
        timings: dict[str, float] | None = None,
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
        timings: dict[str, float] | None = None,
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
        user_turn_id, input_audio_path, asr_result, asr_meta = self._run_asr_with_caching(
            pcm_bytes=pcm_bytes,
            audio_meta=audio_meta,
            conversation_id=conversation_id,
            timings=timings,
        )
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

    def persist_input_audio(
        self,
        conversation_id: str,
        turn_id: str,
        pcm_bytes: bytes,
        audio_meta: AudioMeta,
        kind_suffix: str = "",
    ) -> str:
        """Persist raw input audio alongside standard user audio.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        turn_id : str
            Turn identifier.
        pcm_bytes : bytes
            Raw PCM audio bytes.
        audio_meta : AudioMeta
            Audio metadata for the PCM bytes.
        kind_suffix : str
            Optional suffix for the audio kind (e.g., "raw"). When provided, the
            audio is stored under a `user_{suffix}` kind to distinguish raw
            input from the resampled user audio.

        Returns
        -------
        str
            Path to the stored audio file.
        """
        if self._audio_cache is None:
            raise ValueError("audio_cache must be configured to persist audio.")
        kind = "user"
        if kind_suffix:
            kind = f"{kind}_{kind_suffix}"
        path = self._audio_cache.save_audio(
            conversation_id=conversation_id,
            turn_id=turn_id,
            kind=kind,
            pcm_bytes=pcm_bytes,
            meta=audio_meta,
        )
        return str(path)

    def generate_reply(
        self,
        conversation_id: str,
        user_turn_id: str,
        user_text: str,
        conversation_history: str,
        timings: dict[str, float] | None = None,
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
        profile = self.get_user_profile()
        user_name = self._user_name_for_prompt(self._language, profile)
        user_level = self._user_level_for(self._language, profile)
        user_kanji_level = self._user_kanji_level_for(profile)
        profile_vars = {
            "user_name": user_name,
            "user_level": user_level,
            "user_kanji_level": user_kanji_level,
        }
        conversation_history = self._truncate_conversation_history(
            conversation_history=conversation_history,
            user_text=user_text,
            extra_render_vars=profile_vars,
        )
        start = time.perf_counter()
        prompt = self._prompt_loader.render(
            "conversation.md",
            {
                "language": self._language,
                "conversation_history": conversation_history,
                "user_text": user_text,
                **profile_vars,
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
        llm_meta: dict[str, Any] = {
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

    def _truncate_conversation_history(
        self,
        conversation_history: str,
        user_text: str,
        extra_render_vars: dict[str, Any] | None = None,
    ) -> str:
        token_counter = getattr(self._llm, "count_tokens", None)
        max_context_tokens = getattr(self._llm, "max_context_tokens", None)
        if not callable(token_counter) or not isinstance(max_context_tokens, int):
            return conversation_history

        history = conversation_history
        while True:
            render_vars: dict[str, Any] = {
                "language": self._language,
                "conversation_history": history,
                "user_text": user_text,
            }
            if extra_render_vars:
                render_vars.update(extra_render_vars)
            prompt = self._prompt_loader.render(
                "conversation.md",
                render_vars,
            )
            tokens = token_counter(prompt.text)
            if tokens is None or tokens <= max_context_tokens:
                return history
            truncated = self._drop_oldest_turn(history)
            if not truncated:
                return ""
            if truncated == history:
                return history
            history = truncated

    @staticmethod
    def _drop_oldest_turn(history: str) -> str:
        lines = [line for line in history.splitlines() if line.strip()]
        if not lines:
            return ""
        if lines[0].startswith("User:"):
            if len(lines) > 1 and lines[1].startswith("Assistant:"):
                lines = lines[2:]
            else:
                lines = lines[1:]
        else:
            lines = lines[1:]
        return "\n".join(lines)

    def run_corrections(
        self,
        user_turn_id: str,
        user_text: str,
        assistant_turn_id: str | None = None,
        timings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
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
        profile = self.get_user_profile()
        user_level = self._user_level_for(self._language, profile)
        user_kanji_level = self._user_kanji_level_for(profile)

        # Call 1: detect errors and produce corrected sentence in a single LLM call.
        # No repair path: the errors field is a list, which makes a generic repair
        # schema hard to specify safely. On failure both fields fall back to defaults
        # (empty errors list, original user_text as corrected).
        detect_correct_prompt = self._prompt_loader.render(
            "detect_and_correct.md",
            {
                "language": self._language,
                "user_text": user_text,
                "user_level": user_level,
                "user_kanji_level": user_kanji_level,
            },
        )
        t0 = time.perf_counter()
        detect_correct_result = self._safe_generate_json(
            prompt=detect_correct_prompt.text, role="detect_and_correct"
        )
        timings["corrections_detect_correct_seconds"] = time.perf_counter() - t0
        errors = []
        corrected_text = user_text
        if detect_correct_result.model is not None:
            errors = list(getattr(detect_correct_result.model, "errors", []))
            corrected_text = getattr(detect_correct_result.model, "corrected", user_text)

        # Call 2: explanation (English) and native rewrite in a single LLM call.
        # Tradeoff: combining these means a single JSON parse failure loses both
        # fields. Previously a failed native rewrite still produced an explanation.
        # Repair is available here to mitigate this; the repair schema covers both fields.
        # Escape braces so format_map doesn't treat LLM-generated error strings as placeholders.
        errors_text = "\n".join(f"- {e}" for e in errors) if errors else "(none)"
        errors_text = errors_text.replace("{", "{{").replace("}", "}}")
        explain_native_prompt = self._prompt_loader.render(
            "explain_and_native.md",
            {
                "language": self._language,
                "user_text": user_text,
                "corrected_text": corrected_text,
                "errors": errors_text,
                "user_level": user_level,
            },
        )
        t0 = time.perf_counter()
        _, explain_native_result, explain_native_fallback_used = self._generate_with_repair(
            prompt=explain_native_prompt.text,
            role="explain_and_native",
            repair_schema='{"explanation": "<English explanation>", "native": "<natural rewrite>"}',
        )
        timings["corrections_explain_native_seconds"] = time.perf_counter() - t0
        explanation_text = ""
        native_text = None
        if explain_native_result.model is not None:
            explanation_text = getattr(explain_native_result.model, "explanation", "")
            native_text = getattr(explain_native_result.model, "native", None)
        if explain_native_result.model is None and explain_native_result.error:
            _logger.warning(
                "corrections.explain_native_invalid language=%s error=%s",
                self._language,
                explain_native_result.error,
            )
        if explain_native_fallback_used:
            _logger.info("corrections.explain_native_fallback_used language=%s", self._language)

        correction_id = str(uuid.uuid4())
        errors_json = json.dumps(errors, ensure_ascii=False)
        prompt_hash = detect_correct_prompt.sha256

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

        t0 = time.perf_counter()
        self._db.run_write(_insert_correction)
        timings["corrections_insert_seconds"] = time.perf_counter() - t0
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
        timings: dict[str, float] | None = None,
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

        normalised_text = reply_text
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
        except Exception:  # noqa: BLE001 — TTS backend raises are unspecified; log and degrade gracefully
            _logger.exception(
                "TTS synthesis failed for turn_id=%s text_len=%d language=%s",
                assistant_turn_id,
                len(normalised_text),
                self._language,
            )
            return None

    def generate_narration(self, text: str) -> str:
        """Synthesise text to audio using self._language; return the raw audio path.

        The route handler converts the path to a URL via audio_path_to_url().
        """
        if self._tts is None:
            raise ValueError("TTS must be configured to generate narration.")

        normalised_text = self._normalise_for_tts(text)
        voice = self._tts_voice
        if voice == "default":
            voice = None

        result = self._tts.synthesize(
            conversation_id="narrations",
            turn_id=str(uuid.uuid4()),
            text=normalised_text,
            voice=voice,
            speed=self._tts_speed,
            language=self._language,
        )
        return result.audio_path

    def create_monologue_conversation(self) -> str:
        """Create a conversation with conversation_type='monologue'.

        Title starts as NULL and is set from the first user input by
        process_monologue_turn, matching chat conversation behaviour.

        Returns
        -------
        str
            The new conversation_id.
        """
        conversation_id = str(uuid.uuid4())

        def _insert(conn) -> None:
            conn.execute(
                """
                INSERT INTO conversations (
                    id, title, language, asr_model_id, llm_model_id, tts_model_id,
                    model_metadata_json, conversation_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    None,
                    self._language,
                    self._asr_model_id,
                    self._llm_model_id,
                    self._tts_model_id,
                    "{}",
                    "monologue",
                ),
            )
            conn.commit()

        self._db.run_write(_insert)
        return conversation_id

    def process_monologue_turn(
        self,
        conversation_id: str,
        text: str | None = None,
        pcm_bytes: bytes | None = None,
        audio_meta: AudioMeta | None = None,
        on_stage: Callable[[str, str, dict], None] | None = None,
    ) -> MonologueTurnResult:
        """Run the monologue pipeline: ASR (if audio) → corrections → summary.

        No assistant_turn row is created. Audio preparation (WebM→PCM conversion,
        AudioMeta construction) must be done by the route handler before calling
        this method.

        Parameters
        ----------
        conversation_id : str
            A monologue conversation created by create_monologue_conversation().
        text : str | None
            Pre-typed text input (mutually exclusive with pcm_bytes).
        pcm_bytes : bytes | None
            PCM audio bytes (mutually exclusive with text).
        audio_meta : AudioMeta | None
            Required when pcm_bytes is provided.
        on_stage : Callable | None
            Callback for stage progress events: on_stage(stage, status, data).

        Returns
        -------
        MonologueTurnResult
        """
        if text is None and pcm_bytes is None:
            raise ValueError("Either text or pcm_bytes must be provided.")
        if pcm_bytes is not None and audio_meta is None:
            raise ValueError("audio_meta is required when pcm_bytes is provided.")

        timings: dict[str, float] = {}
        asr_text: str | None = None
        asr_meta_out: dict | None = None

        # ASR stage (audio input only)
        if pcm_bytes is not None:
            if self._asr is None or self._audio_cache is None:
                raise ValueError("ASR and audio_cache must be configured for audio monologue.")
            if on_stage:
                on_stage("asr", "running", {})
            user_turn_id, _input_audio_path, asr_result, asr_meta_out = self._run_asr_with_caching(
                pcm_bytes=pcm_bytes,
                audio_meta=audio_meta,  # type: ignore[arg-type]
                conversation_id=conversation_id,
                timings=timings,
            )
            asr_text = asr_result.text
            input_text = asr_result.text
            if on_stage:
                on_stage("asr", "complete", {"transcript": asr_text})
        else:
            # Text input: persist user turn directly
            input_text = text  # type: ignore[assignment]
            user_turn_id = str(uuid.uuid4())

            def _insert_user(conn) -> None:
                conn.execute(
                    "INSERT INTO user_turns (id, conversation_id, input_text) VALUES (?, ?, ?)",
                    (user_turn_id, conversation_id, input_text),
                )
                conn.commit()

            t0 = time.perf_counter()
            self._db.run_write(_insert_user)
            timings["user_insert_seconds"] = time.perf_counter() - t0

        # Set title from first user input, matching chat conversation behaviour.
        self._maybe_set_auto_title(conversation_id, input_text)

        # Corrections stage
        if on_stage:
            on_stage("corrections", "running", {})
        corrections = self.run_corrections(
            user_turn_id=user_turn_id,
            user_text=input_text,
            assistant_turn_id=None,
            timings=timings,
        )
        if on_stage:
            on_stage("corrections", "complete", {
                "errors": corrections["errors"],
                "corrected": corrections["corrected"],
                "native": corrections["native"],
                "explanation": corrections["explanation"],
            })

        # Summary stage
        if on_stage:
            on_stage("summary", "running", {})
        summary = self._run_monologue_summary(
            corrections=corrections,
            timings=timings,
        )
        if on_stage:
            on_stage("summary", "complete", {
                "improvement_areas": summary["improvement_areas"],
                "overall_assessment": summary["overall_assessment"],
            })

        self._log_timings("monologue_turn", timings)

        return MonologueTurnResult(
            conversation_id=conversation_id,
            user_turn_id=user_turn_id,
            input_text=input_text,
            asr_text=asr_text,
            asr_meta=asr_meta_out,
            corrections=corrections,
            summary=summary,
        )

    def _run_monologue_summary(
        self,
        corrections: dict[str, Any],
        timings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Call the monologue_summary LLM role and return the result dict."""
        if timings is None:
            timings = {}
        profile = self.get_user_profile()
        user_level = self._user_level_for(self._language, profile)

        errors_list: list[str] = corrections.get("errors") or []
        errors_str = "; ".join(errors_list) if errors_list else "No errors detected"
        # Escape braces so format_map doesn't treat error strings as placeholders.
        errors_str = errors_str.replace("{", "{{").replace("}", "}}")
        corrected = (corrections.get("corrected") or "").replace("{", "{{").replace("}", "}}")
        explanation = (corrections.get("explanation") or "").replace("{", "{{").replace("}", "}}")

        prompt = self._prompt_loader.render(
            "monologue_summary.md",
            {
                "language": self._language,
                "errors": errors_str,
                "corrected": corrected,
                "explanation": explanation,
                "user_level": user_level,
            },
        )
        t0 = time.perf_counter()
        result = self._safe_generate_json(prompt=prompt.text, role="monologue_summary")
        if timings is not None:
            timings["monologue_summary_seconds"] = time.perf_counter() - t0

        if result.model is not None:
            return {
                "improvement_areas": list(getattr(result.model, "improvement_areas", [])),
                "overall_assessment": getattr(result.model, "overall_assessment", ""),
            }
        _logger.warning(
            "monologue_summary.invalid language=%s error=%s",
            self._language,
            result.error,
        )
        return {"improvement_areas": [], "overall_assessment": ""}

    def get_corrections_for_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return up to 20 most recent corrections for a conversation, oldest-first.

        Returns a list of dicts with keys: errors_json, corrected_text, created_at.
        """
        with self._db.read_connection() as conn:
            raw = conn.execute(
                """
                SELECT c.errors_json, c.corrected_text, c.created_at
                FROM corrections c
                JOIN user_turns ut ON c.user_turn_id = ut.id
                WHERE ut.conversation_id = ?
                ORDER BY c.created_at DESC LIMIT 20
                """,
                (conversation_id,),
            ).fetchall()
        rows = [
            {"errors_json": r[0], "corrected_text": r[1], "created_at": r[2]}
            for r in raw
        ]
        rows.reverse()
        return rows

    def summarise_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Read up to 20 most recent corrections and call the summarise_conversation LLM role.

        Returns a dict with top_error_patterns, priority_areas, overall_notes.
        When no corrections exist, returns an informational dict without calling the LLM.
        """
        rows = self.get_corrections_for_conversation(conversation_id)

        # Build corrections_text; skip rows with no errors.
        lines: list[str] = []
        n = 0
        for row in rows:
            try:
                errors: list[str] = json.loads(row["errors_json"]) if row["errors_json"] else []
            except (json.JSONDecodeError, TypeError):
                errors = []
            if not errors:
                continue
            n += 1
            errors_str = "; ".join(errors)
            corrected = row.get("corrected_text") or ""
            lines.append(f"[{n}] Errors: {errors_str}  |  Corrected: {corrected}")

        if not lines:
            with self._db.read_connection() as conn:
                turn_count = conn.execute(
                    "SELECT COUNT(*) FROM user_turns WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            if turn_count == 0:
                return {
                    "top_error_patterns": [],
                    "priority_areas": [],
                    "overall_notes": "No conversation is available to summarise.",
                }
            return {
                "top_error_patterns": [],
                "priority_areas": [],
                "overall_notes": "No corrections were recorded for this conversation.",
            }

        corrections_text = "\n".join(lines)
        # Escape braces so format_map doesn't treat correction content as placeholders.
        corrections_text = corrections_text.replace("{", "{{").replace("}", "}}")

        user_level = self._user_level_for(self._language)

        prompt = self._prompt_loader.render(
            "summarise_conversation.md",
            {
                "language": self._language,
                "corrections_text": corrections_text,
                "user_level": user_level,
            },
        )

        result = self._safe_generate_json(prompt=prompt.text, role="summarise_conversation")

        if result.model is not None:
            return {
                "top_error_patterns": list(result.model.top_error_patterns),
                "priority_areas": list(result.model.priority_areas),
                "overall_notes": result.model.overall_notes,
            }

        _logger.warning(
            "summarise_conversation.invalid language=%s error=%s",
            self._language,
            result.error,
        )
        return {
            "top_error_patterns": [],
            "priority_areas": [],
            "overall_notes": "",
        }

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

    def regenerate_conversation_audio(
        self,
        conversation_id: str,
        on_turn: Callable[[str, str | None], None] | None = None,
    ) -> list[TTSResult]:
        """Regenerate TTS audio for every assistant turn in a conversation.

        Parameters
        ----------
        conversation_id:
            Conversation identifier.
        on_turn:
            Optional callback invoked after each turn is synthesised.
            Receives ``(assistant_turn_id, audio_path_or_none)``.
        """
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
            if on_turn:
                on_turn(turn_id, result.audio_path)
            results.append(result)
        return results

    def get_latest_corrections(self, user_turn_id: str) -> dict[str, Any]:
        """Fetch corrections for a user turn."""
        def _fetch(conn) -> dict[str, Any]:
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
                except json.JSONDecodeError:
                    _logger.warning(
                        "Failed to parse corrections JSON for turn_id=%s",
                        user_turn_id,
                    )
                    errors = []
            return {
                "errors": errors,
                "corrected": corrected or "",
                "native": native or "",
                "explanation": explanation or "",
            }

        with self._db.read_connection() as conn:
            return _fetch(conn)

    def get_user_profile(self) -> dict[str, Any]:
        """Return user profile including name variants and language proficiency."""
        with self._db.read_connection() as conn:
            row = conn.execute(
                "SELECT user_name, user_name_romanised, user_name_katakana, language_proficiency_json"
                " FROM user_profile WHERE id = 1"
            ).fetchone()
        if row is None:
            return {
                "user_name": None,
                "user_name_romanised": None,
                "user_name_katakana": None,
                "language_proficiency": {},
            }
        try:
            proficiency = json.loads(row[3]) if row[3] else {}
        except (json.JSONDecodeError, TypeError):
            proficiency = {}
        return {
            "user_name": row[0],
            "user_name_romanised": row[1],
            "user_name_katakana": row[2],
            "language_proficiency": proficiency,
        }

    def set_user_profile(self, user_name: str | None, language_proficiency: dict[str, str]) -> None:
        """Persist user name, derived name forms, and per-language proficiency levels."""
        romanised: str | None = None
        katakana: str | None = None
        if user_name:
            prompt = self._prompt_loader.render("normalise_name.md", {"name": user_name})
            result = self._safe_generate_json(prompt=prompt.text, role="normalise_name")
            if result.model is not None:
                romanised = getattr(result.model, "romanised", None) or None
                katakana = getattr(result.model, "katakana", None) or None
        self._db.execute_update(
            "user_profile",
            {
                "user_name": user_name,
                "user_name_romanised": romanised,
                "user_name_katakana": katakana,
                "language_proficiency_json": json.dumps(language_proficiency, ensure_ascii=False),
            },
            {"id": 1},
        )

    def _user_level_for(self, language: str, profile: dict | None = None) -> str:
        """Return the stored grammar/overall level for the given language.

        Defaults: 'N5' for 'ja', 'A1' for all other supported languages.
        Pass a pre-fetched profile dict to avoid an extra DB read.
        """
        if profile is None:
            profile = self.get_user_profile()
        stored = profile["language_proficiency"].get(language)
        if stored:
            return stored
        return "N5" if language == "ja" else "A1"

    def _user_kanji_level_for(self, profile: dict | None = None) -> str:
        """Return the stored kanji reading level (key 'ja_kanji').

        Falls back to the overall 'ja' level if 'ja_kanji' is not set.
        Returns '' for non-Japanese sessions so callers can pass it as an empty
        string to PromptLoader.render() without rendering the literal 'None'.
        Pass a pre-fetched profile dict to avoid an extra DB read.
        """
        if self._language != "ja":
            return ""
        if profile is None:
            profile = self.get_user_profile()
        proficiency = profile["language_proficiency"]
        stored = proficiency.get("ja_kanji")
        if stored:
            return stored
        return proficiency.get("ja") or "N5"

    def _user_name_for_prompt(self, language: str, profile: dict | None = None) -> str:
        """Return the language-appropriate name form, or '' if not set.

        Uses the stored katakana form for Japanese sessions and the romanised form
        for all others, falling back to the raw user_name if the derived form is absent.
        Callers must not pass None to PromptLoader.render() — it renders as 'None'.
        Pass a pre-fetched profile dict to avoid an extra DB read.
        """
        if profile is None:
            profile = self.get_user_profile()
        if language == "ja":
            return profile.get("user_name_katakana") or profile.get("user_name") or ""
        return profile.get("user_name_romanised") or profile.get("user_name") or ""

    def reset_session(self) -> None:
        """Reset session-scoped state such as audio and LLM caches."""
        if self._audio_cache is not None:
            self._audio_cache.cleanup()
        self._asr_cache.clear()
        if hasattr(self._llm, "clear_cache"):
            self._llm.clear_cache()
        self._logger.info("session_reset")

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        """Update the session language and keep ASR in sync.

        Parameters
        ----------
        language : str
            New language code for the session (e.g., "ja", "fr", "en").
        """
        if language == self._language:
            return
        self._language = language
        if self._asr is not None:
            setter = getattr(self._asr, "set_language", None)
            if callable(setter):
                setter(language)
            elif hasattr(self._asr, "_language"):
                setattr(self._asr, "_language", language)

    def update_conversation_language(self, conversation_id: str, language: str) -> None:
        """Persist the language for an existing conversation.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier.
        language : str
            Language code to store for the conversation.
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}.")

        def _update(conn) -> None:
            conn.execute(
                "UPDATE conversations SET language = ? WHERE id = ?",
                (language, conversation_id),
            )

        self._db.run_write(_update)

    def list_conversations(self, conversation_type: str | None = None) -> list[dict[str, Any]]:
        """Return a summary list of conversations for UI selection.

        Parameters
        ----------
        conversation_type : str | None
            If set, filter to conversations with this type ('chat' or 'monologue').
            If None, return all conversations.
        """
        base_query = """
            SELECT
                id,
                title,
                language,
                updated_at,
                (
                    SELECT a.reply_text
                    FROM assistant_turns a
                    WHERE a.conversation_id = conversations.id
                    ORDER BY datetime(a.created_at) DESC
                    LIMIT 1
                ) AS preview_text,
                conversation_type
            FROM conversations
        """
        where = "WHERE conversation_type = ?" if conversation_type is not None else ""
        params: tuple = (conversation_type,) if conversation_type is not None else ()
        with self._db.read_connection() as conn:
            rows = conn.execute(
                f"{base_query} {where} ORDER BY datetime(updated_at) DESC",
                params,
            ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "language": row[2],
                "updated_at": row[3],
                "preview_text": row[4],
                "conversation_type": row[5],
            }
            for row in rows
        ]

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Fetch a conversation and its turns for replay."""
        with self._db.read_connection() as conn:
            convo = conn.execute(
                """
                SELECT id, title, language, created_at, updated_at, conversation_type
                FROM conversations WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if convo is None:
                raise ValueError(f"Unknown conversation_id: {conversation_id}")
            turns = conn.execute(
                """
                SELECT
                    u.id AS user_turn_id,
                    u.input_text,
                    u.asr_text,
                    a.id AS assistant_turn_id,
                    a.reply_text
                FROM user_turns u
                LEFT JOIN assistant_turns a
                    ON a.user_turn_id = u.id
                WHERE u.conversation_id = ?
                ORDER BY datetime(u.created_at) ASC
                """,
                (conversation_id,),
            ).fetchall()
        return {
            "id": convo[0],
            "title": convo[1],
            "language": convo[2],
            "created_at": convo[3],
            "updated_at": convo[4],
            "conversation_type": convo[5],
            "turns": [
                {
                    "user_turn_id": row[0],
                    "input_text": row[1],
                    "asr_text": row[2],
                    "assistant_turn_id": row[3],
                    "reply_text": row[4],
                }
                for row in turns
            ],
        }

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and cascade dependent rows."""
        def _delete(conn) -> None:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

        self._db.run_write(_delete)

    def delete_all_conversations(self) -> None:
        """Delete all conversations and cascade dependent rows."""
        def _delete(conn) -> None:
            conn.execute("DELETE FROM conversations")
            conn.commit()

        self._db.run_write(_delete)

    def _asr_cache_key(self) -> tuple[str, str]:
        """Return (model_id, language) for ASR cache key construction.

        Uses getattr to probe public then private attributes because ASRProtocol
        does not currently expose these fields. Consolidates all access in one place.
        """
        model_id = getattr(self._asr, "model_id", None) or getattr(self._asr, "_model_id", "unknown")
        language = getattr(self._asr, "language", None) or getattr(self._asr, "_language", self._language)
        return str(model_id), str(language)

    @staticmethod
    def _resolve_model_id(model: Any | None) -> str:
        if model is None:
            return "unknown"
        return getattr(model, "model_id", None) or getattr(model, "_model_id", None) or "unknown"

    def _log_timings(self, label: str, timings: dict[str, float]) -> None:
        if not self._timing_logs_enabled:
            return
        if not timings:
            return
        self._finalize_timings(timings)
        ordered = ", ".join(f"{key}={timings[key]:.4f}s" for key in sorted(timings.keys()))
        self._logger.info("timings.%s %s", label, ordered)

    @staticmethod
    def _finalize_timings(timings: dict[str, float]) -> None:
        if "total_seconds" in timings:
            return
        total = sum(
            value
            for key, value in timings.items()
            if not key.endswith("_total_seconds") and key != "total_seconds"
        )
        timings["total_seconds"] = total

    def finalize_and_log_timings(self, label: str, timings: dict[str, float]) -> None:
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

    def _maybe_set_auto_title(self, conversation_id: str, user_text: str) -> None:
        """Set the conversation title from the first user message if no title exists yet."""
        with self._db.read_connection() as conn:
            row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None or row[0]:
                return  # already titled or conversation missing
            count = conn.execute(
                "SELECT COUNT(*) FROM user_turns WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
        if count != 1:
            return  # not the first turn
        title = user_text.strip()
        if len(title) > 50:
            title = title[:49] + "…"
        self._db.execute_update("conversations", {"title": title}, {"id": conversation_id})

    def _update_assistant_meta(self, assistant_turn_id: str, updates: dict[str, Any]) -> None:
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
