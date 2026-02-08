"""Tests for conversation orchestrator text flows."""

from __future__ import annotations

import json
from pathlib import Path

from kaiwacoach.models.asr_whisper import ASRResult, WhisperASR
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.tts_kokoro import TTSResult
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.blobs import AudioMeta, SessionAudioCache
from kaiwacoach.storage.db import SQLiteWriter


class _Backend:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text

    def generate(self, prompt: str, max_tokens: int) -> str:
        if "Error Detection" in prompt:
            return "{\"errors\": [\"err1\"]}"
        if "Corrected Sentence" in prompt:
            return "{\"corrected\": \"こんにちは。\"}"
        if "Native Reformulation" in prompt:
            return "{\"native\": \"こんにちは。\"}"
        if "Explanation" in prompt:
            return "{\"explanation\": \"Fixed punctuation.\"}"
        return self._reply_text


class _FakeTTS:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize(
        self,
        conversation_id: str,
        turn_id: str,
        text: str,
        voice: str | None,
        speed: float,
        lang_code: str | None = None,
        language: str | None = None,
    ) -> TTSResult:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang_code": lang_code,
                "language": language,
            }
        )
        return TTSResult(audio_path="/tmp/fake.wav", meta={"backend": "fake"})


class _FakeASR(WhisperASR):
    def __init__(self, text: str, meta: dict[str, object]) -> None:
        self._text = text
        self._meta = meta

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        return ASRResult(text=self._text, meta=dict(self._meta))


class _FailingASR:
    def transcribe(self, audio_path: str | Path) -> ASRResult:
        raise RuntimeError("ASR failed")


def _setup_db(tmp_path: Path) -> SQLiteWriter:
    db_path = tmp_path / "db.sqlite"
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    return db


def test_persists_turns_and_reply(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        with db.read_connection() as conn:
            user_rows = conn.execute("SELECT input_text FROM user_turns").fetchall()
            assistant_rows = conn.execute("SELECT reply_text, llm_meta_json FROM assistant_turns").fetchall()
            correction_rows = conn.execute(
                "SELECT errors_json, corrected_text, native_text, explanation_text, prompt_hash FROM corrections"
            ).fetchall()

            assert user_rows == [("こんにちは",)]
            assert assistant_rows[0][0] == "hello"
            # Ensure metadata is valid JSON and contains prompt_hash
            meta = json.loads(assistant_rows[0][1])
            assert "prompt_hash" in meta
            assert correction_rows
            assert json.loads(correction_rows[0][0]) == ["err1"]
            assert correction_rows[0][1] == "こんにちは。"
            assert correction_rows[0][2] == "こんにちは。"
            assert correction_rows[0][3] == "Fixed punctuation."
            assert correction_rows[0][4]
            assert result.reply_text == "hello"
    finally:
        db.close()


def test_create_conversation_persists_language(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="fr")

        conversation_id = orch.create_conversation("Bonjour")

        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT title, language FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

        assert row == ("Bonjour", "fr")
    finally:
        db.close()


def test_corrections_empty_errors(tmp_path: Path) -> None:
    class _NoErrorBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Error Detection" in prompt:
                return "{\"errors\": []}"
            if "Corrected Sentence" in prompt:
                return "{\"corrected\": \"こんにちは\"}"
            if "Explanation" in prompt:
                return "{\"explanation\": \"\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_NoErrorBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        with db.read_connection() as conn:
            row = conn.execute("SELECT errors_json FROM corrections").fetchone()

        assert json.loads(row[0]) == []
    finally:
        db.close()


def test_fallback_when_llm_returns_invalid_json(tmp_path: Path) -> None:
    class _BadJsonBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            return "not json"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_BadJsonBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert result.reply_text == ""
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT llm_meta_json FROM assistant_turns WHERE id = ?",
                (result.assistant_turn_id,),
            ).fetchone()

        meta = json.loads(row[0])
        assert meta.get("raw_output") == "not json"
        assert "error" in meta
    finally:
        db.close()


def test_repair_path_marks_fallback_used(tmp_path: Path) -> None:
    class _RepairBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "JSON Repair" in prompt:
                return "{\"reply\": \"fixed\"}"
            return "not json"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_RepairBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert result.reply_text == "fixed"
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT llm_meta_json FROM assistant_turns WHERE id = ?",
                (result.assistant_turn_id,),
            ).fetchone()

        meta = json.loads(row[0])
        assert meta.get("fallback_used") is True
        assert meta.get("repaired") is True
        assert meta.get("raw_output") == "not json"
    finally:
        db.close()


def test_salvage_path_extracts_reply(tmp_path: Path) -> None:
    class _SalvageBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            return 'thinking... "reply": "こんにちは" trailing'

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_SalvageBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert result.reply_text == "こんにちは"
        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT llm_meta_json FROM assistant_turns WHERE id = ?",
                (result.assistant_turn_id,),
            ).fetchone()

        meta = json.loads(row[0])
        assert meta.get("salvage_used") is True
        assert meta.get("raw_output") == 'thinking... "reply": "こんにちは" trailing'
    finally:
        db.close()


def test_native_reformulation_invalid_falls_back_to_none(tmp_path: Path) -> None:
    class _NativeInvalidBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Error Detection" in prompt:
                return "{\"errors\": [\"err1\"]}"
            if "Corrected Sentence" in prompt:
                return "{\"corrected\": \"こんにちは。\"}"
            if "Native Reformulation" in prompt:
                return "not json"
            if "Explanation" in prompt:
                return "{\"explanation\": \"Fixed punctuation.\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_NativeInvalidBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT corrected_text, native_text, explanation_text FROM corrections"
            ).fetchone()

        assert row == ("こんにちは。", None, "Fixed punctuation.")
    finally:
        db.close()


def test_native_reformulation_empty_string_persists(tmp_path: Path) -> None:
    class _NativeEmptyBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Error Detection" in prompt:
                return "{\"errors\": []}"
            if "Corrected Sentence" in prompt:
                return "{\"corrected\": \"こんにちは\"}"
            if "Native Reformulation" in prompt:
                return "{\"native\": \"\"}"
            if "Explanation" in prompt:
                return "{\"explanation\": \"\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_NativeEmptyBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        with db.read_connection() as conn:
            row = conn.execute("SELECT native_text FROM corrections").fetchone()

        assert row == ("",)
    finally:
        db.close()


def test_correction_prompt_hash_matches_template(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        expected_prompt = prompts.render(
            "correct_sentence.md",
            {"language": "ja", "user_text": "こんにちは"},
        )

        with db.read_connection() as conn:
            row = conn.execute("SELECT prompt_hash FROM corrections").fetchone()

        assert row == (expected_prompt.sha256,)
    finally:
        db.close()


def test_normalises_and_synthesises_tts(tmp_path: Path) -> None:
    class _TtsBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Conversation Reply" in prompt:
                return "{\"reply\": \"すごい！！！本当？？\"}"
            if "Error Detection" in prompt:
                return "{\"errors\": []}"
            if "Corrected Sentence" in prompt:
                return "{\"corrected\": \"すごい！！！本当？？\"}"
            if "Native Reformulation" in prompt:
                return "{\"native\": \"すごい！！！本当？？\"}"
            if "Explanation" in prompt:
                return "{\"explanation\": \"\"}"
            if "Japanese TTS Normalisation" in prompt:
                return "{\"text\": \"すごい！！！本当？？\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_TtsBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        tts = _FakeTTS()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            tts=tts,
            tts_voice="default",
            tts_speed=1.0,
        )

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert result.tts_audio_path == "/tmp/fake.wav"
        assert tts.calls
        assert tts.calls[0]["text"] == "すごい！ 本当？"
        assert tts.calls[0]["language"] == "ja"
        assert tts.calls[0]["voice"] is None
    finally:
        db.close()


def test_skips_tts_when_not_configured(tmp_path: Path) -> None:
    class _ReplyOnlyBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_ReplyOnlyBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert result.tts_audio_path is None
    finally:
        db.close()


def test_non_japanese_language_bypasses_normalisation(tmp_path: Path) -> None:
    class _FrenchBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Conversation Reply" in prompt:
                return "{\"reply\": \"Bonjour...\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
            },
            backend=_FrenchBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        tts = _FakeTTS()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="fr",
            tts=tts,
            tts_voice="default",
            tts_speed=1.0,
        )

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "Salut", conversation_history="")

        assert tts.calls
        assert tts.calls[0]["text"] == "Bonjour..."
    finally:
        db.close()


def test_invariant_violation_falls_back(tmp_path: Path) -> None:
    class _InvariantBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Conversation Reply" in prompt:
                return "{\"reply\": \"こんにちは\"}"
            if "Japanese TTS Normalisation" in prompt:
                return "{\"text\": \"コンニチハ\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_InvariantBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        tts = _FakeTTS()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            tts=tts,
            tts_voice="default",
            tts_speed=1.0,
        )

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert tts.calls
        assert tts.calls[0]["text"] == "こんにちは"
    finally:
        db.close()


def test_tts_uses_explicit_voice(tmp_path: Path) -> None:
    class _ReplyBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            if "Conversation Reply" in prompt:
                return "{\"reply\": \"hello\"}"
            return "{\"reply\": \"ok\"}"

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_ReplyBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        tts = _FakeTTS()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            tts=tts,
            tts_voice="jf_alpha",
            tts_speed=1.0,
        )

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert tts.calls
        assert tts.calls[0]["voice"] == "jf_alpha"
    finally:
        db.close()


def test_llm_failure_falls_back_and_persists_error(tmp_path: Path) -> None:
    class _FailingBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            raise RuntimeError("LLM failure")

    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_FailingBackend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        assert result.reply_text == ""
        assert result.tts_audio_path is None

        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT reply_text, llm_meta_json FROM assistant_turns WHERE id = ?",
                (result.assistant_turn_id,),
            ).fetchone()

        assert row[0] == ""
        meta = json.loads(row[1])
        assert "error" in meta
        assert meta.get("corrections_persisted") is True
    finally:
        db.close()


def test_process_audio_turn_persists_asr_text_and_meta(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    cache = SessionAudioCache(root_dir=tmp_path / "audio")
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        asr = _FakeASR(text="konnichiwa", meta={"backend": "fake"})
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            asr=asr,
            audio_cache=cache,
        )

        audio_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)
        pcm_bytes = b"\x00\x00" * 200
        conversation_id = orch.create_conversation("Test")
        result = orch.process_audio_turn(conversation_id, pcm_bytes, audio_meta, conversation_history="")

        assert result.asr_text == "konnichiwa"
        assert Path(result.input_audio_path).exists()

        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT input_text, asr_text, asr_meta_json FROM user_turns WHERE id = ?",
                (result.user_turn_id,),
            ).fetchone()

        assert row[0] is None
        assert row[1] == "konnichiwa"
        meta = json.loads(row[2])
        assert meta["backend"] == "fake"
        assert meta["audio_path"] == result.input_audio_path
    finally:
        db.close()
        cache.cleanup()


def test_audio_turn_requires_asr_and_cache(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        audio_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)
        pcm_bytes = b"\x00\x00" * 10
        conversation_id = orch.create_conversation("Test")

        try:
            orch.process_audio_turn(conversation_id, pcm_bytes, audio_meta, conversation_history="")
        except ValueError as exc:
            assert "ASR and audio_cache" in str(exc)
        else:
            raise AssertionError("Expected ValueError for missing ASR/audio_cache.")
    finally:
        db.close()


def test_audio_turn_fallback_persists_failure(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    cache = SessionAudioCache(root_dir=tmp_path / "audio")
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        asr = _FailingASR()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            asr=asr,
            audio_cache=cache,
        )

        audio_meta = AudioMeta(sample_rate=16000, channels=1, sample_width=2)
        pcm_bytes = b"\x00\x00" * 50
        conversation_id = orch.create_conversation("Test")
        result = orch.process_audio_turn(conversation_id, pcm_bytes, audio_meta, conversation_history="")

        assert result.asr_text == ""
        assert result.reply_text == ""
        assert result.assistant_turn_id == ""
        assert "error" in result.asr_meta

        with db.read_connection() as conn:
            row = conn.execute(
                "SELECT asr_text, asr_meta_json FROM user_turns WHERE id = ?",
                (result.user_turn_id,),
            ).fetchone()

        assert row[0] is None
        meta = json.loads(row[1])
        assert meta["error"] == "ASR failed"
    finally:
        db.close()
        cache.cleanup()


def test_regenerate_turn_audio_invokes_tts(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        tts = _FakeTTS()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            tts=tts,
            tts_voice="default",
            tts_speed=1.0,
        )

        conversation_id = orch.create_conversation("Test")
        result = orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")

        regen = orch.regenerate_turn_audio(result.assistant_turn_id)

        assert regen.audio_path == "/tmp/fake.wav"
        assert tts.calls
        assert tts.calls[-1]["turn_id"] == result.assistant_turn_id
    finally:
        db.close()


def test_regenerate_conversation_audio_handles_all_turns(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={
                "conversation": 5,
                "error_detection": 5,
                "correction": 5,
                "native_reformulation": 5,
                "explanation": 5,
                "jp_tts_normalisation": 5,
            },
            backend=_Backend('{"reply": "hello"}'),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        tts = _FakeTTS()
        orch = ConversationOrchestrator(
            db=db,
            llm=llm,
            prompt_loader=prompts,
            language="ja",
            tts=tts,
            tts_voice="default",
            tts_speed=1.0,
        )

        conversation_id = orch.create_conversation("Test")
        orch.process_text_turn(conversation_id, "こんにちは", conversation_history="")
        orch.process_text_turn(conversation_id, "こんばんは", conversation_history="")

        results = orch.regenerate_conversation_audio(conversation_id)

        assert len(results) == 2
        assert len(tts.calls) >= 2
        assert results[0].audio_path == "/tmp/fake.wav"
    finally:
        db.close()
