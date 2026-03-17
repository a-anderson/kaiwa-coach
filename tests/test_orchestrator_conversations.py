"""Conversation list/read/delete behaviors for persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.orchestrator import ConversationOrchestrator
from kaiwacoach.prompts.loader import PromptLoader
from kaiwacoach.storage.db import SQLiteWriter


class _Backend:
    def generate(self, prompt: str, max_tokens: int, extra_eos_tokens=None) -> str:
        return "{\"reply\": \"ok\"}"


def _setup_db(tmp_path: Path) -> SQLiteWriter:
    db_path = tmp_path / "kaiwacoach.sqlite"
    schema_path = Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "storage" / "schema.sql"
    db = SQLiteWriter(db_path=db_path, schema_path=schema_path)
    db.start()
    return db


def test_list_get_delete_conversation(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Test")

        conversations = orch.list_conversations()
        assert conversations
        assert conversations[0]["id"] == conversation_id

        convo = orch.get_conversation(conversation_id)
        assert convo["id"] == conversation_id
        assert convo["language"] == "ja"
        assert convo["turns"] == []

        orch.delete_conversation(conversation_id)
        remaining = orch.list_conversations()
        assert remaining == []
    finally:
        db.close()


def test_delete_all_conversations(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        orch.create_conversation("One")
        orch.create_conversation("Two")
        assert len(orch.list_conversations()) == 2

        orch.delete_all_conversations()
        assert orch.list_conversations() == []
    finally:
        db.close()


def test_delete_all_conversations_cascades_turns(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation("Has turns")
        user_turn_id = orch.persist_user_text_turn(conversation_id, "こんにちは")
        orch.generate_reply(conversation_id, user_turn_id, "こんにちは", conversation_history="")

        with db.read_connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM user_turns").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM assistant_turns").fetchone()[0] == 1

        orch.delete_all_conversations()

        with db.read_connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM user_turns").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM assistant_turns").fetchone()[0] == 0
    finally:
        db.close()


def test_auto_title_set_on_first_text_turn(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation()  # no title
        assert orch.list_conversations()[0]["title"] is None

        orch.process_text_turn(conversation_id, "Hello, how are you?", corrections_enabled=False)

        title = orch.list_conversations()[0]["title"]
        assert title == "Hello, how are you?"
    finally:
        db.close()


def test_auto_title_truncated_at_50_chars(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation()
        long_text = "A" * 60

        orch.process_text_turn(conversation_id, long_text, corrections_enabled=False)

        title = orch.list_conversations()[0]["title"]
        assert title is not None
        assert len(title) == 50
        assert title.endswith("…")
    finally:
        db.close()


def test_auto_title_not_overwritten_on_second_turn(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation()
        orch.process_text_turn(conversation_id, "First message", corrections_enabled=False)
        orch.process_text_turn(conversation_id, "Second message", corrections_enabled=False)

        title = orch.list_conversations()[0]["title"]
        assert title == "First message"
    finally:
        db.close()


def test_auto_title_not_overwritten_if_explicit_title_set(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        conversation_id = orch.create_conversation(title="My custom title")
        orch.process_text_turn(conversation_id, "First message", corrections_enabled=False)

        title = orch.list_conversations()[0]["title"]
        assert title == "My custom title"
    finally:
        db.close()


def test_delete_all_conversations_idempotent_on_empty(tmp_path: Path) -> None:
    db = _setup_db(tmp_path)
    try:
        llm = QwenLLM(
            model_id="model-x",
            max_context_tokens=100,
            role_max_new_tokens={"conversation": 5},
            backend=_Backend(),
        )
        prompts = PromptLoader(Path(__file__).resolve().parents[1] / "src" / "kaiwacoach" / "prompts")
        orch = ConversationOrchestrator(db=db, llm=llm, prompt_loader=prompts, language="ja")

        assert orch.list_conversations() == []
        orch.delete_all_conversations()
        assert orch.list_conversations() == []
    finally:
        db.close()
