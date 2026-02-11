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
