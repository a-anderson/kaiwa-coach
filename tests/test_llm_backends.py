"""Tests for LLM backend implementations."""

from __future__ import annotations

import pytest

from kaiwacoach.models.llm_backends import MlxLmBackend, OllamaBackend


# --- MlxLmBackend ---

def test_mlx_backend_raises_if_mlx_lm_unavailable() -> None:
    """MlxLmBackend should raise RuntimeError if mlx-lm is not installed."""
    import builtins

    original_import = builtins.__import__

    def _import_hook(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("mlx_lm"):
            raise ImportError("no module named mlx_lm")
        return original_import(name, globals, locals, fromlist, level)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _import_hook)
        with pytest.raises(RuntimeError, match="mlx-lm is not available"):
            MlxLmBackend("model-x")


# --- OllamaBackend stub ---

def test_ollama_backend_init_stores_model_id() -> None:
    """OllamaBackend should store the model ID for later use."""
    backend = OllamaBackend("gemma4:e4b")
    assert backend._model_id == "gemma4:e4b"


def test_ollama_backend_generate_raises_not_implemented() -> None:
    """OllamaBackend.generate should raise NotImplementedError until fully implemented."""
    backend = OllamaBackend("qwen3:14b")
    with pytest.raises(NotImplementedError):
        backend.generate("prompt", max_tokens=100)


def test_ollama_backend_count_tokens_raises_not_implemented() -> None:
    """OllamaBackend.count_tokens should raise NotImplementedError until fully implemented."""
    backend = OllamaBackend("qwen3:14b")
    with pytest.raises(NotImplementedError):
        backend.count_tokens("some text")


def test_ollama_backend_generate_error_message_is_informative() -> None:
    """The NotImplementedError message should mention Gemma 4 integration PR for context."""
    backend = OllamaBackend("any-model")
    with pytest.raises(NotImplementedError, match="Gemma 4 integration"):
        backend.generate("prompt", max_tokens=10)
