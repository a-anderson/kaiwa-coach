"""Tests for the Qwen LLM wrapper."""

from __future__ import annotations

from typing import Dict

import pytest

from kaiwacoach.models.llm_qwen import LLMResult, MlxLmBackend, QwenLLM


class _Backend:
    def __init__(self) -> None:
        self.last_max_tokens: int | None = None

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.last_max_tokens = max_tokens
        return "ok"


def test_rejects_invalid_max_context_tokens() -> None:
    """max_context_tokens must be positive."""
    with pytest.raises(ValueError, match="max_context_tokens must be > 0"):
        QwenLLM(model_id="model-x", max_context_tokens=0, role_max_new_tokens={"role": 1}, backend=_Backend())


def test_unknown_role_raises() -> None:
    """Unknown roles should raise a clear error."""
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=10,
        role_max_new_tokens={"known": 5},
        backend=_Backend(),
    )
    with pytest.raises(ValueError, match="Unknown role"):
        llm.generate("prompt", role="unknown")


def test_token_counter_enforces_context_limit() -> None:
    """Prompt tokens above max_context_tokens should fail."""
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=3,
        role_max_new_tokens={"role": 5},
        backend=_Backend(),
        token_counter=lambda _: 10,
    )
    with pytest.raises(ValueError, match="Prompt exceeds max_context_tokens"):
        llm.generate("prompt", role="role")


def test_max_new_tokens_capped_by_role() -> None:
    """Explicit max_new_tokens should be capped by the role limit."""
    backend = _Backend()
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 5},
        backend=backend,
    )
    llm.generate("prompt", role="role", max_new_tokens=10)
    assert backend.last_max_tokens == 5


def test_max_new_tokens_uses_role_default() -> None:
    """Without an override, role default max_new_tokens is used."""
    backend = _Backend()
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 7},
        backend=backend,
    )
    llm.generate("prompt", role="role")
    assert backend.last_max_tokens == 7


def test_generate_returns_metadata() -> None:
    """Metadata should include timing, role, model_id, and token info."""
    class _MetaBackend:
        def generate(self, prompt: str, max_tokens: int) -> str:
            return "hello"

    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 3},
        backend=_MetaBackend(),
        token_counter=lambda _: 2,
    )

    result = llm.generate("prompt", role="role")

    assert isinstance(result, LLMResult)
    assert result.text == "hello"
    assert result.meta["backend"] == "mlx_lm"
    assert result.meta["model_id"] == "model-x"
    assert result.meta["role"] == "role"
    assert result.meta["max_new_tokens"] == 3
    assert result.meta["prompt_tokens"] == 2
    assert result.meta["elapsed_seconds"] >= 0


def test_default_backend_requires_mlx_lm() -> None:
    """MlxLmBackend should raise if mlx-lm is unavailable."""
    import builtins

    original_import = builtins.__import__

    def _import_hook(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ):
        if name.startswith("mlx_lm"):
            raise ImportError("no module named mlx_lm")
        return original_import(name, globals, locals, fromlist, level)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(builtins, "__import__", _import_hook)
        with pytest.raises(RuntimeError, match="mlx-lm is not available"):
            MlxLmBackend("model-x")


def test_generate_without_token_counter_allows_prompt() -> None:
    """When no token_counter is provided, prompts should pass through."""
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=1,
        role_max_new_tokens={"role": 2},
        backend=_Backend(),
        token_counter=None,
    )

    result = llm.generate("prompt", role="role")
    assert result.text == "ok"
    assert result.meta["prompt_tokens"] is None


def test_negative_max_new_tokens_uses_role_cap() -> None:
    """Negative max_new_tokens should be ignored in favor of role cap."""
    backend = _Backend()
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 5},
        backend=backend,
    )

    llm.generate("prompt", role="role", max_new_tokens=-10)
    assert backend.last_max_tokens == 5


def test_mlx_backend_uses_generate_and_sampler() -> None:
    """MLX-LM backend should call generate with a sampler."""
    captured: Dict[str, object] = {}

    class _FakeSampleUtils:
        @staticmethod
        def make_sampler(temp: float):
            captured["temp"] = temp
            return "sampler"

    class _FakeModule:
        @staticmethod
        def load(model_id: str):
            captured["model_id"] = model_id
            return "model", "tokenizer"

        @staticmethod
        def generate(model, tokenizer, prompt, max_tokens, sampler=None):
            captured["generate"] = {
                "model": model,
                "tokenizer": tokenizer,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
            return "output"

    import sys

    original_mlx_lm = sys.modules.get("mlx_lm")
    original_sample_utils = sys.modules.get("mlx_lm.sample_utils")

    sys.modules["mlx_lm"] = _FakeModule
    sys.modules["mlx_lm.sample_utils"] = _FakeSampleUtils

    backend = MlxLmBackend("model-x")
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 3},
        backend=backend,
    )
    result = llm.generate("prompt", role="role")

    assert result.text == "output"
    assert captured["model_id"] == "model-x"
    assert captured["temp"] == 0.0
    assert captured["generate"]["max_tokens"] == 3

    if original_mlx_lm is not None:
        sys.modules["mlx_lm"] = original_mlx_lm
    else:
        sys.modules.pop("mlx_lm", None)

    if original_sample_utils is not None:
        sys.modules["mlx_lm.sample_utils"] = original_sample_utils
    else:
        sys.modules.pop("mlx_lm.sample_utils", None)
