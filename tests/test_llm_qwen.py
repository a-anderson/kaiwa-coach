"""Tests for the Qwen LLM wrapper."""

from __future__ import annotations

import pytest

from kaiwacoach.models.json_enforcement import ConversationReply
from kaiwacoach.models.llm_backends import MlxLmBackend
from kaiwacoach.models.llm_qwen import QwenLLM, _LLM_CACHE_MAX
from kaiwacoach.models.protocols import LLMResult


class _Backend:
    def __init__(self) -> None:
        self.last_max_tokens: int | None = None
        self.calls = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.last_max_tokens = max_tokens
        self.calls += 1
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
        def generate(
            self,
            prompt: str,
            max_tokens: int,
            extra_eos_tokens: list[str] | None = None,
            temperature: float = 0.0,
        ) -> str:
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
    assert result.meta["cache_hit"] is False
    assert "prompt_hash" in result.meta



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
    assert result.meta["cache_hit"] is False


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
    assert backend.calls == 1
    llm.generate("prompt", role="role", max_new_tokens=-10)
    assert backend.calls == 1


def test_no_think_suffix_added_for_json_roles() -> None:
    """JSON extraction roles should receive the no-think suffix."""
    received_prompts: list[str] = []

    class _CapturingBackend:
        def generate(
            self,
            prompt: str,
            max_tokens: int,
            extra_eos_tokens: list[str] | None = None,
            temperature: float = 0.0,
        ) -> str:
            received_prompts.append(prompt)
            return '{"explanation": "ok"}'

    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={
            "detect_and_correct": 96,
            "normalise_name": 64,
            "conversation": 256,
        },
        backend=_CapturingBackend(),
    )

    llm.generate("my prompt", role="detect_and_correct")
    assert received_prompts[-1].endswith("<think>\n\n</think>")

    # normalise_name must suppress thinking — without this the 64-token cap is
    # consumed by the <think> block and the JSON output is never written.
    llm.generate("my prompt", role="normalise_name")
    assert received_prompts[-1].endswith("<think>\n\n</think>")

    llm.generate("my prompt", role="conversation")
    assert received_prompts[-1] == "my prompt"


def test_no_think_suffix_reflected_in_prompt_hash() -> None:
    """The prompt hash stored in meta should match the effective prompt sent to the model."""
    import hashlib

    received: list[str] = []

    class _CapturingBackend:
        def generate(
            self,
            prompt: str,
            max_tokens: int,
            extra_eos_tokens: list[str] | None = None,
            temperature: float = 0.0,
        ) -> str:
            received.append(prompt)
            return '{"corrected": "ok"}'

    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"detect_and_correct": 48},
        backend=_CapturingBackend(),
    )

    result = llm.generate("my prompt", role="detect_and_correct")
    effective = received[-1]
    assert result.meta["prompt_hash"] == hashlib.sha256(effective.encode()).hexdigest()


def test_conversation_role_uses_configured_temperature() -> None:
    """conversation role should receive the configured temperature; other roles get 0.0."""
    received: list[dict] = []

    class _CapturingBackend:
        def generate(
            self,
            prompt: str,
            max_tokens: int,
            extra_eos_tokens: list[str] | None = None,
            temperature: float = 0.0,
        ) -> str:
            received.append({"temperature": temperature})
            return '{"reply": "ok"}'

    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 5, "detect_and_correct": 5},
        backend=_CapturingBackend(),
        conversation_temperature=0.6,
    )

    llm.generate("prompt one", role="conversation")
    assert received[-1]["temperature"] == 0.6

    # Different prompt to avoid cache hit
    llm.generate("prompt two", role="detect_and_correct")
    assert received[-1]["temperature"] == 0.0


def test_temperature_reflected_in_llm_meta() -> None:
    """The temperature used for generation should appear in the result metadata."""
    class _CapturingBackend:
        def generate(
            self,
            prompt: str,
            max_tokens: int,
            extra_eos_tokens: list[str] | None = None,
            temperature: float = 0.0,
        ) -> str:
            return '{"reply": "ok"}'

    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 5},
        backend=_CapturingBackend(),
        conversation_temperature=0.8,
    )

    result = llm.generate("a prompt", role="conversation")
    assert result.meta["temperature"] == 0.8


def test_mlx_backend_uses_generate_and_sampler() -> None:
    """MLX-LM backend should call generate with a sampler."""
    captured: dict[str, object] = {}

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

    try:
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
    finally:
        if original_mlx_lm is not None:
            sys.modules["mlx_lm"] = original_mlx_lm
        else:
            sys.modules.pop("mlx_lm", None)

        if original_sample_utils is not None:
            sys.modules["mlx_lm.sample_utils"] = original_sample_utils
        else:
            sys.modules.pop("mlx_lm.sample_utils", None)


def test_generate_json_parses_schema() -> None:
    """generate_json should parse role-specific schemas."""
    class _JsonBackend:
        def generate(
            self,
            prompt: str,
            max_tokens: int,
            extra_eos_tokens: list[str] | None = None,
            temperature: float = 0.0,
        ) -> str:
            return '{"reply": "ok"}'

    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 3},
        backend=_JsonBackend(),
    )

    result = llm.generate_json("prompt", role="conversation")
    assert result.error is None
    assert isinstance(result.model, ConversationReply)
    assert result.model.reply == "ok"


def test_generate_uses_cache_for_identical_prompt() -> None:
    backend = _Backend()
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 5},
        backend=backend,
    )

    first = llm.generate("prompt", role="role")
    second = llm.generate("prompt", role="role")

    assert backend.calls == 1
    assert first.text == second.text == "ok"
    assert first.meta["cache_hit"] is False
    assert second.meta["cache_hit"] is True


def test_generate_cache_separates_roles_and_caps() -> None:
    backend = _Backend()
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role_a": 5, "role_b": 5},
        backend=backend,
    )

    llm.generate("prompt", role="role_a")
    llm.generate("prompt", role="role_b")
    llm.generate("prompt", role="role_a", max_new_tokens=3)

    assert backend.calls == 3


def test_cache_bounded_at_max_size() -> None:
    """Cache must not grow beyond _LLM_CACHE_MAX entries."""
    backend = _Backend()
    llm = QwenLLM(
        model_id="model-x",
        max_context_tokens=10_000,
        role_max_new_tokens={"role": 5},
        backend=backend,
    )

    for i in range(_LLM_CACHE_MAX + 10):
        llm.generate(f"unique prompt nonce={i}", role="role")

    assert len(llm._cache) <= _LLM_CACHE_MAX
