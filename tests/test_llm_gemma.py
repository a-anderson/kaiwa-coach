"""Tests for the Gemma LLM wrapper."""

from __future__ import annotations

import pytest

from kaiwacoach.models.json_enforcement import ConversationReply
from kaiwacoach.models.llm_gemma import GemmaLLM, _LLM_CACHE_MAX
from kaiwacoach.models.protocols import LLMResult


class _Backend:
    def __init__(self) -> None:
        self.last_prompt: str = ""
        self.last_max_tokens: int | None = None
        self.last_extra_eos: list[str] | None = None
        self.calls = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        self.last_prompt = prompt
        self.last_max_tokens = max_tokens
        self.last_extra_eos = extra_eos_tokens
        self.calls += 1
        return "ok"


def test_rejects_invalid_max_context_tokens() -> None:
    """max_context_tokens must be positive."""
    with pytest.raises(ValueError, match="max_context_tokens must be > 0"):
        GemmaLLM(model_id="model-x", max_context_tokens=0, role_max_new_tokens={"role": 1}, backend=_Backend())


def test_unknown_role_raises() -> None:
    """Unknown roles should raise a clear error."""
    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=10,
        role_max_new_tokens={"known": 5},
        backend=_Backend(),
    )
    with pytest.raises(ValueError, match="Unknown role"):
        llm.generate("prompt", role="unknown")


def test_no_think_suffix_for_any_role() -> None:
    """GemmaLLM must NOT append the Qwen3 think suffix for any role."""
    backend = _Backend()
    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"detect_and_correct": 96, "conversation": 256, "jp_tts_normalisation": 192},
        backend=backend,
    )

    for role in ("detect_and_correct", "jp_tts_normalisation", "conversation"):
        backend.calls = 0
        llm.clear_cache()
        llm.generate(f"my prompt for {role}", role=role)
        assert not backend.last_prompt.endswith("<think>\n\n</think>"), \
            f"Role {role!r} should not receive the no-think suffix"


def test_no_extra_eos_for_conversation_role() -> None:
    """GemmaLLM must NOT send extra_eos_tokens for the conversation role."""
    backend = _Backend()
    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 256},
        backend=backend,
    )

    llm.generate("prompt", role="conversation")

    assert backend.last_extra_eos is None


def test_conversation_role_uses_configured_temperature() -> None:
    """conversation role should receive the configured temperature; other roles get 0.0."""
    received: list[dict] = []

    class _CapturingBackend:
        def generate(self, prompt, max_tokens, extra_eos_tokens=None, temperature=0.0) -> str:
            received.append({"temperature": temperature})
            return '{"reply": "ok"}'

    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 5, "detect_and_correct": 5},
        backend=_CapturingBackend(),
        conversation_temperature=0.9,
    )

    llm.generate("prompt one", role="conversation")
    assert received[-1]["temperature"] == 0.9

    llm.generate("prompt two", role="detect_and_correct")
    assert received[-1]["temperature"] == 0.0


def test_generate_returns_metadata_with_backend_label() -> None:
    """Metadata should include the backend_label passed at construction."""
    class _MetaBackend:
        def generate(self, prompt, max_tokens, extra_eos_tokens=None, temperature=0.0) -> str:
            return "hello"

    llm = GemmaLLM(
        model_id="gemma4-model",
        max_context_tokens=100,
        role_max_new_tokens={"role": 3},
        backend=_MetaBackend(),
        token_counter=lambda _: 2,
        backend_label="ollama",
    )

    result = llm.generate("prompt", role="role")

    assert isinstance(result, LLMResult)
    assert result.text == "hello"
    assert result.meta["backend"] == "ollama"
    assert result.meta["model_id"] == "gemma4-model"
    assert result.meta["role"] == "role"
    assert result.meta["prompt_tokens"] == 2
    assert result.meta["cache_hit"] is False


def test_default_backend_label_is_mlx_lm() -> None:
    """Default backend_label should be 'mlx_lm' to match MLX convention."""
    class _MetaBackend:
        def generate(self, prompt, max_tokens, extra_eos_tokens=None, temperature=0.0) -> str:
            return "ok"

    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 5},
        backend=_MetaBackend(),
    )

    result = llm.generate("prompt", role="role")
    assert result.meta["backend"] == "mlx_lm"


def test_token_counter_enforces_context_limit() -> None:
    """Prompt tokens above max_context_tokens should raise ValueError."""
    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=3,
        role_max_new_tokens={"role": 5},
        backend=_Backend(),
        token_counter=lambda _: 10,
    )
    with pytest.raises(ValueError, match="Prompt exceeds max_context_tokens"):
        llm.generate("prompt", role="role")


def test_generate_uses_cache_for_identical_prompt() -> None:
    """Cache hit on the same prompt+role should avoid a second backend call."""
    backend = _Backend()
    llm = GemmaLLM(
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


def test_cache_bounded_at_max_size() -> None:
    """Cache must not grow beyond _LLM_CACHE_MAX entries."""
    backend = _Backend()
    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=10_000,
        role_max_new_tokens={"role": 5},
        backend=backend,
    )

    for i in range(_LLM_CACHE_MAX + 10):
        llm.generate(f"unique prompt nonce={i}", role="role")

    assert len(llm._cache) <= _LLM_CACHE_MAX


def test_generate_json_parses_schema() -> None:
    """generate_json should parse role-specific schemas."""
    class _JsonBackend:
        def generate(self, prompt, max_tokens, extra_eos_tokens=None, temperature=0.0) -> str:
            return '{"reply": "Bonjour."}'

    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"conversation": 3},
        backend=_JsonBackend(),
    )

    result = llm.generate_json("prompt", role="conversation")
    assert result.error is None
    assert isinstance(result.model, ConversationReply)
    assert result.model.reply == "Bonjour."


def test_max_new_tokens_capped_by_role() -> None:
    """Explicit max_new_tokens above the role cap should be clamped."""
    backend = _Backend()
    llm = GemmaLLM(
        model_id="model-x",
        max_context_tokens=100,
        role_max_new_tokens={"role": 5},
        backend=backend,
    )
    llm.generate("prompt", role="role", max_new_tokens=100)
    assert backend.last_max_tokens == 5
