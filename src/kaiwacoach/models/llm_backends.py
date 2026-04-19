"""LLM backend implementations.

Each backend implements the LLMBackend protocol and handles the mechanics of
inference (model loading, token sampling, HTTP calls). Model-family-specific
logic (think-tag suppression, EOS heuristics) lives in the wrapper layer
(e.g. QwenLLM, GemmaLLM), not here.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol


class LLMBackend(Protocol):
    """Protocol for LLM backend implementations.

    Implementations must provide a `generate` method that accepts a prompt and
    max token count, returning a decoded string.
    """

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str: ...


class MlxLmBackend:
    """MLX-LM backend loader and generator implementing the LLMBackend protocol."""

    def __init__(self, model_id: str) -> None:
        try:
            from mlx_lm import generate, load  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-lm is not available. Install the LLM dependency to enable generation."
            ) from exc

        model, tokenizer = load(model_id)
        self._model = model
        self._tokenizer = tokenizer
        self._generate_fn = generate
        self._make_sampler = make_sampler
        self._sampler_greedy = make_sampler(temp=0.0)
        self._supports_extra_eos = "extra_eos_token" in inspect.signature(self._generate_fn).parameters

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        sampler = self._make_sampler(temp=temperature) if temperature > 0.0 else self._sampler_greedy
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        if self._supports_extra_eos and extra_eos_tokens:
            kwargs["extra_eos_token"] = list(extra_eos_tokens)
        return str(self._generate_fn(self._model, self._tokenizer, **kwargs))

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))


class OllamaBackend:
    """Ollama HTTP backend stub — not yet implemented.

    Will call the Ollama local API at http://localhost:11434 once implemented
    in the Gemma 4 integration PR. Satisfies LLMBackend protocol so factory
    routing can be wired now without blocking on the full implementation.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError(
            "OllamaBackend is not yet implemented. "
            "Ollama support is planned for the Gemma 4 integration PR."
        )

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError(
            "OllamaBackend is not yet implemented. "
            "Ollama support is planned for the Gemma 4 integration PR."
        )
