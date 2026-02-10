"""Qwen LLM wrapper with token limits and metadata capture."""

from __future__ import annotations

import time
from dataclasses import dataclass
import inspect
import hashlib
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from kaiwacoach.models.json_enforcement import ParseResult, parse_with_schema


@dataclass(frozen=True)
class LLMResult:
    text: str
    meta: Dict[str, Any]


class LLMBackend(Protocol):
    """Protocol for LLM backend implementations.

    Implementations must provide a `generate` method that accepts a prompt and
    max token count, returning a decoded string.
    """

    def generate(self, prompt: str, max_tokens: int, extra_eos_tokens: list[str] | None = None) -> str: ...


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
        self._sampler = make_sampler(temp=0.0)
        self._supports_extra_eos = "extra_eos_token" in inspect.signature(self._generate_fn).parameters

    def generate(self, prompt: str, max_tokens: int, extra_eos_tokens: list[str] | None = None) -> str:
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampler": self._sampler,
        }
        if self._supports_extra_eos and extra_eos_tokens:
            kwargs["extra_eos_token"] = list(extra_eos_tokens)
        return str(self._generate_fn(self._model, self._tokenizer, **kwargs))

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))


class QwenLLM:
    """LLM wrapper enforcing context limits and per-role max tokens."""

    def __init__(
        self,
        model_id: str,
        max_context_tokens: int,
        role_max_new_tokens: Mapping[str, int],
        backend: LLMBackend | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize the LLM wrapper.

        Parameters
        ----------
        model_id : str
            Model identifier to load.
        max_context_tokens : int
            Maximum context tokens allowed for prompts.
        role_max_new_tokens : Mapping[str, int]
            Per-role max token limits for generation.
        backend : LLMBackend | None
            Optional backend for generation (defaults to MLX-LM backend).
        token_counter : Callable[[str], int] | None
            Optional token counter for context length validation.
        """
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        self._model_id = model_id
        self._max_context_tokens = max_context_tokens
        self._role_max_new_tokens = dict(role_max_new_tokens)
        self._token_counter = token_counter
        self._backend = backend or MlxLmBackend(self._model_id)
        self._generator = self._default_generator
        self._cache: Dict[Tuple[str, str, int], LLMResult] = {}

    def generate(self, prompt: str, role: str, max_new_tokens: Optional[int] = None) -> LLMResult:
        """Generate a response for a given role with enforced token limits.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the model.
        role : str
            Logical role name for token cap lookup.
        max_new_tokens : int | None
            Optional override for max new tokens.

        Returns
        -------
        LLMResult
            Generated text and metadata.
        """
        if role not in self._role_max_new_tokens:
            raise ValueError(f"Unknown role: {role}")

        if self._token_counter is not None:
            prompt_tokens = self._token_counter(prompt)
            if prompt_tokens > self._max_context_tokens:
                raise ValueError(
                    f"Prompt exceeds max_context_tokens ({prompt_tokens} > {self._max_context_tokens})."
                )
        else:
            prompt_tokens = None

        role_cap = self._role_max_new_tokens[role]
        if max_new_tokens is None or max_new_tokens <= 0:
            effective_max_new_tokens = role_cap
        else:
            effective_max_new_tokens = min(max_new_tokens, role_cap)

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_key = (prompt_hash, role, effective_max_new_tokens)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_meta = dict(cached.meta)
            cached_meta["cache_hit"] = True
            return LLMResult(text=cached.text, meta=cached_meta)

        start = time.perf_counter()
        text, meta = self._generator(
            prompt=prompt,
            max_new_tokens=effective_max_new_tokens,
            role=role,
        )
        elapsed = time.perf_counter() - start

        meta = dict(meta)
        meta.update(
            {
                "model_id": self._model_id,
                "role": role,
                "max_new_tokens": effective_max_new_tokens,
                "prompt_tokens": prompt_tokens,
                "elapsed_seconds": elapsed,
                "prompt_hash": prompt_hash,
                "cache_hit": False,
            }
        )
        result = LLMResult(text=text, meta=meta)
        self._cache[cache_key] = result
        return result

    def count_tokens(self, text: str) -> int | None:
        if self._token_counter is None:
            return None
        return self._token_counter(text)

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def generate_json(
        self,
        prompt: str,
        role: str,
        max_new_tokens: Optional[int] = None,
        repair_fn: Callable[[str], str] | None = None,
    ) -> ParseResult:
        """Generate and parse JSON output using the role schema.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the model.
        role : str
            Logical role name for schema selection.
        max_new_tokens : int | None
            Optional override for max new tokens.
        repair_fn : Callable[[str], str] | None
            Optional repair function to attempt once on invalid JSON.

        Returns
        -------
        ParseResult
            Parsed model or error details.
        """
        result = self.generate(prompt=prompt, role=role, max_new_tokens=max_new_tokens)
        return parse_with_schema(role=role, text=result.text, repair_fn=repair_fn)

    def _default_generator(self, **_: Any) -> tuple[str, Dict[str, Any]]:
        """Default generator hook (uses the configured backend).

        Returns
        -------
        tuple[str, dict]
            Generated text and metadata.

        Raises
        ------
        RuntimeError
            If no LLM backend is configured.
        """
        prompt = _["prompt"]
        max_tokens = _["max_new_tokens"]
        role = _["role"]
        # Tradeoff: stopping on "}" reduces trailing garbage but can truncate
        # malformed outputs before a full JSON object is complete.
        extra_eos = ["}"] if role == "conversation" else None
        return (
            self._backend.generate(prompt=prompt, max_tokens=max_tokens, extra_eos_tokens=extra_eos),
            {"backend": "mlx_lm"},
        )

    def clear_cache(self) -> None:
        """Clear the in-memory response cache."""
        self._cache.clear()
