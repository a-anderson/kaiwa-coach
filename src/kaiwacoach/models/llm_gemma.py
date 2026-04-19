"""Gemma LLM wrapper with token limits and metadata capture.

Mirrors QwenLLM but omits Qwen3-specific behaviour:
- No _NO_THINK_ROLES: does not append <think>\\n\\n</think> to any prompt.
- No extra_eos=["}"] heuristic (test empirically before enabling).

The backend label in result metadata is set at construction time so MLX and
Ollama calls are distinguishable in logs.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Mapping

from kaiwacoach.models.json_enforcement import ParseResult, parse_with_schema
from kaiwacoach.models.llm_backends import LLMBackend, MlxLmBackend
from kaiwacoach.models.protocols import LLMResult
from kaiwacoach.utils import BoundedDict

_LLM_CACHE_MAX = 256


class GemmaLLM:
    """LLM wrapper for Gemma 4 models, enforcing context limits and per-role token caps."""

    def __init__(
        self,
        model_id: str,
        max_context_tokens: int,
        role_max_new_tokens: Mapping[str, int],
        backend: LLMBackend | None = None,
        token_counter: Callable[[str], int] | None = None,
        conversation_temperature: float = 0.0,
        backend_label: str = "mlx_lm",
    ) -> None:
        """Initialise the Gemma LLM wrapper.

        Parameters
        ----------
        model_id : str
            Model identifier passed to the backend.
        max_context_tokens : int
            Maximum context tokens; prompts exceeding this raise ValueError.
        role_max_new_tokens : Mapping[str, int]
            Per-role generation token caps.
        backend : LLMBackend | None
            Backend for generation; defaults to MlxLmBackend.
        token_counter : Callable[[str], int] | None
            Token counting function; None skips context-limit enforcement.
        conversation_temperature : float
            Sampling temperature for the conversation role. All other roles use 0.0.
        backend_label : str
            Label recorded in result metadata to identify the backend in use.
        """
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        self._model_id = model_id
        self._max_context_tokens = max_context_tokens
        self._role_max_new_tokens = dict(role_max_new_tokens)
        self._token_counter = token_counter
        self._backend = backend or MlxLmBackend(self._model_id)
        self._conversation_temperature = conversation_temperature
        self._backend_label = backend_label
        self._generator = self._default_generator
        self._cache: BoundedDict[tuple[str, str, int], LLMResult] = BoundedDict(maxsize=_LLM_CACHE_MAX)

    def generate(self, prompt: str, role: str, max_new_tokens: int | None = None) -> LLMResult:
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

        # No think-suppression suffix for Gemma — Qwen3-specific behaviour.
        effective_prompt = prompt

        if self._token_counter is not None:
            prompt_tokens = self._token_counter(effective_prompt)
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

        prompt_hash = hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest()
        cache_key = (prompt_hash, role, effective_max_new_tokens)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_meta = dict(cached.meta)
            cached_meta["cache_hit"] = True
            return LLMResult(text=cached.text, meta=cached_meta)

        start = time.perf_counter()
        text, meta = self._generator(
            prompt=effective_prompt,
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
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def generate_json(
        self,
        prompt: str,
        role: str,
        max_new_tokens: int | None = None,
        repair_fn: Callable[[str], str] | None = None,
    ) -> ParseResult:
        """Generate and parse JSON output using the role schema.

        Parameters
        ----------
        prompt : str
            Prompt text.
        role : str
            Role name for schema selection.
        max_new_tokens : int | None
            Optional override for max new tokens.
        repair_fn : Callable[[str], str] | None
            Optional repair function for a single retry on invalid JSON.

        Returns
        -------
        ParseResult
            Parsed model or error details.
        """
        result = self.generate(prompt=prompt, role=role, max_new_tokens=max_new_tokens)
        return parse_with_schema(role=role, text=result.text, repair_fn=repair_fn)

    def _default_generator(self, **_: Any) -> tuple[str, dict[str, Any]]:
        prompt = _["prompt"]
        max_tokens = _["max_new_tokens"]
        role = _["role"]
        # No extra_eos heuristic for Gemma — test empirically before enabling.
        temperature = self._conversation_temperature if role == "conversation" else 0.0
        return (
            self._backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                extra_eos_tokens=None,
                temperature=temperature,
            ),
            {"backend": self._backend_label, "temperature": temperature},
        )

    def clear_cache(self) -> None:
        """Clear the in-memory response cache."""
        self._cache.clear()
