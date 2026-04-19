"""Qwen LLM wrapper with token limits and metadata capture."""

from __future__ import annotations

import time
import hashlib
from typing import Any, Callable, Mapping

from kaiwacoach.models.json_enforcement import ParseResult, parse_with_schema
from kaiwacoach.models.llm_backends import LLMBackend, MlxLmBackend
from kaiwacoach.models.protocols import LLMResult
from kaiwacoach.utils import BoundedDict

_LLM_CACHE_MAX = 256


class QwenLLM:
    """LLM wrapper enforcing context limits and per-role max tokens."""

    def __init__(
        self,
        model_id: str,
        max_context_tokens: int,
        role_max_new_tokens: Mapping[str, int],
        backend: LLMBackend | None = None,
        token_counter: Callable[[str], int] | None = None,
        conversation_temperature: float = 0.0,
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
        conversation_temperature : float
            Sampling temperature for the conversation role. All other roles use 0.0.
            The application default (0.7) lives in LLMConfig; this parameter
            defaults to 0.0 so direct instantiation in tests stays deterministic.
        """
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        self._model_id = model_id
        self._max_context_tokens = max_context_tokens
        self._role_max_new_tokens = dict(role_max_new_tokens)
        self._token_counter = token_counter
        self._backend = backend or MlxLmBackend(self._model_id)
        self._conversation_temperature = conversation_temperature
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

        # Apply no-think prefix before token counting, hashing, and generation
        # so all metadata reflects the prompt actually sent to the model.
        effective_prompt = prompt + "<think>\n\n</think>" if role in self._NO_THINK_ROLES else prompt

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
        # Temperature is intentionally excluded from the cache key. It is fixed
        # at startup and conversation prompts are effectively unique (history
        # grows each turn), so cache collisions across different temperatures
        # cannot occur in practice.
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

    # Roles that should suppress Qwen3 thinking mode. Appending the empty
    # think block signals to the model that the reasoning phase is complete
    # and it should output the answer directly, avoiding wasted token budget.
    _NO_THINK_ROLES: frozenset[str] = frozenset(
        {"jp_tts_normalisation", "detect_and_correct", "explain_and_native"}
    )

    def _default_generator(self, **_: Any) -> tuple[str, dict[str, Any]]:
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
        # Only the conversation role uses non-zero temperature; all correction
        # and normalisation roles stay deterministic at 0.0.
        temperature = self._conversation_temperature if role == "conversation" else 0.0
        return (
            self._backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                extra_eos_tokens=extra_eos,
                temperature=temperature,
            ),
            {"backend": "mlx_lm", "temperature": temperature},
        )

    def clear_cache(self) -> None:
        """Clear the in-memory response cache."""
        self._cache.clear()
