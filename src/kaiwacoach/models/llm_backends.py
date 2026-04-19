"""LLM backend implementations.

Each backend implements the LLMBackend protocol and handles the mechanics of
inference (model loading, token sampling, HTTP calls). Model-family-specific
logic (think-tag suppression, EOS heuristics) lives in the wrapper layer
(e.g. QwenLLM, GemmaLLM), not here.
"""

from __future__ import annotations

import inspect
import json
import urllib.error
import urllib.request
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
    """Ollama HTTP backend — calls the local Ollama daemon at localhost:11434.

    Offline-first: all calls go to localhost; no external network traffic.
    Requires the Ollama daemon to be running before the application starts.
    Call check_available() at startup to surface a clear error if it is not.
    """

    _BASE_URL = "http://localhost:11434"
    _GENERATE_TIMEOUT = 120  # seconds; LLM generation can take time

    def __init__(self, model_id: str, suppress_thinking: bool = False) -> None:
        self._model_id = model_id
        # When True, adds "think": false to API requests. Required for thinking
        # models (e.g. gemma4:26b) where the thought phase consumes token budget
        # before the actual answer is generated.
        self._suppress_thinking = suppress_thinking

    @classmethod
    def check_available(cls) -> None:
        """Verify the Ollama daemon is reachable. Raises RuntimeError if not.

        Call this at startup (from factory.build_llm) so a missing daemon
        produces a clear error before any model loading is attempted.
        """
        try:
            with urllib.request.urlopen(cls._BASE_URL, timeout=3):
                pass
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama daemon is not available at {cls._BASE_URL}. "
                "Start Ollama before launching kaiwa-coach with llm_backend=ollama. "
                f"Detail: {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        if extra_eos_tokens:
            options["stop"] = extra_eos_tokens

        payload: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self._suppress_thinking:
            payload["think"] = False
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._BASE_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._GENERATE_TIMEOUT) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama API returned HTTP {exc.code} for model {self._model_id!r}. "
                f"Detail: {exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed for model {self._model_id!r}: {exc}"
            ) from exc

        return str(response_data.get("response", ""))
