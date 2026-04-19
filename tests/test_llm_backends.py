"""Tests for LLM backend implementations."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

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


# --- OllamaBackend ---

def _make_urlopen_response(body: dict) -> MagicMock:
    """Build a mock context-manager response for urllib.request.urlopen."""
    encoded = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_ollama_backend_init_stores_model_id() -> None:
    """OllamaBackend should store the model ID for later use."""
    backend = OllamaBackend("gemma4:e4b")
    assert backend._model_id == "gemma4:e4b"


def test_ollama_backend_generate_returns_response_field() -> None:
    """OllamaBackend.generate should return the 'response' field from the API reply."""
    mock_resp = _make_urlopen_response({"response": "Bonjour.", "done": True})

    with patch("urllib.request.urlopen", return_value=mock_resp):
        backend = OllamaBackend("gemma4:e4b")
        result = backend.generate("Hello", max_tokens=64)

    assert result == "Bonjour."


def test_ollama_backend_generate_sends_correct_payload() -> None:
    """OllamaBackend.generate should POST the right payload to Ollama."""
    captured: dict = {}
    mock_resp = _make_urlopen_response({"response": "ok"})

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        backend = OllamaBackend("qwen3:14b")
        backend.generate("my prompt", max_tokens=128, temperature=0.7)

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["method"] == "POST"
    assert captured["payload"]["model"] == "qwen3:14b"
    assert captured["payload"]["prompt"] == "my prompt"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["num_predict"] == 128
    assert captured["payload"]["options"]["temperature"] == 0.7


def test_ollama_backend_generate_passes_stop_sequences() -> None:
    """extra_eos_tokens should be forwarded to Ollama as 'stop' option."""
    captured: dict = {}
    mock_resp = _make_urlopen_response({"response": "ok"})

    def _fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        backend = OllamaBackend("gemma4:e4b")
        backend.generate("prompt", max_tokens=32, extra_eos_tokens=["}"])

    assert captured["payload"]["options"]["stop"] == ["}"]


def test_ollama_backend_generate_omits_stop_when_none() -> None:
    """When extra_eos_tokens is None, 'stop' should not appear in the payload."""
    captured: dict = {}
    mock_resp = _make_urlopen_response({"response": "ok"})

    def _fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        backend = OllamaBackend("gemma4:e4b")
        backend.generate("prompt", max_tokens=32, extra_eos_tokens=None)

    assert "stop" not in captured["payload"]["options"]


def test_ollama_backend_suppress_thinking_adds_think_false_to_payload() -> None:
    """suppress_thinking=True should add 'think': false to the Ollama request payload."""
    captured: dict = {}
    mock_resp = _make_urlopen_response({"response": "ok"})

    def _fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        backend = OllamaBackend("gemma4:26b", suppress_thinking=True)
        backend.generate("prompt", max_tokens=256)

    assert captured["payload"].get("think") is False


def test_ollama_backend_no_think_field_when_suppress_thinking_false() -> None:
    """suppress_thinking=False (default) should not add 'think' to the payload."""
    captured: dict = {}
    mock_resp = _make_urlopen_response({"response": "ok"})

    def _fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        backend = OllamaBackend("qwen3:14b")
        backend.generate("prompt", max_tokens=256)

    assert "think" not in captured["payload"]


def test_ollama_backend_generate_raises_on_url_error() -> None:
    """Network errors should be wrapped in RuntimeError with a clear message."""
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        backend = OllamaBackend("gemma4:e4b")
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            backend.generate("prompt", max_tokens=32)


def test_ollama_backend_generate_raises_on_http_error() -> None:
    """HTTP error responses should be wrapped in RuntimeError."""
    http_err = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),  # type: ignore[arg-type]
        fp=BytesIO(b"model not found"),
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        backend = OllamaBackend("unknown:model")
        with pytest.raises(RuntimeError, match="HTTP 404"):
            backend.generate("prompt", max_tokens=32)


def test_ollama_backend_check_available_raises_when_daemon_not_running() -> None:
    """check_available should raise RuntimeError when Ollama is unreachable."""
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        with pytest.raises(RuntimeError, match="Ollama daemon is not available"):
            OllamaBackend.check_available()


def test_ollama_backend_check_available_succeeds_when_daemon_running() -> None:
    """check_available should not raise when Ollama responds."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        OllamaBackend.check_available()  # should not raise
