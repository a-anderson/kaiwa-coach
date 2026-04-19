"""Tests for model factory functions."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaiwacoach.models.factory as factory_module
from kaiwacoach.config.models import ASR_MODEL_ID, LLM_MODEL_ID_4BIT, LLM_MODEL_ID_8BIT, LLM_MODEL_ID_BF16, TTS_MODEL_ID
from kaiwacoach.models.asr_whisper import WhisperASR
from kaiwacoach.models.factory import _detect_family, build_asr, build_llm, build_tts
from kaiwacoach.models.llm_qwen import QwenLLM
from kaiwacoach.models.protocols import ASRProtocol, LLMProtocol, TTSProtocol
from kaiwacoach.models.tts_kokoro import KokoroTTS
from kaiwacoach.settings import AppConfig, LLMConfig, LLMRoleCaps, LoggingConfig, ModelsConfig, SessionConfig, StorageConfig, TTSConfig, UIConfig
from kaiwacoach.storage.blobs import SessionAudioCache


def _asr_config(asr_id: str = "test-asr-model", language: str = "ja") -> SimpleNamespace:
    """Minimal config stub sufficient for build_asr."""
    return SimpleNamespace(
        models=SimpleNamespace(asr_id=asr_id),
        session=SimpleNamespace(language=language),
    )


# --- build_asr (non-slow: WhisperASR does not load the model on init) ---

# Routing tests below (test_build_*_routes_to_*) document the *current* backend
# selection for each model type. They are expected to change when new backends
# are added and the routing logic in factory.py is updated.

def test_build_asr_routes_to_whisper_asr() -> None:
    """build_asr should route to WhisperASR for the current default config."""
    assert isinstance(build_asr(_asr_config()), WhisperASR)


def test_build_asr_wires_model_id() -> None:
    """build_asr should pass the configured model ID to the wrapper."""
    asr = build_asr(_asr_config(asr_id="mlx-community/whisper-large-v3-mlx"))
    assert asr.model_id == "mlx-community/whisper-large-v3-mlx"


def test_build_asr_wires_language() -> None:
    """build_asr should pass the configured session language to the wrapper."""
    asr = build_asr(_asr_config(language="fr"))
    assert asr.language == "fr"


def test_build_asr_satisfies_asr_protocol() -> None:
    """build_asr output should satisfy ASRProtocol."""
    assert isinstance(build_asr(_asr_config()), ASRProtocol)


# --- model ID constants ---

def test_llm_model_id_bf16_matches_expected_hub_path() -> None:
    """LLM_MODEL_ID_BF16 should point to the bf16 Qwen3-14B variant on the MLX hub."""
    assert LLM_MODEL_ID_BF16 == "mlx-community/Qwen3-14B-bf16"


def test_llm_model_id_4bit_matches_expected_hub_path() -> None:
    """LLM_MODEL_ID_4BIT should point to the 4-bit Qwen3-14B variant on the MLX hub."""
    assert LLM_MODEL_ID_4BIT == "mlx-community/Qwen3-14B-4bit"


def test_llm_model_id_constants_are_distinct() -> None:
    """The 8-bit, bf16, and 4-bit constants must all refer to different model IDs."""
    assert LLM_MODEL_ID_8BIT != LLM_MODEL_ID_BF16
    assert LLM_MODEL_ID_8BIT != LLM_MODEL_ID_4BIT
    assert LLM_MODEL_ID_BF16 != LLM_MODEL_ID_4BIT


# --- build_llm (non-slow with stubbed backend) ---


class _StubBackend:
    """Stub backend — satisfies LLMBackend protocol without loading a model or daemon.

    Also provides a no-op check_available classmethod so it can substitute for
    OllamaBackend in factory tests without requiring a running Ollama daemon.
    """

    def __init__(self, model_id: str, suppress_thinking: bool = False) -> None:
        pass

    @classmethod
    def check_available(cls) -> None:
        pass  # no-op in tests

    def generate(self, prompt: str, max_tokens: int, extra_eos_tokens: list[str] | None = None, temperature: float = 0.0) -> str:
        return ""

    def count_tokens(self, text: str) -> int:
        return 0


def _llm_config(llm_id: str, tmp_path: Path) -> AppConfig:
    """Minimal AppConfig sufficient for build_llm."""
    return AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id=llm_id, tts_id=TTS_MODEL_ID),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )


def test_build_llm_wires_bf16_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should wire the bf16 model ID through to QwenLLM when configured."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    llm = build_llm(_llm_config(LLM_MODEL_ID_BF16, tmp_path))

    assert isinstance(llm, QwenLLM)
    assert llm.model_id == LLM_MODEL_ID_BF16
    assert isinstance(llm, LLMProtocol)


def test_build_llm_wires_8bit_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should wire the 8-bit model ID through to QwenLLM when configured."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    llm = build_llm(_llm_config(LLM_MODEL_ID_8BIT, tmp_path))

    assert isinstance(llm, QwenLLM)
    assert llm.model_id == LLM_MODEL_ID_8BIT


def test_build_llm_wires_4bit_model_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should wire the 4-bit model ID through to QwenLLM when configured."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    llm = build_llm(_llm_config(LLM_MODEL_ID_4BIT, tmp_path))

    assert isinstance(llm, QwenLLM)
    assert llm.model_id == LLM_MODEL_ID_4BIT
    assert isinstance(llm, LLMProtocol)


def test_build_llm_wires_conversation_temperature(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should pass conversation_temperature from AppConfig through to QwenLLM."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id=LLM_MODEL_ID_8BIT, tts_id=TTS_MODEL_ID),
        llm=LLMConfig(conversation_temperature=0.4),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    llm = build_llm(config)

    assert isinstance(llm, QwenLLM)
    assert llm._conversation_temperature == 0.4


# --- build_llm (slow: MlxLmBackend loads the model on init) ---

@pytest.mark.slow
def test_build_llm_routes_to_qwen_llm(tmp_path: Path) -> None:
    """build_llm should route to QwenLLM for the default config (Qwen3+MLX)."""
    from kaiwacoach.settings import load_config
    config = load_config(config_path=tmp_path / "nonexistent.yaml")
    try:
        llm = build_llm(config)
    except RuntimeError as exc:
        pytest.skip(f"LLM backend unavailable: {exc}")
    try:
        assert isinstance(llm, QwenLLM)
        assert llm.model_id == config.models.llm_id
        assert llm.max_context_tokens == config.llm.max_context_tokens
        assert isinstance(llm, LLMProtocol)
    finally:
        del llm
        gc.collect()


# --- _detect_family ---

def test_detect_family_returns_gemma4_for_mlx_gemma4_prefix() -> None:
    """MLX Gemma 4 model IDs should resolve to the gemma4 family."""
    assert _detect_family("mlx-community/gemma-4-e4b-it-4bit") == "gemma4"
    assert _detect_family("mlx-community/gemma-4-26b-a4b-it-8bit") == "gemma4"
    assert _detect_family("mlx-community/gemma-4-e2b-it-8bit") == "gemma4"


def test_detect_family_returns_gemma4_for_ollama_gemma4_prefix() -> None:
    """Ollama Gemma 4 model IDs should resolve to the gemma4 family."""
    assert _detect_family("gemma4:e4b") == "gemma4"
    assert _detect_family("gemma4:26b") == "gemma4"
    assert _detect_family("gemma4:31b") == "gemma4"


def test_detect_family_returns_qwen3_for_mlx_qwen3_prefix() -> None:
    """MLX Qwen3 model IDs should resolve to the qwen3 family."""
    assert _detect_family("mlx-community/Qwen3-14B-8bit") == "qwen3"
    assert _detect_family("mlx-community/Qwen3-14B-4bit") == "qwen3"
    assert _detect_family("mlx-community/Qwen3-14B-bf16") == "qwen3"


def test_detect_family_returns_qwen3_for_ollama_qwen3_prefix() -> None:
    """Ollama Qwen3 model IDs should resolve to the qwen3 family."""
    assert _detect_family("qwen3:14b") == "qwen3"
    assert _detect_family("qwen3:7b") == "qwen3"


def test_detect_family_raises_for_unknown_prefix() -> None:
    """An unrecognised model ID prefix should raise ValueError at startup."""
    with pytest.raises(ValueError, match="Cannot determine LLM model family"):
        _detect_family("unknown-org/some-model")


def test_detect_family_error_message_lists_supported_prefixes() -> None:
    """The ValueError for an unknown prefix should list the known prefixes."""
    with pytest.raises(ValueError, match="mlx-community/Qwen3-"):
        _detect_family("totally-unknown")


# --- build_llm: backend routing ---

def test_build_llm_routes_to_gemma_llm_for_mlx_gemma4_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm with an MLX Gemma 4 model ID should return GemmaLLM."""
    from kaiwacoach.models.llm_gemma import GemmaLLM

    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="mlx-community/gemma-4-e4b-it-4bit", tts_id=TTS_MODEL_ID),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    llm = build_llm(config)

    assert isinstance(llm, GemmaLLM)
    assert llm.model_id == "mlx-community/gemma-4-e4b-it-4bit"
    assert isinstance(llm, LLMProtocol)


def test_build_llm_gemma_mlx_backend_label_is_mlx_lm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GemmaLLM built with MLX backend should report backend_label='mlx_lm' in metadata."""
    from kaiwacoach.models.llm_gemma import GemmaLLM

    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="mlx-community/gemma-4-e4b-it-4bit", tts_id=TTS_MODEL_ID),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    llm = build_llm(config)

    assert isinstance(llm, GemmaLLM)
    assert llm._backend_label == "mlx_lm"


def test_build_llm_gemma_ollama_suppresses_thinking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GemmaLLM via Ollama should have suppress_thinking=True to avoid token starvation."""
    created_backends: list = []

    class _CapturingStubBackend(_StubBackend):
        def __init__(self, model_id: str, suppress_thinking: bool = False) -> None:
            super().__init__(model_id)
            self.suppress_thinking = suppress_thinking
            created_backends.append(self)

    monkeypatch.setattr(factory_module, "OllamaBackend", _CapturingStubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="gemma4:26b", tts_id=TTS_MODEL_ID, llm_backend="ollama"),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    build_llm(config)

    assert len(created_backends) == 1
    assert created_backends[0].suppress_thinking is True


def test_build_llm_qwen3_ollama_does_not_suppress_thinking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """QwenLLM via Ollama should not suppress thinking (Qwen3 uses a different mechanism)."""
    created_backends: list = []

    class _CapturingStubBackend(_StubBackend):
        def __init__(self, model_id: str, suppress_thinking: bool = False) -> None:
            super().__init__(model_id)
            self.suppress_thinking = suppress_thinking
            created_backends.append(self)

    monkeypatch.setattr(factory_module, "OllamaBackend", _CapturingStubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="qwen3:14b", tts_id=TTS_MODEL_ID, llm_backend="ollama"),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    build_llm(config)

    assert len(created_backends) == 1
    assert created_backends[0].suppress_thinking is False


def test_build_llm_gemma_ollama_backend_label_is_ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GemmaLLM built with Ollama backend should report backend_label='ollama' in metadata."""
    from kaiwacoach.models.llm_gemma import GemmaLLM

    monkeypatch.setattr(factory_module, "OllamaBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="gemma4:e4b", tts_id=TTS_MODEL_ID, llm_backend="ollama"),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    llm = build_llm(config)

    assert isinstance(llm, GemmaLLM)
    assert llm._backend_label == "ollama"


def test_build_llm_ollama_backend_routes_to_qwen_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm with ollama backend and a qwen3: model ID should return QwenLLM."""
    monkeypatch.setattr(factory_module, "OllamaBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="qwen3:14b", tts_id=TTS_MODEL_ID, llm_backend="ollama"),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    llm = build_llm(config)

    assert isinstance(llm, QwenLLM)
    assert llm.model_id == "qwen3:14b"
    assert isinstance(llm, LLMProtocol)


def test_build_llm_unknown_model_id_prefix_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build_llm should raise ValueError when the model ID prefix is not recognised."""
    monkeypatch.setattr(factory_module, "MlxLmBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="unknown-org/mystery-model", tts_id=TTS_MODEL_ID),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    with pytest.raises(ValueError, match="Cannot determine LLM model family"):
        build_llm(config)


def test_build_llm_ollama_unknown_prefix_raises_at_build_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unknown Ollama model ID prefix passes config validation but raises at build_llm.

    Ollama IDs are pass-through in _validate_config (no allowlist). Family detection
    happens in build_llm, so the ValueError surfaces at startup during app.main() with
    a descriptive message listing the supported prefixes.
    """
    monkeypatch.setattr(factory_module, "OllamaBackend", _StubBackend)

    config = AppConfig(
        session=SessionConfig(language="ja"),
        models=ModelsConfig(asr_id=ASR_MODEL_ID, llm_id="custom-model:v1", tts_id=TTS_MODEL_ID, llm_backend="ollama"),
        llm=LLMConfig(),
        storage=StorageConfig(root_dir=str(tmp_path / "storage")),
        tts=TTSConfig(),
        logging=LoggingConfig(),
        ui=UIConfig(logo_dir=str(tmp_path / "logo")),
    )

    with pytest.raises(ValueError, match="Cannot determine LLM model family"):
        build_llm(config)


# --- build_tts (slow: MlxAudioBackend loads the model on init) ---

@pytest.mark.slow
def test_build_tts_routes_to_kokoro_tts(tmp_path: Path) -> None:
    """build_tts should route to KokoroTTS for the current default config."""
    from kaiwacoach.settings import load_config
    config = load_config()
    cache = SessionAudioCache(root_dir=tmp_path)
    try:
        tts = build_tts(config, cache)
    except RuntimeError as exc:
        pytest.skip(f"TTS backend unavailable: {exc}")
    try:
        assert isinstance(tts, KokoroTTS)
        assert tts.model_id == config.models.tts_id
        assert isinstance(tts, TTSProtocol)
    finally:
        cache.cleanup()
        del tts
        gc.collect()
