# TODO: pin model revisions
ASR_MODEL_ID = "mlx-community/whisper-large-v3-mlx"

# Default LLM: 8-bit quantised (lower VRAM, faster). Switch to bf16 for full
# precision at the cost of ~2x memory, or 4-bit for minimum VRAM at the cost
# of quality. All variants use the same MLX-LM backend and QwenLLM wrapper —
# set via config.yaml or KAIWACOACH_MODELS_LLM_ID.
LLM_MODEL_ID_4BIT = "mlx-community/Qwen3-14B-4bit"
LLM_MODEL_ID_8BIT = "mlx-community/Qwen3-14B-8bit"
LLM_MODEL_ID_BF16 = "mlx-community/Qwen3-14B-bf16"

TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# Supported execution backends for the LLM.
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"mlx", "ollama"})

# Per-backend LLM model ID allowlists.
# MLX IDs are validated strictly against this curated set.
# Ollama IDs are pass-through (None) — Ollama validates availability at model-load time.
SUPPORTED_LLM_MODELS: dict[str, frozenset[str] | None] = {
    "mlx": frozenset({LLM_MODEL_ID_4BIT, LLM_MODEL_ID_8BIT, LLM_MODEL_ID_BF16}),
    "ollama": None,
}

# ASR and TTS use a flat allowlist (single backend each, no per-backend split needed).
SUPPORTED_MODELS: dict[str, frozenset[str]] = {
    "asr": frozenset({ASR_MODEL_ID}),
    "tts": frozenset({TTS_MODEL_ID}),
}
