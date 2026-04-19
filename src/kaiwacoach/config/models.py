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

# Gemma 4 IT (instruction-tuned) MLX variants.
# Note: e4b is only available in 4bit and bf16 (no 8bit checkpoint exists).
GEMMA4_E2B_8BIT = "mlx-community/gemma-4-e2b-it-8bit"
GEMMA4_E4B_4BIT = "mlx-community/gemma-4-e4b-it-4bit"
GEMMA4_E4B_BF16 = "mlx-community/gemma-4-e4b-it-bf16"
GEMMA4_26B_4BIT = "mlx-community/gemma-4-26b-a4b-it-4bit"
GEMMA4_26B_8BIT = "mlx-community/gemma-4-26b-a4b-it-8bit"

# Supported execution backends for the LLM.
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"mlx", "ollama"})

# Per-backend LLM model ID allowlists.
# MLX IDs are validated strictly against this curated set.
# Ollama IDs are pass-through (None) — Ollama validates availability at model-load time.
SUPPORTED_LLM_MODELS: dict[str, frozenset[str] | None] = {
    "mlx": frozenset({
        LLM_MODEL_ID_4BIT, LLM_MODEL_ID_8BIT, LLM_MODEL_ID_BF16,  # Qwen3
        GEMMA4_E2B_8BIT, GEMMA4_E4B_4BIT, GEMMA4_E4B_BF16,        # Gemma 4 small
        GEMMA4_26B_4BIT, GEMMA4_26B_8BIT,                          # Gemma 4 MoE
    }),
    "ollama": None,
}

# ASR and TTS use a flat allowlist (single backend each, no per-backend split needed).
SUPPORTED_MODELS: dict[str, frozenset[str]] = {
    "asr": frozenset({ASR_MODEL_ID}),
    "tts": frozenset({TTS_MODEL_ID}),
}
