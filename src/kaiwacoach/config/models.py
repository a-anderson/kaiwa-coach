# TODO: pin model revisions
ASR_MODEL_ID = "mlx-community/whisper-large-v3-mlx"

# Default LLM: 8-bit quantised (lower VRAM, faster). Switch to bf16 for full
# precision at the cost of ~2x memory. Both variants use the same MLX-LM
# backend and QwenLLM wrapper — set via config.yaml or KAIWACOACH_MODELS_LLM_ID.
LLM_MODEL_ID = "mlx-community/Qwen3-14B-8bit"
LLM_MODEL_ID_BF16 = "mlx-community/Qwen3-14B-bf16"

TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"

# All validated model IDs per type. Adding a new model requires updating this dict.
SUPPORTED_MODELS: dict[str, frozenset[str]] = {
    "asr": frozenset({ASR_MODEL_ID}),
    "llm": frozenset({LLM_MODEL_ID, LLM_MODEL_ID_BF16}),
    "tts": frozenset({TTS_MODEL_ID}),
}