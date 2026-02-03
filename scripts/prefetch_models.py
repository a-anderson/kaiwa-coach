#!/usr/bin/env python
"""
Prefetch and cache all required MLX / Hugging Face models.
Run once after environment setup.
These models are large and may take significant time to download.
"""

from mlx_lm import load as load_lm
from mlx_audio.stt import load_model as load_asr
from mlx_audio.tts import load_model as load_tts

from kaiwacoach.config.models import (
    ASR_MODEL_ID,
    LLM_MODEL_ID,
    TTS_MODEL_ID,
)

def main():
    print("[prefetch] Loading ASR model...")
    load_asr(ASR_MODEL_ID)

    print("[prefetch] Loading LLM model...")
    load_lm(LLM_MODEL_ID)

    print("[prefetch] Loading TTS model...")
    load_tts(TTS_MODEL_ID)

    print("[prefetch] All models cached successfully.")

if __name__ == "__main__":
    main()
