# asr_smoke.py
"""
ASR smoke test for KaiwaCoach (offline, Apple Silicon).
Records a short microphone clip, saves WAV, runs MLX Whisper STT via mlx-audio with forced language.

Usage:
  python asr_smoke.py --language ja --seconds 6
  python asr_smoke.py --language fr --seconds 6

Install:
  pip install -U mlx-audio sounddevice soundfile numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo-asr-fp16"


@dataclass
class ASRResult:
    text: str
    meta: dict


def record_wav(out_path: Path, seconds: float, sample_rate: int = 16000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[asr_smoke] Recording {seconds:.1f}s @ {sample_rate}Hz to: {out_path}")
    print("[asr_smoke] Speak now...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio_i16 = np.clip(audio.squeeze(), -1.0, 1.0)
    sf.write(out_path, audio_i16, samplerate=sample_rate, subtype="PCM_16")
    print("[asr_smoke] Recording complete.")


def transcribe_mlx_audio_whisper(
    wav_path: Path,
    language: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> ASRResult:
    """Transcribe using mlx-audio Whisper STT.

    mlx-audio docs show:
      from mlx_audio.stt.generate import generate_transcription
      result = generate_transcription(model="...", audio="audio.wav")
      print(result.text)
    """
    try:
        from mlx_audio.stt.generate import generate_transcription  # type: ignore
    except Exception as e:
        raise RuntimeError("Failed to import mlx-audio STT. Ensure `pip install -U mlx-audio` works.") from e

    t0 = time.time()
    # Some mlx-audio versions may accept a language argument; others may not.
    try:
        res = generate_transcription(model=model_id, audio=str(wav_path), language=language)
    except TypeError:
        res = generate_transcription(model=model_id, audio=str(wav_path))
    dt = time.time() - t0

    text = getattr(res, "text", None)
    if text is None:
        text = str(res)

    meta = {"latency_s": dt, "model": model_id, "language_forced": language}
    return ASRResult(text=str(text).strip(), meta=meta)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["ja", "fr"], required=True, help="Forced ASR language for the session")
    p.add_argument("--seconds", type=float, default=6.0, help="Recording duration")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    p.add_argument("--out", type=str, default="tmp/asr_smoke.wav", help="Output WAV path")
    args = p.parse_args()

    wav_path = Path(args.out)
    record_wav(wav_path, seconds=args.seconds)

    print(f"[asr_smoke] Transcribing with model={args.model} language={args.language} ...")
    result = transcribe_mlx_audio_whisper(wav_path, language=args.language, model_id=args.model)

    print("\n=== TRANSCRIPT ===")
    print(result.text)
    print("==================\n")
    print("[asr_smoke] Meta:", result.meta)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[asr_smoke] Cancelled.")
        sys.exit(130)
