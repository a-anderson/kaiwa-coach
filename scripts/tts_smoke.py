# tts_smoke.py
"""
TTS smoke test for KaiwaCoach using Kokoro via mlx-audio.
Generates a WAV file from text.

Usage:
  python tts_smoke.py --text "こんにちは。元気ですか？" --lang_code j --voice jf_alpha --out tmp/tts_ja.wav
  python tts_smoke.py --text "Bonjour, comment ça va ?" --lang_code f --voice ff_siwis --out tmp/tts_fr.wav

Install:
  pip install -U mlx-audio soundfile numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MODEL_ID = "mlx-community/Kokoro-82M-bf16"


def preflight_kokoro_deps() -> None:
    """
    Fail fast with an actionable message if the misaki/phonemizer stack is incompatible.
    This catches the common:
      AttributeError: EspeakWrapper has no attribute set_data_path
    """
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing phonemizer backend. Install phonemizer-fork:\n"
            "  pip uninstall -y phonemizer\n"
            "  pip install -U phonemizer-fork\n"
        ) from e

    if not hasattr(EspeakWrapper, "set_data_path"):
        import phonemizer  # type: ignore
        raise RuntimeError(
            "Incompatible phonemizer backend detected.\n"
            "Misaki expects EspeakWrapper.set_data_path, but it is missing.\n\n"
            "Fix:\n"
            "  pip uninstall -y phonemizer\n"
            "  pip install -U phonemizer-fork\n"
            "  pip install -U \"misaki[en]\" espeakng-loader\n\n"
            f"Debug:\n  phonemizer imported from: {getattr(phonemizer, '__file__', 'unknown')}\n"
        )
    
    # Japanese pipeline dependency check: unidic dictionary assets (mecabrc) must exist
    try:
        import unidic  # type: ignore
        import pathlib
        mecabrc = pathlib.Path(unidic.DICDIR) / "mecabrc"
        if not mecabrc.exists():
            raise RuntimeError(
                "UniDic dictionary assets missing (required for Japanese Kokoro pipeline).\n"
                f"Expected mecabrc at: {mecabrc}\n\n"
                "Fix:\n"
                "  pip install -U unidic fugashi\n"
                "  python -m unidic download\n"
            )
    except ImportError:
        pass


def synth_kokoro_mlx_audio(
    text: str,
    model_id: str = DEFAULT_MODEL_ID,
    voice: str = "jf_alpha",
    speed: float = 1.0,
    lang_code: str = "j",
) -> tuple[np.ndarray, int]:
    """Synthesis using mlx-audio TTS API."""
    try:
        from mlx_audio.tts.utils import load_model  # type: ignore
    except Exception as e:
        raise RuntimeError("Failed to import mlx-audio TTS. Ensure `pip install -U mlx-audio` works.") from e

    preflight_kokoro_deps()
    model = load_model(model_id)

    results = list(model.generate(text=text, voice=voice, speed=speed, lang_code=lang_code))
    if not results:
        raise RuntimeError("mlx-audio TTS returned no audio results.")

    r0 = results[0]
    audio_mx = getattr(r0, "audio", None)
    if audio_mx is None:
        raise RuntimeError("mlx-audio TTS result had no `.audio` field.")

    audio = np.asarray(audio_mx, dtype=np.float32)

    sr = getattr(r0, "sample_rate", None)
    if sr is None:
        sr = getattr(model, "sample_rate", 24000)
    return audio, int(sr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--out", type=str, default="tmp/tts_smoke.wav")
    p.add_argument("--voice", type=str, default="jf_alpha")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument(
        "--lang_code",
        type=str,
        default="j",
        help="mlx-audio Kokoro lang_code (e.g. j for Japanese, f for French)",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[tts_smoke] Synthesising with model={args.model} voice={args.voice} lang_code={args.lang_code}")
    t0 = time.time()
    audio, sr = synth_kokoro_mlx_audio(
        args.text,
        model_id=args.model,
        voice=args.voice,
        speed=args.speed,
        lang_code=args.lang_code,
    )
    dt = time.time() - t0

    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak

    sf.write(out_path, audio, samplerate=sr, subtype="PCM_16")
    print(f"[tts_smoke] Wrote: {out_path} (sr={sr}) in {dt:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[tts_smoke] Cancelled.")
        sys.exit(130)
