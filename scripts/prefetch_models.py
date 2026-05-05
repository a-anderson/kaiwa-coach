#!/usr/bin/env python
"""
Prefetch and cache all models required by kaiwa-coach.

For mlx backend: downloads ASR, LLM, and TTS weights from HuggingFace Hub.
For ollama backend: checks/pulls the LLM via Ollama; downloads ASR and TTS from HuggingFace Hub.

Re-running is safe: HuggingFace Hub caching makes MLX downloads idempotent,
and the Ollama path checks availability before pulling.
"""

import argparse
import shutil
import subprocess
import sys

from kaiwacoach.config.models import (
    ASR_MODEL_ID,
    LLM_MODEL_ID_8BIT,
    OLLAMA_DEFAULT_LLM_MODEL_ID,
    SUPPORTED_LLM_MODELS,
    TTS_MODEL_ID,
)

_mlx_allowed: frozenset[str] = SUPPORTED_LLM_MODELS.get("mlx") or frozenset()
_MLX_MODEL_IDS: list[str] = sorted(_mlx_allowed)


def _build_parser() -> argparse.ArgumentParser:
    mlx_list = "\n  ".join(_MLX_MODEL_IDS)
    parser = argparse.ArgumentParser(
        description="Prefetch models required by kaiwa-coach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Available MLX LLM model IDs:
  {mlx_list}

Ollama model IDs are passed through to 'ollama pull' (e.g. {OLLAMA_DEFAULT_LLM_MODEL_ID}, qwen3:14b).
The Ollama server must be running when using --backend ollama.
""",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "ollama"],
        default="mlx",
        help="LLM execution backend (default: mlx)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"LLM model ID to prefetch. "
            f"Defaults to '{LLM_MODEL_ID_8BIT}' for mlx or '{OLLAMA_DEFAULT_LLM_MODEL_ID}' for ollama."
        ),
    )
    return parser


def _prefetch_asr() -> None:
    print(f"[prefetch] Loading ASR model ({ASR_MODEL_ID})...")
    from mlx_whisper.load_models import load_model  # type: ignore
    load_model(ASR_MODEL_ID)
    print("[prefetch] ASR model ready.")


def _prefetch_tts() -> None:
    print(f"[prefetch] Loading TTS model ({TTS_MODEL_ID})...")
    from mlx_audio.tts.utils import load_model  # type: ignore
    load_model(TTS_MODEL_ID)
    print("[prefetch] TTS model ready.")


def _prefetch_mlx_llm(model_id: str) -> None:
    if model_id not in _mlx_allowed:
        valid = "\n  ".join(_MLX_MODEL_IDS)
        print(
            f"[prefetch] Error: '{model_id}' is not a supported MLX model ID.\n"
            f"Valid options:\n  {valid}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[prefetch] Loading LLM model ({model_id})...")
    from mlx_lm import load  # type: ignore
    load(model_id)
    print("[prefetch] LLM model ready.")


def _prefetch_ollama_llm(model_id: str) -> None:
    if not shutil.which("ollama"):
        print(
            "[prefetch] Error: 'ollama' binary not found. "
            "Install Ollama from https://ollama.com before using --backend ollama.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[prefetch] Checking Ollama for model '{model_id}'...")
    try:
        check = subprocess.run(
            ["ollama", "show", model_id],
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        print(
            "[prefetch] Error: 'ollama show' timed out. "
            "Verify the Ollama server is running and reachable.",
            file=sys.stderr,
        )
        sys.exit(1)

    if check.returncode == 0:
        print(f"[prefetch] Model '{model_id}' already available in Ollama.")
        return

    print(f"[prefetch] Model '{model_id}' not found locally. Pulling via Ollama...")
    pull = subprocess.run(["ollama", "pull", model_id])
    if pull.returncode != 0:
        print(
            f"[prefetch] Error: 'ollama pull {model_id}' failed. "
            "Verify the Ollama server is running and the model ID is correct.",
            file=sys.stderr,
        )
        sys.exit(pull.returncode)
    print(f"[prefetch] Model '{model_id}' pulled successfully.")


def main() -> None:
    args = _build_parser().parse_args()
    backend: str = args.backend
    model_id: str = args.model or (LLM_MODEL_ID_8BIT if backend == "mlx" else OLLAMA_DEFAULT_LLM_MODEL_ID)

    _prefetch_asr()

    if backend == "mlx":
        _prefetch_mlx_llm(model_id)
    else:
        _prefetch_ollama_llm(model_id)

    _prefetch_tts()

    print("[prefetch] All models ready.")


if __name__ == "__main__":
    main()
