"""
Regression tests for the model import paths used by scripts/prefetch_models.py.

These paths were previously stale (mlx_audio.stt, mlx_audio.tts) and would have
silently broken fresh setups. This suite verifies the correct paths are importable
so CI catches any future breakage before it reaches users.
"""

from __future__ import annotations


def test_asr_prefetch_import_path() -> None:
    from mlx_whisper.load_models import load_model  # type: ignore  # noqa: F401


def test_tts_prefetch_import_path() -> None:
    from mlx_audio.tts.utils import load_model  # type: ignore  # noqa: F401


def test_llm_prefetch_import_path() -> None:
    from mlx_lm import load  # type: ignore  # noqa: F401
