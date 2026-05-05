"""Regression tests for model import paths fixed in the prefetch script (previously mlx_audio.stt, mlx_audio.tts)."""


def test_asr_prefetch_import_path() -> None:
    from mlx_whisper.load_models import load_model  # type: ignore  # noqa: F401


def test_tts_prefetch_import_path() -> None:
    from mlx_audio.tts.utils import load_model  # type: ignore  # noqa: F401


def test_llm_prefetch_import_path() -> None:
    from mlx_lm import load  # type: ignore  # noqa: F401
