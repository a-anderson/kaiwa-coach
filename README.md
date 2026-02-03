<p align="center">
  <img src="assets/logo/kaiwacoach_logo.png" alt="KaiwaCoach logo" width="200"/>
</p>

<p align="center">
  <strong>KaiwaCoach</strong><br/>
  An offline conversational language coach
</p>

# Kaiwa Coach

An offline conversational language coaching application for Japanese and French, designed to run locally on Apple Silicon macOS.

## Docs

- `docs/PRD_v2.md`
- `docs/TDD_v1.md`
- `docs/Implementation_Checklist.md`

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.11.x
- Poetry

### Speech stack (current)

- ASR + TTS via `mlx-audio`
- Japanese TTS requires:
    - `phonemizer-fork` (not `phonemizer`)
    - `unidic` dictionary assets (via `python -m unidic download`)

## Setup (Poetry)

```bash
poetry env use 3.11
poetry install
poetry run bash scripts/setup_macos.sh
```

> KaiwaCoach uses the Hugging Face cache (typically ~/.cache/huggingface/) for model storage.
> Models are prefetched via scripts/prefetch_models.py.

## Smoke tests

```bash
poetry run python scripts/smoke_asr.py --language ja --seconds 6
poetry run python scripts/smoke_tts.py --text "こんにちは。元気ですか？" --lang_code j --voice jf_alpha
poetry run python scripts/smoke_llm.py --language ja
```

## Run (planned)

```bash
poetry run python -m kaiwacoach.app
```
