# Kaiwa Coach

Offline-first Japanese and French conversation coach (ASR + LLM + TTS) for Apple Silicon macOS.

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

## Conventions

Modular code under src/kaiwacoach/

Prompts live in src/kaiwacoach/prompts/templates/

No inline prompts in code
