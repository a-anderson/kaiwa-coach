# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install
poetry install --with dev   # include pytest

# Run the app
poetry run python -m kaiwacoach.app

# Run tests
poetry run pytest -q                  # all tests
poetry run pytest -q -m "not slow"   # fast tests only (what CI runs)
poetry run pytest -q -m slow         # integration/model tests only
poetry run pytest -q tests/test_json_enforcement.py  # single file

# Smoke-test local models
poetry run python scripts/asr_smoke.py --language ja --seconds 6
poetry run python scripts/tts_smoke.py --text "こんにちは。" --lang_code j --voice jf_alpha
poetry run python scripts/llm_smoke.py --language ja
```

macOS setup (required once for local model dependencies):
```bash
poetry run bash scripts/setup_macos.sh
```

## Architecture

**Module boundaries** — do not blur these:

| Module | Responsibility |
|---|---|
| `src/kaiwacoach/ui/gradio_app.py` | Gradio layout and callback wiring only |
| `src/kaiwacoach/orchestrator.py` | Turn lifecycle: sequencing, timing, persistence |
| `src/kaiwacoach/models/` | Typed wrappers for ASR (`asr_whisper.py`), LLM (`llm_qwen.py`), TTS (`tts_kokoro.py`), and JSON enforcement (`json_enforcement.py`) |
| `src/kaiwacoach/prompts/` | Markdown prompt templates + `loader.py` (renders with `{var}` substitution, returns SHA256) |
| `src/kaiwacoach/textnorm/` | Japanese TTS normalisation, katakana conversion, protected-span tracking, invariant checks |
| `src/kaiwacoach/storage/` | `db.py` (SQLite single-writer queue with thread-safe reads), `blobs.py` (audio file cache) |
| `src/kaiwacoach/settings.py` | Config loading: defaults → `config.yaml` → env vars (`KAIWACOACH_*`) |
| `src/kaiwacoach/config/models.py` | Default model IDs (ASR/LLM/TTS) |

**Turn pipeline** (in `orchestrator.py`):
1. User input (text or audio) → ASR if audio
2. Conversation LLM role → `ConversationReply`
3. Optional correction pipeline (if errors detected): `error_detection` → `correction` → `native_reformulation` → `explanation`
4. TTS normalisation for Japanese → TTS synthesis
5. Persist all artefacts to SQLite + blob storage

**LLM role system**: Each LLM call uses a named role (`conversation`, `error_detection`, `correction`, `native_reformulation`, `explanation`, `jp_tts_normalisation`). Each role has a corresponding Pydantic schema in `json_enforcement.py` and a markdown prompt in `src/kaiwacoach/prompts/`. `parse_with_schema()` validates output and makes one repair attempt via an LLM repair call if validation fails.

**Storage**: `SQLiteWriter` runs all writes through a single background thread with a queue; reads use separate short-lived connections. Schema is applied at startup; a column-set mismatch triggers a DB reset (with a warning log). Raw `UPDATE` statements are forbidden — use `execute_update()` with the allowlist.

**Config**: `load_config()` in `settings.py` merges three layers. Env var names follow `KAIWACOACH_<SECTION>_<KEY>` (e.g. `KAIWACOACH_SESSION_LANGUAGE`). Model defaults come from `config/models.py`. Copy `config.example.yaml` to `config.yaml` for file-based overrides.

**Default models** (local, MLX-based, Apple Silicon only):
- ASR: `mlx-community/whisper-large-v3-mlx`
- LLM: `mlx-community/Qwen3-14B-8bit`
- TTS: `mlx-community/Kokoro-82M-bf16`

## Key conventions

- Prompts live in `src/kaiwacoach/prompts/` as `.md` files with `{var}` placeholders — never inline prompts in Python code.
- The textnorm pipeline applies invariant checks before TTS: if Japanese text is altered unexpectedly, it falls back rather than propagating a corrupted string.
- Slow tests (marked `@pytest.mark.slow`) require local models to be installed; CI only runs non-slow tests.
- `SQLiteWriter._ALLOWED_UPDATE_COLUMNS` must be kept in sync with schema changes.
