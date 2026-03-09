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

`tools/export_conversation.py` — stub for a planned conversation export utility; not yet implemented.

## Environment — do not modify

The Python environment is pre-created and owned by the maintainer.

- Do **not** create, delete, or recreate virtual environments.
- Do **not** run `poetry env use`, `python -m venv`, `pyenv`, or similar.
- Do **not** modify `pyproject.toml`, `poetry.lock`, or dependency versions.
- Do **not** install or uninstall packages.
- Assume all dependencies are already installed.
- If environment issues are detected: **stop, report clearly, and wait**. Do not attempt autonomous fixes.
- Only modify the environment when explicitly asked with language like "update the Poetry environment" or "change dependencies".

## Architecture

**Module boundaries** — do not blur these:

| Module | Responsibility |
|---|---|
| `src/kaiwacoach/ui/gradio_app.py` | Gradio layout and callback wiring only — no model logic |
| `src/kaiwacoach/orchestrator.py` | Turn lifecycle: sequencing, timing, persistence; steps are pure functions |
| `src/kaiwacoach/models/` | Typed wrappers for ASR (`asr_whisper.py`), LLM (`llm_qwen.py`), TTS (`tts_kokoro.py`), JSON enforcement (`json_enforcement.py`) — no DB or UI dependencies |
| `src/kaiwacoach/prompts/` | Markdown prompt templates + `loader.py` (renders with `{var}` substitution, returns SHA256) |
| `src/kaiwacoach/textnorm/` | All normalisation logic — no normalisation inside model wrappers |
| `src/kaiwacoach/storage/` | `db.py` (SQLite single-writer queue, single source of truth — no hidden caches), `blobs.py` (audio file cache) |
| `src/kaiwacoach/settings.py` | Config loading: defaults → `config.yaml` → env vars (`KAIWACOACH_*`) |
| `src/kaiwacoach/config/models.py` | Default model IDs (ASR/LLM/TTS) |

**Turn pipeline** (in `orchestrator.py`):
1. User input (text or audio) → ASR if audio
2. Persist intermediates **before** side effects (e.g. save text before TTS)
3. Conversation LLM role → `ConversationReply`
4. Optional correction pipeline (if errors detected): `error_detection` → `correction` → `native_reformulation` → `explanation`
5. TTS normalisation for Japanese → TTS synthesis
6. Persist all artefacts to SQLite + blob storage

**LLM role system**: Each LLM call uses a named role (`conversation`, `error_detection`, `correction`, `native_reformulation`, `explanation`, `jp_tts_normalisation`). Each role has:
- A Pydantic schema in `json_enforcement.py`
- A markdown prompt in `src/kaiwacoach/prompts/`
- An explicit per-role token cap

`parse_with_schema()` extracts the first valid JSON object (trailing content is ignored and logged), validates it, and makes **at most one** repair attempt via a repair prompt. On failure: fail safe with a minimal reply. Prompt hashes (SHA256 of the rendered prompt) are stored with each LLM call for reproducibility.

**Storage**: `SQLiteWriter` runs all writes through a single background thread with a queue; reads use separate short-lived connections. Schema is applied at startup; a column-set mismatch triggers a DB reset (with a warning log). Raw `UPDATE` statements are forbidden — use `execute_update()` with the allowlist.

**Config**: `load_config()` in `settings.py` merges three layers. Env var names follow `KAIWACOACH_<SECTION>_<KEY>` (e.g. `KAIWACOACH_SESSION_LANGUAGE`). Model defaults come from `config/models.py`. Copy `config.example.yaml` to `config.yaml` for file-based overrides. UI asset locations (e.g. logos) come from config and are passed into UI builders explicitly — do not use brittle `Path(...).parents[n]` traversal in feature code.

**Default models** (local, MLX-based, Apple Silicon only):
- ASR: `mlx-community/whisper-large-v3-mlx`
- LLM: `mlx-community/Qwen3-14B-8bit`
- TTS: `mlx-community/Kokoro-82M-bf16`

## Non-negotiable rules

- **Offline-first**: no network calls at runtime.
- **Deterministic pipelines**: randomness must be explicit, logged, and bounded.
- **Persist before side effects**: save text/audio before TTS or other irreversible steps.
- **Preserve raw user input** (text + audio) permanently.
- **Prompts only in `prompts/*.md`**: never inline prompt text in Python code.
- **All LLM outputs schema-validated**: one repair retry max, then fail safe.
- **Japanese invariant**: TTS normalisation must not alter Japanese substrings (byte-identical, with fallback to original on violation).

## Performance

Caching is hash-based:
- ASR: audio content hash
- LLM: prompt hash + role
- TTS: text + voice + speed

Per-step timings are recorded (ASR, LLM converse/analyse, normalise, TTS). UX targets (best-effort, M1 Pro): text turn < 4s, audio turn < 7s.

## Editing prompts

After changing any `.md` file in `src/kaiwacoach/prompts/`, run the prompt tests then do a manual smoke test:

```bash
poetry run pytest -q tests/test_prompt_schemas.py tests/test_prompt_rendering_suite.py tests/test_prompt_loader.py
poetry run python -m kaiwacoach.app   # verify the affected role works end-to-end
```

## Textnorm (Japanese)

The Japanese normalisation pipeline (`textnorm/`) is sensitive. The core rule: **Japanese character spans (hiragana, katakana, kanji) must be preserved byte-identical** through any normalisation step. If a candidate text violates this, `enforce_japanese_invariant()` falls back to the original and logs a warning. Run the full textnorm suite after any change to normalisation logic or prompts:

```bash
poetry run pytest -q tests/test_invariants.py tests/test_jp_katakana.py tests/test_jp_normalisation_golden.py tests/test_jp_tts_normalisation.py tests/test_protected_spans.py
```

Golden-case inputs and expected outputs live in `tests/fixtures/jp_normalisation_cases.json`.

## SQLite schema changes

When modifying `src/kaiwacoach/storage/schema.sql`, also update:

1. **`_ALLOWED_UPDATE_COLUMNS`** in `storage/db.py` — add/remove column names to match.
2. **`_schema_needs_reset`** in `storage/db.py` — update the `required_columns` set if columns on `conversations` change.
3. **`schema_version`** — bump the version integer in `schema.sql`.

Prefer nullable columns or columns with defaults for additive changes. The current behaviour on a column-set mismatch is a full local DB reset (acceptable for single-user MVP — note this in any schema PR).

## Gradio / UI

- Gradio version is pinned at `6.5.1` — verify constructor args against that version before use.
- Prefer `container=False` to remove outer component frames instead of CSS-heavy overrides.
- Keep UI CSS in named module-level constants, not inline strings inside `build_ui()`.
- Use stable `elem_id` values for CSS targeting and tests.
- For dynamic UI updates (theme/logo/audio remounts), keep outputs centralised in shared lists to reduce ordering bugs.
- Theme/logo updates must be deterministic and language-driven; loading a conversation must also sync language-dependent UI state.
- Fallback behaviour for missing assets must be explicit and tested (e.g. language logo falls back to `ja` logo).

## Testing

- Slow tests (`@pytest.mark.slow`) require local models; CI runs only non-slow tests.
- Prefer `poetry run pytest -q <targeted tests>` during iteration, then run broader suites before claiming completion.
- When fixing a bug, add or adjust at least one regression test that would have caught it.
- Every LLM role and repair prompt path must have schema tests.
- Storage changes require round-trip tests covering DB and audio blobs.
- End-to-end smoke tests are required for single text turn and single audio turn paths.
- For UI callback changes: read the existing tests first — they encode required Gradio output ordering. Add targeted tests for callback output shape/order. Expect both manual UI validation and test updates to be required when touching `gradio_app.py`.
- For startup/config wiring changes: add tests in `tests/test_app_startup.py`.
- If a change affects user-visible flow (language switching, audio submit, history load, delete), mention manual verification steps explicitly.

## Code quality

- Write to a standard suitable for senior+ review: clear, testable, easy to reason about.
- Prefer small explicit helpers over large callback bodies when output ordering or state transitions are non-trivial.
- When refactoring: move code first, then change behaviour in a separate step.
- Add docstrings/comments only for non-obvious behaviour, tradeoffs, or framework quirks.
- Start with the simplest solution that is likely to work. If choosing a more complex option, document why the simpler one was insufficient.
- Keep changes focused and diffs small; avoid broad refactors unless explicitly requested.
