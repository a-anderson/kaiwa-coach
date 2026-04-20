# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install
poetry install --with dev   # include pytest

# Run the backend (FastAPI + Uvicorn)
poetry run python -m kaiwacoach.app

# Run the frontend dev server (Vite, proxies /api → localhost:8000)
cd frontend && npm install && npm run dev

# Build the frontend for production (output to frontend/dist/, served by FastAPI)
cd frontend && npm run build

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

`pytest.ini` sets `pythonpath = src`, so tests can `import kaiwacoach` directly without a `src.` prefix — no `conftest.py` path manipulation is needed.

`tools/export_conversation.py` — stub for a planned conversation export utility; not yet implemented.

## Runtime data (gitignored, never commit)

`storage/`, `*.db`, `audio/`, and `exports/` are all local-only runtime directories. They are gitignored and should never be committed.

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
| `frontend/` | Svelte + Vite SPA — all UI logic; communicates with backend via REST + SSE only |
| `src/kaiwacoach/api/server.py` | FastAPI app factory, router registration, static file serving, lifespan |
| `src/kaiwacoach/api/routes/` | Route handlers: `conversations.py`, `turns.py`, `regen.py`, `audio.py` |
| `src/kaiwacoach/orchestrator.py` | Turn lifecycle: sequencing, timing, persistence; steps are pure functions |
| `src/kaiwacoach/models/protocols.py` | Shared result types (`ASRResult`, `LLMResult`, `TTSResult`) and `@runtime_checkable` protocols (`ASRProtocol`, `LLMProtocol`, `TTSProtocol`) — concrete files import result types from here |
| `src/kaiwacoach/models/factory.py` | `build_asr`, `build_llm`, `build_tts` — routes config model IDs to the correct backend and wrapper; returns protocol types; add new backend routing here |
| `src/kaiwacoach/models/` | Concrete wrappers: `asr_whisper.py`, `llm_qwen.py`, `tts_kokoro.py`; JSON enforcement: `json_enforcement.py` — no DB or UI dependencies |
| `src/kaiwacoach/prompts/` | Markdown prompt templates + `loader.py` (renders with `{var}` substitution, returns SHA256) |
| `src/kaiwacoach/textnorm/` | All normalisation logic — no normalisation inside model wrappers |
| `src/kaiwacoach/storage/` | `db.py` (SQLite single-writer queue, single source of truth — no hidden caches), `blobs.py` (audio file cache) |
| `src/kaiwacoach/settings.py` | Config loading: defaults → `config.yaml` → env vars (`KAIWACOACH_*`) |
| `src/kaiwacoach/config/models.py` | Default model IDs (ASR/LLM/TTS) |

**Turn pipeline** (in `orchestrator.py`):
1. User input (text or audio) → ASR if audio
2. Persist intermediates **before** side effects (e.g. save text before TTS)
3. Conversation LLM role → `ConversationReply`
4. TTS normalisation for Japanese → TTS synthesis (runs before corrections so audio is available sooner)
5. Optional correction pipeline (2 combined LLM calls): `detect_and_correct` → `explain_and_native`
6. Persist all artefacts to SQLite + blob storage

**LLM role system**: Each LLM call uses a named role (`conversation`, `detect_and_correct`, `explain_and_native`, `jp_tts_normalisation`). Each role has:
- A Pydantic schema in `json_enforcement.py`
- A markdown prompt in `src/kaiwacoach/prompts/`
- An explicit per-role token cap
- Temperature: `conversation` uses `llm.conversation_temperature` (default `0.7`); all other roles use `0.0` (deterministic)

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
- **Narrow exception handling**: never use bare `except Exception` to swallow errors silently. Wrap only the specific operation that can fail (e.g. the ASR transcription call, not the entire turn pipeline). If a failure path must persist state before returning, re-raise after cleanup so the error reaches the appropriate handler (e.g. the SSE error event emitter).
- **Bounded in-memory caches**: all caches backed by a plain `dict` must be bounded. Use `_BoundedDict` (in `orchestrator.py`) or equivalent, with a named max-size constant. An unbounded cache is a memory leak in a long-running process.

## Performance

Caching is hash-based:
- ASR: audio content hash
- LLM: prompt hash + role
- TTS: text + voice + speed

Per-step timings are recorded (ASR, LLM converse/analyse, corrections detect_correct/explain_native, TTS). UX targets (best-effort, M1 Pro): text turn < 4s, audio turn < 7s.

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

## Frontend (Svelte / Vite)

- The frontend lives in `frontend/` and is a standalone Vite + Svelte 4 SPA.
- All API calls go through `frontend/src/lib/api/`; components never call `fetch` directly.
- UI state lives in `frontend/src/lib/stores/`; keep store updates co-located with the API call that triggers them.
- Language themes are CSS custom properties on `:root[data-language="<code>"]` in `frontend/src/styles/themes.css`.
- For development, run the Vite dev server (`npm run dev` in `frontend/`) alongside the FastAPI backend — Vite proxies `/api` to `localhost:8000`.
- For production, `npm run build` writes to `frontend/dist/`, which FastAPI serves as static files.
- **Supported languages**: the canonical list lives in `frontend/src/lib/constants.ts` (`SUPPORTED_LANGUAGES`). Validate session language against it before making API calls — surface errors immediately in the UI rather than relying on the backend to reject the request.
- **Blob URL cleanup**: always revoke `URL.createObjectURL()` URLs when they are no longer needed. Track the last-created URL in a component variable, revoke it before creating the next one, and revoke on `onDestroy`.
- **No magic string sentinels**: do not repurpose ID fields as status indicators. Use a typed optional field (e.g. `status?: 'pending'`) for in-flight state so code can check the field explicitly rather than comparing against a hardcoded string.
- **Dev-mode logging**: guard development-only diagnostics with `import.meta.env.DEV` so they are tree-shaken in production builds. Do not leave silent empty `catch {}` blocks where a `console.warn` would aid debugging.

**Progressive rendering**: In-flight turns are tracked via `pendingTurn: TurnRecord | null` in `sessionStore`. `InputArea` populates it immediately on submit and patches it as each SSE stage completes (ASR transcript → LLM reply → TTS audio → corrections). On SSE `complete`, it is moved into `turns` with real IDs. `ChatThread` renders `pendingTurn` separately below committed turns. When a turn completes, `InputArea` dispatches a `turncomplete` event that `App.svelte` catches to trigger a sidebar refresh (picking up the auto-set conversation title).

## Testing

- Slow tests (`@pytest.mark.slow`) require local models; CI runs only non-slow tests.
- Prefer `poetry run pytest -q <targeted tests>` during iteration, then run broader suites before claiming completion.
- After any test count change, update the passing test snapshot in `README.md` (search for "passed" under "Automated reliability checks").
- When fixing a bug, add or adjust at least one regression test that would have caught it.
- Every LLM role and repair prompt path must have schema tests.
- Storage changes require round-trip tests covering DB and audio blobs.
- End-to-end smoke tests are required for single text turn and single audio turn paths.
- For startup/config wiring changes: add tests in `tests/test_app_startup.py`.
- For frontend changes that affect user-visible flow (language switching, audio submit, history load, delete, shadowing), mention manual verification steps explicitly.
- **No `# pragma: no cover` on reachable error paths**: if a failure path exists in production code, it must be tested. A `# pragma: no cover` comment signals a missing test, not an acceptable gap.
- **SSE route tests must include a mid-stream failure case**: test a scenario where stage events are emitted before the orchestrator raises, verify stage events appear before the error event, and that no events follow the error. Both text and audio turn routes need this.
- **API-layer tests must assert all fields**: when a feature produces a multi-field result (e.g. corrections: `errors`, `corrected`, `native`, `explanation`), the API integration test must verify every field is correctly mapped through the full stack, not just a representative subset.

**Frontend testing**: The frontend does not have an automated test suite. This is intentional — components are thin glue code binding stores to the DOM, and all application logic with meaningful test value lives in the Python backend. Do not add a JavaScript test framework unless the frontend acquires substantial standalone logic (e.g. non-trivial state machines or derived computations). For user-visible frontend changes, manual verification is the expected quality gate.

## API conventions

- **HTTP error messages**: use short noun phrases in sentence case, no trailing period — e.g. `"Conversation not found"`, `"Audio file not found"`, `"Empty audio upload"`. Do not use verb-first phrases like `"Failed to retrieve…"`. For 4xx errors where the detail helps the client recover, append the cause after a colon: `f"Audio conversion failed: {exc}"`.
- **SSE error events**: must include a `request_id` field (generated per request in `_build_sse_generator`) so failures can be correlated between server logs and client-side error reports.
- **Shared route utilities**: if a helper function would appear in more than one route file, add it to `src/kaiwacoach/api/utils.py` and import from there. Do not duplicate.

## Investigation discipline

- **Target what you actually need to check.** Use `grep` or targeted `Read` with line offsets rather than reading whole files. Read a file in full only when the full content is genuinely needed.
- For "what's missing / what's broken?" questions: `git diff --name-only`, grep the changed files for relevant symbols, run the tests. Don't read every changed file top-to-bottom.

## Code quality

- Write to a standard suitable for senior+ review: clear, testable, easy to reason about.
- Prefer small explicit helpers over large callback bodies when output ordering or state transitions are non-trivial.
- When refactoring: move code first, then change behaviour in a separate step.
- Add docstrings/comments only for non-obvious behaviour, tradeoffs, or framework quirks. When a docstring is warranted, the description must be one concise sentence; Parameters and Returns sections are permitted where they add clarity.
- Start with the simplest solution that is likely to work. If choosing a more complex option, document why the simpler one was insufficient.
- Keep changes focused and diffs small; avoid broad refactors unless explicitly requested.
- **Python type hints**: use built-in types directly — `dict`, `list`, `tuple` — not `Dict`, `List` from `typing` (Python 3.9+ is required). Use `X | None` instead of `Optional[X]`.
- **Timing variables**: when measuring multiple sub-steps in the same function, use a short-lived local (e.g. `t0`) that is reassigned for each step, and a separate `start_total` that is never reassigned, so the total and per-step durations are unambiguously distinct.
- **Clean up after changes**: when a refactor or replacement makes code, variables, prompt files, schemas, config keys, env vars, or test fixtures obsolete, remove them in the same change. Do not leave dead code behind — unused items (schemas, role caps, prompt files, env var mappings, test role dicts) must be deleted when the feature that used them is replaced.
