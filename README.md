# KaiwaCoach

[![CI](https://github.com/a-anderson/kaiwa-coach/actions/workflows/tests.yaml/badge.svg)](https://github.com/a-anderson/kaiwa-coach/actions/workflows/tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)

KaiwaCoach is an offline-first language coaching app for Apple Silicon macOS.
It supports text and microphone turns, structured correction feedback, TTS (text-to-speech) playback, audio regeneration, and a shadowing mode for pronunciation practice. The UI is a Svelte SPA backed by a FastAPI server, with a clear separation between the frontend, API layer, model pipeline, and persistent storage.

## Demos

All product demos can be viewed in the [Feature Demos](docs/feature_demos.md) file.

### Chat

<p align="center">
  <img src="assets/demos/chat_turn.gif" alt="KaiwaCoach chat demo" style="max-width: 800px; width: 100%; height: auto;" />
</p>

## What It Does

- **Chat**: Runs a local conversational loop with:
    - user text/audio input
    - assistant reply generation
    - optional correction pipeline (detect errors + corrected sentence → explanation + native rewrite)
    - TTS synthesis of assistant reply
- **Narration**: paste text → TTS synthesis → preview and download (stateless, no conversation created)
- **Monologue**: submit text/audio → corrections + improvement summary; persisted to SQLite
- **Conversation Summary**: on-demand analysis of error patterns across a conversation; ephemeral, not persisted
- Persists conversations and supports list, load, resume, and delete
- **Audio regeneration** per-turn or per-conversation
- **Shadowing**: side-by-side listen + record comparison for any assistant turn
- Enforces JSON schema on LLM role outputs with bounded repair behaviour
- Applies Japanese TTS normalisation with invariant checks and fallback behaviour

## Architecture

### High-level flow

1. UI captures a user turn (text or audio).
2. Orchestrator persists inputs, runs the turn pipeline, and records timings.
3. LLM role calls are prompt-rendered and schema-validated.
4. TTS output is synthesised and returned to UI.
5. Conversation and turn artefacts are stored via the storage layer.

### Module boundaries

- [frontend/](frontend/)
    - Svelte + Vite SPA; communicates with the backend via REST + SSE
- [src/kaiwacoach/api/](src/kaiwacoach/api/)
    - FastAPI server, route handlers (conversations, turns, regen, audio), and request schemas
- [src/kaiwacoach/orchestrator.py](src/kaiwacoach/orchestrator.py)
    - turn lifecycle, sequencing, and timing
- [src/kaiwacoach/models/](src/kaiwacoach/models/)
    - `protocols.py`: shared result types and runtime-checkable protocols for ASR/LLM/TTS
    - `factory.py`: config-driven routing to the correct backend wrapper
    - concrete wrappers (`asr_whisper.py`, `llm_qwen.py`, `tts_kokoro.py`) and JSON enforcement
- [src/kaiwacoach/textnorm/](src/kaiwacoach/textnorm/)
    - normalisation and language invariants
- [src/kaiwacoach/storage/](src/kaiwacoach/storage/)
    - SQLite writer queue and media storage handling

## Engineering Notes

- LLM calls are role-based with explicit token caps.
- Role outputs are schema-validated with one repair attempt.
- Turn processing uses deterministic defaults and explicit timing logs.
- Before TTS, Japanese text is verified to ensure it was not accidentally altered.
- The correction pipeline uses two combined LLM calls (`detect_and_correct`, `explain_and_native`) rather than four sequential single-purpose calls, halving correction latency.
- The conversation role uses configurable sampling temperature (default `0.7`) for varied replies; all correction and normalisation roles use `0.0` for determinism.
- Schema migrations are non-destructive: additive changes (new columns with defaults) are applied via `ALTER TABLE` at startup without data loss; a full reset is never triggered automatically.

## Build Learnings

This project also served as a learning exercise for two areas:

- using an LLM coding assistant effectively during a real software project
- integrating local ASR, LLM, and TTS components into a reliable application

### LLM-assisted coding

Small, bounded requests with concrete evidence (real tracebacks, failing tests) consistently outperformed broad requests with vague descriptions. Asking for tradeoffs before implementation reduced overengineering; asking whether a change would pass senior+ review surfaced missing tests and brittle assumptions. Mode selection mattered — planning for learning and high-risk work, auto-accept for routine tasks — and switching modes mid-task when scope changed was more effective than leaving it at the session default. Successive fresh agents at review checkpoints caught things the implementing session missed; a single session reviewing its own work is not a reliable quality gate. Calibrating trust to roughly junior-mid engineer level and keeping the project instruction file (CLAUDE.md) as a living document were consistently useful throughout.

The human side is less tractable. Deep work and LLM-assisted iteration are in tension by design: the back-and-forth inherent in planning mode disrupts the mental state that hard problems require, and removing low-cognition work also removes the ambient attention where incidental discovery used to happen. Intensive assisted work produces a recognisable fatigue pattern — progressively skipping output, skimming, then accepting without review — that is difficult to catch from the inside because self-awareness is itself degraded when fatigued.

### Model integration

Reliability came less from the models themselves and more from the engineering around them. The most important improvements were: schema validation on every LLM role output (with bounded repair — one retry, then fail safe), keeping sequencing logic in the orchestrator rather than scattering it across wrappers, and timing instrumentation that made performance work concrete rather than intuition-driven.

A recurring lesson was that input-format problems are easy to mistake for model problems. Audio sample-rate mismatches, malformed JSON with reasoning-style preamble, and inconsistent parsing paths across similar roles all presented initially as model unreliability. Consistent handling across roles, explicit fallbacks, and tests targeting integration edges were more valuable than any single model improvement.

More detailed notes:

- [LLM-Assisted Coding Learnings](docs/llm_assisted_coding_learnings.md)
- [Model Integration Learnings](docs/model_integration_learnings.md)

## Tech Stack

- Python 3.11, FastAPI, Uvicorn
- Svelte 4, Vite, TypeScript, WaveSurfer.js
- SQLite
- Poetry
- Local ASR/LLM/TTS model wrappers in [src/kaiwacoach/models/](src/kaiwacoach/models/)

## Platform and Scope

- Primary target: **macOS Apple Silicon**
- Runtime target: **offline-first**
- Supported session languages:
    - `ja` - Japanese
    - `fr` - French
    - `en` - English
    - `es` - Spanish
    - `it` - Italian
    - `pt-br` - Portuguese (Brazil)

## Getting Started

### Prerequisites

- macOS Apple Silicon
- Python 3.11
- Poetry
- Node.js 18+

### Install

```bash
# Python backend
poetry install
poetry run bash scripts/setup_macos.sh

# Frontend
cd frontend && npm install
```

For test execution (installs the `dev` dependency group, including `pytest`):

```bash
poetry install --with dev
```

### Run

**Option A — production build** (frontend served by FastAPI):

```bash
cd frontend && npm run build   # writes to frontend/dist/
poetry run python -m kaiwacoach.app
# Open http://localhost:8000
```

**Option B — development** (hot-reload frontend, API on a separate port):

```bash
# Terminal 1
poetry run python -m kaiwacoach.app

# Terminal 2
cd frontend && npm run dev
# Open http://localhost:5173
```

## Configuration

Configuration is loaded from:

1. defaults
2. optional config file
3. environment overrides

For full details (keys, env vars, and load behaviour), see the [Configuration guide](docs/configuration.md).

Use [config.example.yaml](config.example.yaml) as the file-based template.

### LLM model variant

The default LLM is `mlx-community/Qwen3-14B-8bit`. Two model families are supported: **Qwen3** and **Gemma 4**, each available via MLX (Apple Silicon) or Ollama.

Recent model comparison shows the best balance of speed and performance comes from starting the Ollama server with MLX support enabled (`OLLAMA_MLX=1 ollama serve`) and setting `llm_backend: "ollama"` and `llm_id: "gemma4:e4b"` in the `config.yaml`.

**Qwen3 via MLX** (`llm_backend: "mlx"`):

| Variant         | Model ID                       | Trade-off                   |
| --------------- | ------------------------------ | --------------------------- |
| 4-bit           | `mlx-community/Qwen3-14B-4bit` | Minimum VRAM, lower quality |
| 8-bit (default) | `mlx-community/Qwen3-14B-8bit` | Balanced quality and VRAM   |
| bf16            | `mlx-community/Qwen3-14B-bf16` | Full precision, ~2× VRAM    |

**Gemma 4 via MLX** (`llm_backend: "mlx"`):

| Variant          | Model ID                                | Notes                          |
| ---------------- | --------------------------------------- | ------------------------------ |
| e2b-it 8-bit     | `mlx-community/gemma-4-e2b-it-8bit`     | 2B dense, lowest VRAM          |
| e4b-it 4-bit     | `mlx-community/gemma-4-e4b-it-4bit`     | 4B dense, low VRAM             |
| e4b-it bf16      | `mlx-community/gemma-4-e4b-it-bf16`     | 4B dense, full precision       |
| 26b-a4b-it 8-bit | `mlx-community/gemma-4-26b-a4b-it-8bit` | 26B MoE (~4B active), balanced |
| 26b-a4b-it 4-bit | `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE, minimum VRAM          |

**Via Ollama** (`llm_backend: "ollama"`, requires [Ollama](https://ollama.com) running locally):

| Model       | `llm_id`     |
| ----------- | ------------ |
| Qwen3 14B   | `qwen3:14b`  |
| Gemma 4 e4b | `gemma4:e4b` |
| Gemma 4 26b | `gemma4:26b` |

Set in `config.yaml`:

```yaml
models:
    llm_backend: "mlx"
    llm_id: "mlx-community/gemma-4-e4b-it-4bit"
```

Or via environment variables:

```bash
KAIWACOACH_MODELS_LLM_BACKEND=mlx KAIWACOACH_MODELS_LLM_ID=mlx-community/gemma-4-e4b-it-4bit poetry run python -m kaiwacoach.app
```

The active ASR, LLM, and TTS model IDs are logged at startup so the configured variant is always visible.

## Usage

### Text turn

1. Select language.
2. Enter text and send.
3. Review assistant response and optional correction outputs.
4. Play synthesised audio.

### Audio turn

1. Record input with microphone.
2. Send audio.
3. Review ASR-derived user text, assistant response, and optional corrections.
4. Play synthesised audio.

### Conversation persistence

- Past conversations are listed in the sidebar; click to load and resume.
- Conversations are automatically titled from the first user message and update in the sidebar as soon as the turn completes.
- Delete a single conversation or clear all history from the sidebar.

### User settings

- Click the ⚙ gear icon in the top bar to open the settings panel.
- Set an optional display name; the assistant will address you by name where natural. (Currently only works when the name is written with letters from the English alphabet)
- Set a proficiency level per language:
    - Japanese: two independent levels — grammar/vocabulary (N5–Native) and kanji reading (N5–Native)
    - All other languages: CEFR level (A1–Native)
- Levels take effect on the next turn; no conversation restart is required.

### Narration

- Click the **Narration** tab to open the narration panel.
- Paste or type text in the current session language and click **Generate Audio**.
- Preview the synthesised audio inline; click **↓ Download** to save the `.wav` file.
- No conversation is created — narration is stateless and session-only.

### Monologue mode

- Click the **Monologue** tab to open the monologue panel.
- Choose an input mode: **Text**, **Mic**, or **Upload**.
    - Text: type or paste your input, then click **Analyse**.
    - Mic: record using the microphone recorder, then click **Analyse**.
    - Upload: select an audio file; click **Analyse** to submit.
- The pipeline runs: ASR (audio only) → corrections → summary. Progress is shown per stage.
- Results are displayed in two sections:
    - **Results**: transcript (audio only), errors, corrected sentence, native rewrite, explanation.
    - **Summary**: improvement areas and overall assessment.
- Each result section has a copy button to copy the text to the clipboard.
- Past sessions appear in the sidebar under the Monologue tab; clicking one shows the read-only results view.

### Conversation summary

- Click **▼ Summarise** in the conversation header to analyse error patterns across the conversation.
- A collapsible panel appears above the chat thread showing top error patterns, priority areas, and overall notes.
- Click **▲ Summary** to collapse the panel, or use the × button inside it.
- The summary is generated fresh on each click and is not persisted.
- If no corrections were recorded for the conversation, an informational message is shown instead.

### Audio regeneration

- Click **↺** on any assistant bubble to regenerate its TTS audio.
- Click **↺ Regenerate all audio** in the conversation header to regenerate all turns.

### Downloading audio

- Click **↓ Download** on any assistant bubble to save the TTS audio `.wav` file.
- A download button is also available under the reference audio player in shadowing mode.

### Autoplay

Assistant audio plays automatically when a turn completes. Subsequent turns in a loaded conversation do not autoplay.

### Shadowing mode

- Click **Shadow** on any assistant bubble to open the shadowing panel.
- Listen to the reference audio on the left, record your attempt on the right.
- Press **Try Again** to re-record; press **✕** or Esc to close.

## Testing

Ensure dev dependencies are installed first:

```bash
poetry install --with dev
```

Run all tests:

```bash
poetry run pytest -q
```

Run non-slow tests:

```bash
poetry run pytest -q -m "not slow"
```

Run slow/integration tests:

```bash
poetry run pytest -q -m slow
```

### Frontend testing

The frontend does not have an automated test suite. This is a deliberate decision based on the current architecture and scope of the project.

The frontend is a thin Svelte SPA whose components are primarily glue code: they bind Svelte stores to the DOM, forward user events to the API layer, and render reactive state. The application logic that warrants automated verification — the turn pipeline, correction processing, schema validation, persistence, and SSE stream handling — lives entirely in the Python backend, which is covered by the existing test suite.

Adding a JavaScript test framework (e.g. Vitest with `@testing-library/svelte`) would introduce meaningful tooling overhead for tests that would largely exercise Svelte's own reactivity system rather than project-specific behaviour. The one frontend function with non-trivial logic (`buildHistory` in `InputArea.svelte`) is a pure string transform that is implicitly exercised through the end-to-end turn flow tests.

For changes that affect user-visible frontend behaviour, manual verification against the following scenarios is the expected quality gate:

- Language switching creates a new conversation; prior empty conversations are deleted
- Sending a text or audio turn produces progressive rendering (user message → typing indicator → assistant reply → audio)
- Loading a historical conversation restores the correct language, turns, and corrections
- Deleting a single conversation or all history removes it from the sidebar
- Shadowing mode opens, records, and closes correctly
- Audio regeneration updates the turn in place without a page reload
- Monologue text input: submitting text produces corrections and summary results
- Monologue audio input: ASR stage fires before corrections; results render on completion
- Monologue sidebar: completed session appears under the Monologue tab; clicking it shows the read-only results view with no input form
- Conversation summary: Summarise button generates a collapsible panel above the chat; panel collapses and re-generates correctly; conversations with no corrections show an informational message

If the frontend acquires significant standalone logic in future — for example, client-side state machines, complex derived computations, or custom hooks — introducing Vitest at that point would be appropriate.

## Smoke Scripts

These scripts verify that local models are installed correctly, can be loaded, and can run one basic inference path.

- [scripts/asr_smoke.py](scripts/asr_smoke.py)
    - ASR model load and short transcription
- [scripts/tts_smoke.py](scripts/tts_smoke.py)
    - TTS model load and synthesis from sample text
- [scripts/llm_smoke.py](scripts/llm_smoke.py)
    - LLM model load and sample generation

Run:

```bash
poetry run python scripts/asr_smoke.py --language ja --seconds 6
poetry run python scripts/tts_smoke.py --text "こんにちは。元気ですか？" --lang_code j --voice jf_alpha
poetry run python scripts/llm_smoke.py --language ja
```

## Data and Persistence

- Conversation and turn records are persisted in SQLite.
- Audio artefacts are persisted via the storage layer.
- Conversation metadata includes language/model context for resume behaviour.

## Performance and Reliability

- Active ASR, LLM, and TTS model IDs are logged at startup.
- Turn stage timings are logged (ASR, LLM, corrections, TTS, total).
- Role token caps and context limits are configurable for latency control.
- Schema enforcement prevents invalid role outputs from silently propagating.

## Evaluation

The project currently provides evaluation in three areas:

1. **Automated reliability checks**
2. **Schema/repair robustness checks**
3. **Per-stage latency instrumentation**

### Automated reliability checks

- CI runs non-slow tests on each push and pull request:
    - `poetry run pytest -q -m "not slow"`
- Full local suite (including slow tests) is available with:
    - `poetry run pytest -q`

Latest full local snapshot (2026-04-22):

- `385 passed`

### Schema and repair robustness

LLM role outputs are schema-validated, with one repair attempt on invalid output.  
These paths are covered in tests under:

- [tests/test_json_enforcement.py](tests/test_json_enforcement.py)
- [tests/test_orchestrator_text_flow.py](tests/test_orchestrator_text_flow.py)

### Latency instrumentation

Turn processing logs stage timings for:

- ASR
- LLM generation
- corrections (detect_and_correct / explain_and_native)
- TTS
- total turn time

Instrumentation lives in:

- [src/kaiwacoach/orchestrator.py](src/kaiwacoach/orchestrator.py)

## Limitations

- Current scope is local Apple Silicon execution.
- Browser microphone access requires a secure context (localhost or HTTPS).

## Development Notes

This repo is not currently accepting external contributions, but these notes document project conventions.

- Keep prompts in [src/kaiwacoach/prompts/](src/kaiwacoach/prompts/)
- Preserve module boundaries (UI/orchestrator/models/storage/textnorm)
- Add or update tests with behaviour changes
- Prefer small, reviewable pull requests with clear scope

## Licence

This project is licensed under the MIT Licence.  
See [LICENSE](LICENSE).

## Acknowledgements

KaiwaCoach is built on open-source tooling and local model ecosystems, including FastAPI, Svelte, WaveSurfer.js, SQLite, Ollama, and local model runtimes.
