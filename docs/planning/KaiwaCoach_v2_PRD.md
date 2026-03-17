# KaiwaCoach v2 — Technical Planning Document

Status: Draft
Date: 2026-03-17

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure](#2-directory-structure)
3. [API Reference](#3-api-reference)
4. [SSE Streaming Protocol](#4-sse-streaming-protocol)
5. [Frontend Component Architecture](#5-frontend-component-architecture)
6. [Theme and Colour System](#6-theme-and-colour-system)
7. [Audio Pipeline](#7-audio-pipeline)
8. [Shadowing Mode Design](#8-shadowing-mode-design)
9. [Audio Regeneration Design](#9-audio-regeneration-design)
10. [Migration Path](#10-migration-path)
11. [Implementation Phases](#11-implementation-phases)
12. [Testing Strategy](#12-testing-strategy)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        macOS host (local)                        │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              Browser (Safari / Chrome)                    │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────────┐  │   │
│  │  │  Svelte + Vite + TypeScript SPA                     │  │   │
│  │  │                                                     │  │   │
│  │  │  Sidebar │ ChatThread │ CorrectionCard │ Shadowing  │  │   │
│  │  │  WaveSurfer.js (waveform playback + Record plugin)  │  │   │
│  │  │  Svelte stores: session · ui · audio · theme        │  │   │
│  │  └───────────────────┬─────────────────────────────────┘  │   │
│  │                      │  HTTP REST + SSE  (localhost:8000)  │   │
│  └──────────────────────┼───────────────────────────────────-┘   │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐    │
│  │              FastAPI  (src/kaiwacoach/api/)               │    │
│  │                                                          │    │
│  │  routes/conversations.py   routes/turns.py               │    │
│  │  routes/audio.py           routes/stream.py (SSE)        │    │
│  │  deps.py  (orchestrator singleton, settings)             │    │
│  └───────┬───────────────────────────────┬──────────────────┘    │
│          │                               │                        │
│  ┌───────▼──────────┐        ┌───────────▼──────────────────┐    │
│  │  orchestrator.py │        │  storage/db.py + blobs.py    │    │
│  │  (UNCHANGED)     │        │  (UNCHANGED)                 │    │
│  └───────┬──────────┘        └──────────────────────────────┘    │
│          │                                                        │
│  ┌───────▼──────────────────────────────────────────────────┐    │
│  │  models/  ·  prompts/  ·  textnorm/  ·  settings.py      │    │
│  │  (ALL UNCHANGED)                                         │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Key principles:**
- The orchestrator, models, storage, prompts, textnorm, and settings layers are **not touched**.
- FastAPI is a thin routing layer that calls the orchestrator and serves the Svelte build.
- SSE replaces Gradio's chained callback pattern for pipeline progress.
- No external network calls — all JS/CSS dependencies bundled by Vite at build time.

---

## 2. Directory Structure

### Repository root

```
kaiwa-coach/
├── frontend/                   # Svelte app (new)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── app.ts              # Entry point
│       ├── App.svelte          # Root component
│       ├── lib/
│       │   ├── api/
│       │   │   ├── client.ts           # Fetch + SSE wrapper
│       │   │   ├── conversations.ts    # Conversation CRUD
│       │   │   ├── turns.ts            # Turn submission
│       │   │   └── audio.ts            # Audio file endpoints
│       │   ├── stores/
│       │   │   ├── session.ts          # Current conversation, language, turns
│       │   │   ├── ui.ts               # Loading states, errors, active panels
│       │   │   └── theme.ts            # Active language → CSS vars
│       │   ├── components/
│       │   │   ├── layout/
│       │   │   │   ├── Sidebar.svelte
│       │   │   │   └── MainPanel.svelte
│       │   │   ├── chat/
│       │   │   │   ├── ChatThread.svelte
│       │   │   │   ├── TurnPair.svelte         # User + assistant turn together
│       │   │   │   ├── UserBubble.svelte
│       │   │   │   ├── AssistantBubble.svelte
│       │   │   │   └── CorrectionCard.svelte
│       │   │   ├── audio/
│       │   │   │   ├── AudioRecorder.svelte    # MediaRecorder + live waveform
│       │   │   │   ├── AudioPlayer.svelte      # WaveSurfer playback
│       │   │   │   └── ShadowingPanel.svelte   # Side-by-side comparison
│       │   │   ├── history/
│       │   │   │   ├── ConversationList.svelte
│       │   │   │   └── ConversationItem.svelte
│       │   │   └── common/
│       │   │       ├── Spinner.svelte
│       │   │       ├── ErrorBanner.svelte
│       │   │       ├── LanguageSelector.svelte
│       │   │       └── ConfirmDialog.svelte
│       │   └── types/
│       │       └── api.ts              # TS types mirroring Pydantic models
│       └── styles/
│           ├── global.css              # Resets, base typography
│           └── themes.css              # CSS custom property definitions per language
├── src/
│   └── kaiwacoach/
│       ├── api/                        # NEW FastAPI layer
│       │   ├── __init__.py
│       │   ├── server.py               # FastAPI app factory + Uvicorn launch
│       │   ├── deps.py                 # Orchestrator singleton, settings injection
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── conversations.py
│       │   │   ├── turns.py
│       │   │   ├── audio.py
│       │   │   └── stream.py           # SSE endpoint
│       │   └── schemas/                # Pydantic request/response models
│       │       ├── __init__.py
│       │       ├── conversation.py
│       │       └── turn.py
│       ├── app.py                      # REPLACED: was Gradio launch, now calls api/server.py
│       ├── ui/
│       │   └── gradio_app.py           # DELETED after Phase 1 complete
│       ├── orchestrator.py             # UNCHANGED
│       ├── models/                     # UNCHANGED
│       ├── storage/                    # UNCHANGED
│       ├── prompts/                    # UNCHANGED
│       ├── textnorm/                   # UNCHANGED
│       ├── settings.py                 # UNCHANGED
│       └── constants.py                # UNCHANGED
└── docs/
    └── planning/
        ├── KaiwaCoach_v2_PRD.md        # this document
        └── ...
```

### Frontend build output

Vite builds to `frontend/dist/`. FastAPI serves this as static files at `/`. No separate web server needed.

---

## 3. API Reference

All endpoints are prefixed `/api`. The Svelte build is served from `/` (catch-all static).

### 3.1 Conversations

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/conversations` | List all conversations (summary view) |
| `POST` | `/api/conversations` | Create new conversation |
| `GET` | `/api/conversations/{id}` | Get conversation with full turn history + corrections |
| `PATCH` | `/api/conversations/{id}` | Rename conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `DELETE` | `/api/conversations` | Delete all conversations |

**`GET /api/conversations` response:**
```json
[
  {
    "id": "uuid",
    "title": "Conversation preview text...",
    "language": "ja",
    "created_at": "2026-03-17T10:00:00Z",
    "updated_at": "2026-03-17T10:05:00Z",
    "turn_count": 8
  }
]
```

**`GET /api/conversations/{id}` response:**
```json
{
  "id": "uuid",
  "language": "ja",
  "title": "...",
  "turns": [
    {
      "user_turn_id": "uuid",
      "assistant_turn_id": "uuid",
      "user_text": "...",
      "asr_text": "...",
      "reply_text": "...",
      "correction": {
        "id": "uuid",
        "errors": ["..."],
        "corrected": "...",
        "native": "...",
        "explanation": "..."
      },
      "has_user_audio": true,
      "has_assistant_audio": true,
      "created_at": "2026-03-17T10:01:00Z"
    }
  ]
}
```

### 3.2 Turns (SSE streaming)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/turns/text` | Submit text turn — returns SSE stream |
| `POST` | `/api/turns/audio` | Submit audio turn — returns SSE stream |
| `GET` | `/api/turns/{id}/corrections` | Get corrections for a specific turn |

**`POST /api/turns/text` request:**
```json
{
  "conversation_id": "uuid or null",
  "language": "ja",
  "text": "今日は天気がいいですね。",
  "corrections_enabled": true
}
```

**`POST /api/turns/audio` request:**
- `multipart/form-data`
- Fields: `audio` (binary WebM blob), `conversation_id` (string or empty), `language` (string), `corrections_enabled` (bool)

Both turn endpoints return an `text/event-stream` response. See Section 4.

### 3.3 Audio

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/audio/{session_path}` | Serve audio file from session cache |
| `POST` | `/api/turns/{id}/regenerate-audio` | Regenerate TTS for an assistant turn |
| `POST` | `/api/conversations/{id}/regenerate-audio` | Regenerate TTS for all turns |

**`POST /api/turns/{id}/regenerate-audio` response:**
```json
{
  "audio_url": "/api/audio/sessions/abc123/turn_xyz.wav"
}
```

### 3.4 Session / Settings

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/settings` | Get current config (language, model IDs, etc.) |
| `POST` | `/api/session/reset` | Clear current session state |

---

## 4. SSE Streaming Protocol

The turn endpoints stream pipeline progress as `text/event-stream`. The frontend subscribes using `EventSource` (or a custom `fetch`-based SSE reader for POST support, since `EventSource` is GET-only).

### Event types

```
event: stage
data: {"stage": "asr", "status": "running"}

event: stage
data: {"stage": "asr", "status": "complete", "transcript": "こんにちは！"}

event: stage
data: {"stage": "llm", "status": "running"}

event: stage
data: {"stage": "llm", "status": "complete", "reply": "こんにちは！元気ですか？"}

event: stage
data: {"stage": "corrections", "status": "running"}

event: stage
data: {
  "stage": "corrections",
  "status": "complete",
  "data": {
    "errors": ["Missing particle は"],
    "corrected": "...",
    "native": "...",
    "explanation": "..."
  }
}

event: stage
data: {"stage": "tts", "status": "running"}

event: stage
data: {"stage": "tts", "status": "complete", "audio_url": "/api/audio/sessions/.../turn.wav"}

event: complete
data: {
  "conversation_id": "uuid",
  "user_turn_id": "uuid",
  "assistant_turn_id": "uuid",
  "timings": { "asr_transcribe_seconds": 1.2, "llm_generate_seconds": 2.8, ... }
}

event: error
data: {"stage": "llm", "message": "LLM failed to generate valid response"}
```

### Stage sequence

- **Text turn:** `llm` → `corrections` (if enabled) → `tts` → `complete`
- **Audio turn:** `asr` → `llm` → `corrections` (if enabled) → `tts` → `complete`
- Any stage can emit `error` without terminating — subsequent stages continue where possible (matches current orchestrator graceful degradation).

### Frontend SSE client (`lib/api/client.ts`)

Because `EventSource` only supports GET, POST SSE requires a `fetch`-based reader:

```typescript
async function* streamTurn(url: string, body: object): AsyncGenerator<SSEEvent> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
    body: JSON.stringify(body),
  });
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // parse SSE frames from buffer, yield events
  }
}
```

---

## 5. Frontend Component Architecture

### Svelte stores (`lib/stores/`)

**`session.ts`**
```typescript
// Current conversation and turn state
export const sessionStore = writable<SessionState>({
  conversationId: null,
  language: 'ja',
  turns: [],               // TurnPair[] — displayed in ChatThread
  corrections_enabled: true,
});
```

**`ui.ts`**
```typescript
export const uiStore = writable<UIState>({
  pipelineStage: null,     // 'asr' | 'llm' | 'corrections' | 'tts' | null
  pipelineStatus: null,    // 'running' | 'complete' | 'error'
  isSubmitting: false,
  activePanel: 'chat',     // 'chat' | 'history'
  shadowingTurnId: null,   // Which turn is in shadowing mode
  errorMessage: null,
});
```

**`theme.ts`**
```typescript
export const themeStore = derived(sessionStore, ($s) => LANGUAGE_THEMES[$s.language]);
// On change: sets data-language attr on <html>, triggering CSS custom props
```

### Component tree

```
App.svelte
├── LanguageSelector.svelte       # Header — language dropdown, drives theme
├── Sidebar.svelte
│   ├── ConversationList.svelte
│   │   └── ConversationItem.svelte  (×N)
│   └── [New Conversation] button
└── MainPanel.svelte
    ├── ChatThread.svelte
    │   ├── TurnPair.svelte (×N)
    │   │   ├── UserBubble.svelte
    │   │   │   └── AudioPlayer.svelte  (user audio, if present)
    │   │   ├── AssistantBubble.svelte
    │   │   │   ├── AudioPlayer.svelte  (assistant audio)
    │   │   │   ├── [Shadow] button
    │   │   │   └── [Regen Audio] button
    │   │   └── CorrectionCard.svelte  (collapsible)
    │   └── PipelineProgress.svelte   # Shows current stage during submission
    └── InputArea.svelte
        ├── TextInput.svelte
        ├── AudioRecorder.svelte      # MediaRecorder + live WaveSurfer
        └── [Send] / [Send Audio] buttons
    └── ShadowingPanel.svelte         # Slides in below ChatThread when active
        ├── AudioPlayer.svelte        # Reference (assistant audio)
        └── AudioRecorder.svelte      # User's shadowing attempt
```

### Key component responsibilities

**`TurnPair.svelte`** — owns a single user/assistant exchange. Knows its `turn_id` for audio regen and shadowing. Receives correction data and expands/collapses `CorrectionCard`.

**`CorrectionCard.svelte`** — displays errors (list), corrected, native, explanation. Collapsible. Styled as a card, not textboxes.

**`AudioPlayer.svelte`** — wraps WaveSurfer.js. Props: `src: string`, `autoplay?: boolean`. Emits: `play`, `pause`, `ended`. Shows full waveform, play/pause, duration.

**`AudioRecorder.svelte`** — wraps WaveSurfer Record plugin. Shows live waveform during recording. On stop, emits the audio blob. Used in both InputArea and ShadowingPanel.

**`PipelineProgress.svelte`** — subscribes to `uiStore.pipelineStage`. Shows animated stage indicators (ASR → LLM → Corrections → TTS) during submission.

**`ShadowingPanel.svelte`** — activated when `uiStore.shadowingTurnId` is set. Fetches reference audio URL from the active turn. Side-by-side layout: reference (left) + user attempt (right). Multiple attempts supported.

---

## 6. Theme and Colour System

Language themes are defined as CSS custom properties. The Svelte `themeStore` sets `data-language` on `<html>` when the language changes.

**`styles/themes.css`:**
```css
:root[data-language="ja"] {
  --kc-primary: #c0392b;
  --kc-primary-light: #e74c3c;
  --kc-user-bubble: #fdf2f2;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #c0392b;
  --kc-waveform-active: #c0392b;
  --kc-waveform-inactive: #e0b0b0;
  --kc-correction-border: #e74c3c;
}

:root[data-language="fr"] {
  --kc-primary: #1565c0;
  --kc-primary-light: #1976d2;
  --kc-user-bubble: #f0f4ff;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #1565c0;
  --kc-waveform-active: #1565c0;
  --kc-waveform-inactive: #90b0d8;
  --kc-correction-border: #1976d2;
}

/* en, es, it, pt — similarly */
```

**`theme.ts` store:**
```typescript
import { derived } from 'svelte/store';
import { sessionStore } from './session';

themeStore.subscribe((lang) => {
  document.documentElement.setAttribute('data-language', lang);
});
```

WaveSurfer's `waveColor` and `progressColor` options are bound reactively to `--kc-waveform-inactive` and `--kc-waveform-active` via `getComputedStyle`.

---

## 7. Audio Pipeline

### Recording → Backend

1. **Browser records** via `MediaRecorder` API. Format: `audio/webm;codecs=opus` (default in Chrome/Safari).
2. On stop, `AudioRecorder.svelte` collects chunks into a `Blob`.
3. Blob is sent to `POST /api/turns/audio` as `multipart/form-data` with field name `audio`.
4. **FastAPI receives** the upload (`UploadFile`), writes to a temp file.
5. `deps.py` converts WebM → WAV using `soundfile` + `scipy.io.wavfile` (already in deps). Target: 16kHz mono (Whisper's expected format).
6. WAV is passed to `orchestrator.run_audio_turn()` as a file path.

### Backend → Frontend (audio playback)

1. TTS audio is written to session blob cache by `storage/blobs.py` (existing behaviour).
2. FastAPI's `GET /api/audio/{session_path}` streams the WAV file with `FileResponse`.
3. `AudioPlayer.svelte` receives the URL from the SSE `tts.complete` event and initialises WaveSurfer with it.

### Audio format notes

- All backend audio is WAV (16kHz or 24kHz depending on Kokoro output).
- WaveSurfer handles WAV natively via Web Audio API — no transcoding needed for playback.
- User audio uploaded as WebM is converted once on the backend and cached by hash (existing ASR cache behaviour preserved).

---

## 8. Shadowing Mode Design

### UX flow

1. Each `AssistantBubble.svelte` has a **[Shadow]** button (visible after audio has been generated).
2. Clicking [Shadow] sets `uiStore.shadowingTurnId = turn_id` and slides `ShadowingPanel.svelte` into view below the chat thread (fixed at bottom, above input area).
3. **ShadowingPanel layout:**
   ```
   ┌──────────────────────────────────────────────────────────┐
   │  Shadowing: [assistant reply text]               [Close] │
   ├─────────────────────────┬────────────────────────────────┤
   │  Reference              │  Your attempt                  │
   │  ┌───────────────────┐  │  ┌──────────────────────────┐  │
   │  │  [WaveSurfer wfm] │  │  │  [WaveSurfer Record wfm] │  │
   │  │  ▶ Play  ⏹ Stop  │  │  │  ⏺ Record  ⏹ Stop       │  │
   │  └───────────────────┘  │  └──────────────────────────┘  │
   │                         │  [Try Again]                   │
   └─────────────────────────┴────────────────────────────────┘
   ```
4. User presses **Record**, speaks, presses **Stop**. The waveform renders on the right.
5. User can play both side-by-side for comparison.
6. **[Try Again]** clears the right panel and restarts recording.
7. **[Close]** sets `uiStore.shadowingTurnId = null`.

### Implementation notes

- Shadowing recordings are **not persisted** — session-only, like the current audio cache.
- WaveSurfer Record plugin (`wavesurfer.js/dist/plugins/record.js`) handles live waveform visualisation during recording.
- The reference `AudioPlayer` and the attempt `AudioPlayer` share no state — they are independent WaveSurfer instances.
- Multiple shadowing attempts can be made; only the latest is displayed.

---

## 9. Audio Regeneration Design

### UX flow

1. Each `AssistantBubble.svelte` has a **[↺ Regen]** button (always visible, even when audio was previously generated).
2. Clicking [↺ Regen] shows a spinner inside the bubble and calls `POST /api/turns/{assistant_turn_id}/regenerate-audio`.
3. The backend calls the existing `orchestrator.regenerate_turn_audio()` method.
4. On success, the response includes `{ audio_url }`. The `AudioPlayer` in that bubble is updated with the new URL.
5. If shadowing was active for that turn, the reference audio updates automatically.

### Per-conversation regeneration

- A **[↺ Regen All Audio]** button in the conversation header calls `POST /api/conversations/{id}/regenerate-audio`.
- Returns a stream of per-turn results (can reuse SSE pattern or simple JSON array).
- A progress bar shows `n/total` turns regenerated.

---

## 10. Migration Path

### What to delete

| File | Action |
|------|--------|
| `src/kaiwacoach/ui/gradio_app.py` | Delete after Phase 1 complete |
| `src/kaiwacoach/app.py` (Gradio launch) | Replace with FastAPI launcher |

### What to keep unchanged

Everything under: `orchestrator.py`, `models/`, `storage/`, `prompts/`, `textnorm/`, `settings.py`, `constants.py`

### What to add

| Addition | Notes |
|----------|-------|
| `src/kaiwacoach/api/` | FastAPI app, routes, schemas |
| `frontend/` | Svelte app |
| `pyproject.toml` deps | Add `fastapi`, `uvicorn[standard]`, `python-multipart` |

### `app.py` replacement

```python
# src/kaiwacoach/app.py (new)
from kaiwacoach.api.server import create_app, run

if __name__ == "__main__":
    run()
```

### CLAUDE.md update required

Update the module boundary table to add:

| Module | Responsibility |
|--------|---------------|
| `src/kaiwacoach/api/server.py` | FastAPI app factory, Uvicorn launch, static file serving |
| `src/kaiwacoach/api/routes/` | HTTP routing only — calls orchestrator, returns schemas |
| `src/kaiwacoach/api/schemas/` | Pydantic request/response types for the API layer (distinct from ML model schemas) |
| `frontend/src/` | Svelte UI — no model logic, communicates only via API |

---

## 11. Implementation Phases

Each phase is independently shippable and runnable.

---

### Phase 1 — FastAPI skeleton + static serving (no frontend yet)

**Goal:** Replace Gradio launcher with FastAPI. Existing JSON API testable via `curl`. No UI.

**Tasks:**
- Add `fastapi`, `uvicorn[standard]`, `python-multipart` to `pyproject.toml`
- Create `src/kaiwacoach/api/server.py` — FastAPI app, static file mount (empty dir initially), Uvicorn launch
- Create `src/kaiwacoach/api/deps.py` — orchestrator singleton (lazy init, thread-safe), settings injection
- Create `src/kaiwacoach/api/schemas/` — `ConversationSummary`, `ConversationDetail`, `TurnPair`, `TurnTextRequest`, `TurnAudioRequest`, `CorrectionResponse`
- Create `src/kaiwacoach/api/routes/conversations.py` — list, get, create, rename, delete, delete-all
- Create `src/kaiwacoach/api/routes/audio.py` — serve audio files from session cache
- Replace `src/kaiwacoach/app.py` to call `api/server.py`
- Keep `gradio_app.py` in place but unused

**Done when:** `poetry run python -m kaiwacoach.app` starts on port 8000; `GET /api/conversations` returns JSON.

---

### Phase 2 — SSE turn streaming

**Goal:** Text and audio turn submission work end-to-end via API with progress streaming.

**Tasks:**
- Create `src/kaiwacoach/api/routes/turns.py`:
  - `POST /api/turns/text` — async generator wrapping `orchestrator.run_text_turn()`, emitting SSE stage events
  - `POST /api/turns/audio` — receive `multipart/form-data`, convert WebM→WAV, then stream as above
- Create `src/kaiwacoach/api/routes/stream.py` — SSE helper (`EventSourceResponse` from `sse-starlette`)
- Add `sse-starlette` to deps
- Add WebM→WAV conversion utility in `api/` (using existing `soundfile`/`scipy`)
- Write integration tests for SSE event sequence (text turn + audio turn)

**Done when:** `curl -N -X POST /api/turns/text` emits stage events and a complete event with audio URL.

---

### Phase 3 — Svelte app scaffold + theme system

**Goal:** Frontend skeleton served by FastAPI. Language selector changes colour theme. No real functionality yet.

**Tasks:**
- Initialise `frontend/` with `npm create svelte@latest` or `npm create vite@latest` (Svelte + TS template)
- Configure Vite to output to `frontend/dist/`; FastAPI mounts `frontend/dist` as static
- Configure Vite dev proxy (`/api` → `localhost:8000`) for hot reload during development
- Create `styles/themes.css` with all 6 language themes
- Create `lib/stores/session.ts`, `ui.ts`, `theme.ts`
- Create `LanguageSelector.svelte` — dropdown, updates `sessionStore.language`, `themeStore` sets `data-language`
- Create `App.svelte` shell — sidebar placeholder, main panel placeholder
- Create `lib/types/api.ts` — TypeScript interfaces matching API schemas

**Done when:** `npm run dev` shows a themed page; changing language changes colour scheme.

---

### Phase 4 — Conversation history sidebar

**Goal:** Full conversation list, load, delete, new conversation.

**Tasks:**
- Create `lib/api/conversations.ts` — typed fetch wrappers
- Create `ConversationList.svelte`, `ConversationItem.svelte`
- Create `Sidebar.svelte` — list + New Conversation button
- Load conversation into `sessionStore` (populates `turns` array with historical turns)
- Create `ConfirmDialog.svelte` for delete confirmations
- Loading and error states throughout

**Done when:** Can browse, load, and delete conversations. History reloads correctly including corrections.

---

### Phase 5 — Chat thread + text input

**Goal:** Text turns work end-to-end in the new UI.

**Tasks:**
- Create `lib/api/client.ts` — fetch-based SSE reader (POST support)
- Create `lib/api/turns.ts` — `submitTextTurn()`, `submitAudioTurn()`
- Create `ChatThread.svelte` — renders `TurnPair` list, auto-scrolls
- Create `TurnPair.svelte`, `UserBubble.svelte`, `AssistantBubble.svelte`
- Create `CorrectionCard.svelte` — collapsible card with errors, corrected, native, explanation
- Create `PipelineProgress.svelte` — animated stage indicators (subscribes to `uiStore`)
- Create `InputArea.svelte` — text input + Send button + corrections toggle
- Wire SSE: on each `stage` event update `uiStore`; on `complete` append turn to `sessionStore.turns`

**Done when:** Can have a text conversation; corrections appear; pipeline progress animates.

---

### Phase 6 — Audio recording + playback

**Goal:** Audio turns work; WaveSurfer playback for both user and assistant audio.

**Tasks:**
- Install WaveSurfer.js: `npm install wavesurfer.js`
- Create `AudioPlayer.svelte` — WaveSurfer instance, `src` prop, play/pause controls, full waveform, language-reactive colours
- Create `AudioRecorder.svelte` — WaveSurfer Record plugin, live waveform during recording, emits blob on stop
- Integrate `AudioRecorder` into `InputArea.svelte` — audio submit path calls `submitAudioTurn()`
- Add user audio playback in `UserBubble.svelte` (if `has_user_audio`)
- Add assistant audio playback in `AssistantBubble.svelte` (autoplay option)

**Done when:** Full audio turn works; waveforms display correctly for both user and assistant audio.

---

### Phase 7 — Audio regeneration

**Goal:** Per-turn and per-conversation audio regeneration.

**Tasks:**
- Create `lib/api/audio.ts` — `regenerateTurnAudio()`, `regenerateConversationAudio()`
- Add [↺] button to `AssistantBubble.svelte` — calls API, updates audio URL in turn, re-inits AudioPlayer
- Add [↺ Regen All] button to conversation header
- Progress feedback for conversation-level regen

**Done when:** Clicking regen produces new audio and the waveform updates.

---

### Phase 8 — Shadowing mode

**Goal:** Side-by-side listen + record comparison for any assistant turn.

**Tasks:**
- Create `ShadowingPanel.svelte` — reference `AudioPlayer` + `AudioRecorder` side by side
- Add [Shadow] button to `AssistantBubble.svelte`
- Wire `uiStore.shadowingTurnId` to show/hide `ShadowingPanel`
- Handle multiple attempts (Try Again clears recorder state)
- Ensure `ShadowingPanel` works correctly when reference audio was just regenerated

**Done when:** User can shadow any assistant turn with real-time waveform comparison.

---

### Phase 9 — Polish + cleanup

**Goal:** Production-quality feel. Remove Gradio.

**Tasks:**
- Delete `src/kaiwacoach/ui/gradio_app.py`
- Remove `gradio` from `pyproject.toml` deps
- Error states: network errors, pipeline failures, ASR failure, missing audio
- Empty states: no conversations yet, no corrections
- Keyboard shortcuts: Enter to send text, Esc to close shadowing panel
- Update `CLAUDE.md` module boundaries
- Update `README.md` with new setup/run instructions
- Final manual smoke test: text turn, audio turn, load history, shadowing, regen

---

## 12. Testing Strategy

### Existing tests — unaffected

All tests under `tests/` that don't import from `kaiwacoach.ui.gradio_app` are unaffected:
- `test_json_enforcement.py`, `test_prompt_schemas.py`, `test_prompt_loader.py`
- `test_invariants.py`, `test_jp_katakana.py`, `test_jp_normalisation_golden.py`
- `test_jp_tts_normalisation.py`, `test_protected_spans.py`
- `test_storage.py` (db + blobs)
- `tests/test_app_startup.py` — **needs updating** once `app.py` changes

### Tests to add

**Phase 1–2 (FastAPI/SSE):**
- `tests/test_api_conversations.py` — CRUD endpoints with a test DB
- `tests/test_api_turns.py` — SSE event sequence for text and audio turns (mock orchestrator)
- `tests/test_audio_conversion.py` — WebM→WAV conversion utility

**Phase 5–6 (frontend integration):**
- Manual smoke tests documented in `docs/feature_demos.md`:
  - Text turn end-to-end
  - Audio turn end-to-end
  - Load conversation with corrections
  - Shadowing flow
  - Audio regeneration

**Regression:**
- Any existing UI callback tests (`test_app_startup.py`, others importing `gradio_app`) → delete or rewrite as API tests.

### Test isolation

FastAPI tests use `httpx.AsyncClient` with `TestClient` from `fastapi.testclient`. Orchestrator is injected via `deps.py` — in tests, override the dependency to use a mock orchestrator or a real orchestrator with a temp DB.

```python
# tests/conftest.py (new)
from fastapi.testclient import TestClient
from kaiwacoach.api.server import create_app

@pytest.fixture
def client(tmp_path):
    app = create_app(storage_root=tmp_path)
    return TestClient(app)
```

---

## Appendix A — Orchestrator Integration Reference

> This appendix exists because the main plan omits critical integration details that would block a new implementer. Read this before writing any `api/routes/` code.

---

### A.1 Correction: actual method names

The plan body refers to `run_text_turn()` and `run_audio_turn()`. **These methods do not exist.** The correct names are:

| Plan body (wrong) | Actual method |
|---|---|
| `orchestrator.run_text_turn()` | `orchestrator.process_text_turn()` |
| `orchestrator.run_audio_turn()` | `orchestrator.process_audio_turn()` |

---

### A.2 Verified orchestrator public API

All signatures confirmed by reading `src/kaiwacoach/orchestrator.py`.

```python
class ConversationOrchestrator:

    def __init__(
        self,
        db: SQLiteWriter,
        llm: LLMProtocol,
        prompt_loader: PromptLoader,
        language: str,
        tts: TTSProtocol | None = None,
        tts_voice: str | None = None,
        tts_speed: float = 1.0,
        asr: ASRProtocol | None = None,
        audio_cache: SessionAudioCache | None = None,
        timing_logs_enabled: bool = True,
    ) -> None: ...

    # ── Turn methods ─────────────────────────────────────────────────────

    def process_text_turn(
        self,
        conversation_id: str,
        user_text: str,
        conversation_history: str = "",
    ) -> TextTurnResult: ...
    # Always runs: LLM reply → corrections → TTS (see A.5 for corrections toggle)

    def process_audio_turn(
        self,
        conversation_id: str,
        pcm_bytes: bytes,          # Raw PCM, NOT a file path, NOT WebM/WAV bytes
        audio_meta: AudioMeta,     # See A.7 for conversion details
        conversation_history: str = "",
    ) -> AudioTurnResult: ...

    # ── Read methods ──────────────────────────────────────────────────────

    def list_conversations(self) -> list[dict[str, Any]]:
        # Returns: [{"id", "title", "language", "updated_at", "preview_text"}, ...]
        # NOTE: does NOT include turn_count — add a separate SQL query if needed
        ...

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        # Returns:
        # {
        #   "id": str, "title": str, "language": str,
        #   "created_at": str, "updated_at": str,
        #   "turns": [
        #     {
        #       "user_turn_id": str,
        #       "input_text": str | None,   # typed input; None for audio turns
        #       "asr_text": str | None,     # ASR transcript; None for text turns
        #       "assistant_turn_id": str,
        #       "reply_text": str,
        #     }, ...
        #   ]
        # }
        # IMPORTANT: does NOT include corrections — see A.3
        ...

    def get_latest_corrections(self, user_turn_id: str) -> dict[str, Any]:
        # Returns: {"errors": list[str], "corrected": str, "native": str, "explanation": str}
        # Returns empty strings/lists (not None) when no corrections exist
        ...

    # ── Write / lifecycle methods ─────────────────────────────────────────

    def create_conversation(self, title: str | None = None) -> str:
        # Returns conversation_id (UUID str). Title defaults to None.
        # Conversation gets the orchestrator's current language stamped on it.
        ...

    def delete_conversation(self, conversation_id: str) -> None: ...

    def delete_all_conversations(self) -> None: ...

    def set_language(self, language: str) -> None:
        # Updates self._language AND syncs the ASR model's language.
        # Call this when the user changes the language selector.
        ...

    def reset_session(self) -> None:
        # Clears LLM and TTS caches. Does not delete DB rows.
        ...

    # ── Audio regeneration ────────────────────────────────────────────────

    def regenerate_turn_audio(self, assistant_turn_id: str) -> TTSResult:
        # Fetches reply_text from DB, re-runs TTS with self._language.
        # Returns TTSResult (has .audio_path: str | None).
        ...

    def regenerate_conversation_audio(self, conversation_id: str) -> list[TTSResult]:
        # Regenerates TTS for every assistant turn in the conversation.
        # Returns list of TTSResult in turn order.
        ...

    # ── Corrections (public, also called internally by process_*_turn) ───

    def run_corrections(
        self,
        user_turn_id: str,
        user_text: str,
        assistant_turn_id: str | None = None,
        timings: dict | None = None,
    ) -> dict[str, Any]:
        # Returns {"errors", "corrected", "native", "explanation"}
        # Persists to corrections table automatically.
        ...

    def finalize_and_log_timings(self, label: str, timings: dict[str, float]) -> None: ...

    @property
    def language(self) -> str: ...

    @property
    def expected_sample_rate(self) -> int | None:
        # Returns the sample rate the ASR model expects (e.g. 16000).
        # Use this as the target sample rate when converting browser audio.
        ...
```

**Result types** (from top of `orchestrator.py`):

```python
@dataclass(frozen=True)
class TextTurnResult:
    conversation_id: str
    user_turn_id: str
    assistant_turn_id: str
    reply_text: str
    tts_audio_path: str | None = None

@dataclass(frozen=True)
class AudioTurnResult:
    conversation_id: str
    user_turn_id: str
    assistant_turn_id: str
    reply_text: str
    input_audio_path: str      # path to cached user audio WAV
    asr_text: str
    asr_meta: dict[str, Any]
    tts_audio_path: str | None = None
```

---

### A.3 The corrections-reload bug — root cause and fix

**Root cause:** `get_conversation()` does **not** include corrections. It returns turns with `user_turn_id`, `input_text`, `asr_text`, `assistant_turn_id`, `reply_text` — but no correction data. The old Gradio `_load_conversation()` never called `get_latest_corrections()`, so corrections were always blank on load.

**Fix in the API layer:** The `GET /api/conversations/{id}` route must enrich each turn with corrections:

```python
# routes/conversations.py
@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, orc: ConversationOrchestrator = Depends(get_orchestrator)):
    convo = orc.get_conversation(conversation_id)
    for turn in convo["turns"]:
        turn["correction"] = orc.get_latest_corrections(turn["user_turn_id"])
    return convo
```

This is why the `GET /api/conversations/{id}` response schema in Section 3.1 includes a `correction` object — it is assembled here by the route handler, not returned directly by `get_conversation()`.

---

### A.4 Language is a singleton on the orchestrator

`language` is stored as `self._language` on the orchestrator instance. It is **not** a per-call parameter on `process_text_turn()` or `process_audio_turn()`.

When the user changes the language selector in the frontend:
1. Frontend calls `POST /api/session/language` with `{"language": "fr"}`
2. Route handler calls `orchestrator.set_language("fr")`
3. `set_language()` updates `self._language` **and** syncs the ASR model's language

**Important:** `create_conversation()` stamps the orchestrator's current language onto the new conversation row. Always call `set_language()` before `create_conversation()` if the language differs from the current orchestrator state.

When **loading** a historical conversation with a different language from the current session, call `set_language()` with the loaded conversation's language to restore the correct context.

---

### A.5 The `corrections_enabled` toggle

The `corrections_enabled` checkbox in the old UI was handled entirely in the Gradio callback layer — it checked the flag and simply did not call `orchestrator.run_corrections()` if disabled. The orchestrator's `process_text_turn()` and `process_audio_turn()` always run corrections internally.

**Required small change to orchestrator** (backward-compatible):

```python
def process_text_turn(
    self,
    conversation_id: str,
    user_text: str,
    conversation_history: str = "",
    corrections_enabled: bool = True,   # ← add this
) -> TextTurnResult:
    ...
    if corrections_enabled:
        corrections = self.run_corrections(...)
    ...
```

Add the same parameter to `process_audio_turn()`. Existing callers with no keyword argument continue to work unchanged. Add a regression test.

---

### A.6 SSE streaming architecture — the async bridge problem

`process_text_turn()` is **blocking synchronous** code (it loads models, calls LLM, writes to SQLite). FastAPI is **async**. You cannot call it directly in an async route handler without blocking the entire event loop.

Additionally, `process_text_turn()` is monolithic — it does not yield between stages, so you cannot emit intermediate SSE events (`stage: llm running`, `stage: corrections running`, etc.) unless the orchestrator accepts a callback.

**Recommended approach: thread executor + asyncio.Queue + stage callbacks**

This is the minimum change that achieves true per-stage SSE streaming:

**Step 1 — Add an `on_stage` callback parameter to the orchestrator** (backward-compatible; defaults to `None`):

```python
# In orchestrator.py — add to process_text_turn and process_audio_turn:
from typing import Callable

def process_text_turn(
    self,
    conversation_id: str,
    user_text: str,
    conversation_history: str = "",
    corrections_enabled: bool = True,
    on_stage: Callable[[str, str, dict], None] | None = None,
) -> TextTurnResult:
    # At the start of each major step, call:
    #   on_stage("llm", "running", {})
    # At the end of each step:
    #   on_stage("llm", "complete", {"reply": reply_text})
    ...
```

Emit `on_stage` calls at these points in the pipeline:
- Before/after `generate_reply()` → stage `"llm"`
- Before/after `run_corrections()` (if enabled) → stage `"corrections"`
- Before/after `run_tts()` → stage `"tts"`
- For audio turns, before/after ASR → stage `"asr"`

**Step 2 — FastAPI SSE route using thread pool + asyncio.Queue:**

```python
# routes/turns.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sse_starlette.sse import EventSourceResponse

_executor = ThreadPoolExecutor(max_workers=1)  # one turn at a time

@router.post("/turns/text")
async def submit_text_turn(request: TurnTextRequest, orc = Depends(get_orchestrator)):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_stage(stage: str, status: str, data: dict):
        # Called from background thread — must use thread-safe queue put
        event = {"stage": stage, "status": status, **data}
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_sync():
        try:
            result = orc.process_text_turn(
                conversation_id=request.conversation_id,
                user_text=request.text,
                conversation_history=request.conversation_history,
                corrections_enabled=request.corrections_enabled,
                on_stage=on_stage,
            )
            loop.call_soon_threadsafe(queue.put_nowait, {"_type": "complete", "result": result})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"_type": "error", "message": str(e)})

    loop.run_in_executor(_executor, run_sync)

    async def event_generator():
        while True:
            event = await queue.get()
            if event.get("_type") == "complete":
                result = event["result"]
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "conversation_id": result.conversation_id,
                        "user_turn_id": result.user_turn_id,
                        "assistant_turn_id": result.assistant_turn_id,
                    })
                }
                break
            elif event.get("_type") == "error":
                yield {"event": "error", "data": json.dumps({"message": event["message"]})}
                break
            else:
                yield {"event": "stage", "data": json.dumps(event)}

    return EventSourceResponse(event_generator())
```

**Important notes:**
- Use `max_workers=1` — the orchestrator holds ML models in memory and is not designed for concurrent turn processing.
- `loop.call_soon_threadsafe()` is required to safely put items on the asyncio queue from a background thread.
- `sse-starlette` must be added to `pyproject.toml`: `sse-starlette = ">=1.6"`

---

### A.7 WebM → raw PCM conversion

`process_audio_turn()` takes `pcm_bytes: bytes` (raw, uncompressed PCM samples) and an `AudioMeta` dataclass — **not** a WAV file and not WebM. `soundfile` and `scipy` cannot decode WebM/Opus.

**`AudioMeta` definition** (from `storage/blobs.py`):

```python
@dataclass(frozen=True)
class AudioMeta:
    sample_rate: int
    channels: int
    sample_width: int          # bytes per sample: 2 for int16
    num_frames: int | None = None
    duration_seconds: float | None = None
```

**Recommended conversion: ffmpeg via subprocess** (ffmpeg is standard on macOS via Homebrew; document as a setup prerequisite):

```python
# api/audio_conversion.py
import subprocess, tempfile, wave
from pathlib import Path
from kaiwacoach.storage.blobs import AudioMeta

def webm_to_pcm(webm_bytes: bytes, target_sample_rate: int = 16000) -> tuple[bytes, AudioMeta]:
    """Convert WebM/Opus browser recording to raw PCM + AudioMeta."""
    with tempfile.TemporaryDirectory() as tmp:
        webm_path = Path(tmp) / "input.webm"
        wav_path = Path(tmp) / "output.wav"
        webm_path.write_bytes(webm_bytes)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(webm_path),
                "-ar", str(target_sample_rate),
                "-ac", "1",           # mono
                "-sample_fmt", "s16", # 16-bit signed int
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            pcm_bytes = wf.readframes(num_frames)
        duration = num_frames / sample_rate if sample_rate > 0 else None
    return pcm_bytes, AudioMeta(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        num_frames=num_frames,
        duration_seconds=duration,
    )
```

Use `orchestrator.expected_sample_rate` as the `target_sample_rate` rather than hardcoding 16000, in case the ASR model changes.

Add `ffmpeg` to setup prerequisites in `README.md`: `brew install ffmpeg`.

Add `tests/test_audio_conversion.py` with a synthetic WAV-wrapped test (no real WebM needed for unit testing the WAV reading path).

---

### A.8 Historical audio availability

Session audio (both user recordings and TTS) is **deleted when the app exits** (`SessionAudioCache` cleans up on `__del__` / explicit `cleanup()`). A loaded historical conversation will have `has_user_audio`/`has_assistant_audio` flags indicating audio was generated, but the files no longer exist.

**Design rule for the API and frontend:**

- The API response for `GET /api/conversations/{id}` should include `user_audio_url` and `assistant_audio_url` as actual URLs **only when the file exists on disk**. When missing, return `null`.
- Keep `has_user_audio` / `has_assistant_audio` as booleans meaning "audio was ever generated for this turn" — used to decide whether a Regen button should appear.
- The frontend must only render `AudioPlayer` and the `[Shadow]` button when the audio URL is non-null.
- When `has_assistant_audio` is `true` but `assistant_audio_url` is `null`, show `[↺ Regen]` with a tooltip like "Audio not available — regenerate to play".

```python
# routes/conversations.py — URL resolution
from pathlib import Path

def resolve_audio_url(audio_path: str | None) -> str | None:
    """Return a serveable URL if the file exists, else None."""
    if audio_path is None:
        return None
    if Path(audio_path).exists():
        # Strip the absolute prefix and return as a relative API path
        return f"/api/audio/{audio_path}"
    return None
```

The `GET /api/audio/{path:path}` route must validate that the resolved path is within the session cache root to prevent path traversal (even for a local app, this is good practice):

```python
# routes/audio.py
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

AUDIO_ROOT = Path(settings.storage_root) / "sessions"

@router.get("/audio/{audio_path:path}")
async def serve_audio(audio_path: str):
    full_path = (AUDIO_ROOT / audio_path).resolve()
    if not full_path.is_relative_to(AUDIO_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid audio path")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(full_path, media_type="audio/wav")
```

---

### A.9 `conversation_history` format and assembly

The orchestrator expects `conversation_history` as a plain newline-separated string of alternating `User:` / `Assistant:` prefixed lines. It internally truncates this based on token limits.

**Format:**
```
User: こんにちは
Assistant: こんにちは！元気ですか？
User: はい、元気です
Assistant: それは良かったです！
```

**Assembly from a loaded conversation:**

```python
def build_conversation_history(turns: list[dict]) -> str:
    """Build the conversation_history string from get_conversation() turn dicts."""
    lines = []
    for turn in turns:
        user_text = turn.get("asr_text") or turn.get("input_text") or ""
        reply_text = turn.get("reply_text") or ""
        if user_text:
            lines.append(f"User: {user_text}")
        if reply_text:
            lines.append(f"Assistant: {reply_text}")
    return "\n".join(lines)
```

**Where to call this:** When the frontend loads a conversation and the user then submits a new turn, the `POST /api/turns/text` request must include the `conversation_history` built from all prior turns. The frontend should maintain this as part of its session state (updated after each new turn) rather than fetching it from the server on every submission.

Add `conversation_history` as a field on `TurnTextRequest` and `TurnAudioRequest`. The frontend builds it from `sessionStore.turns`.

---

### A.10 Session cleanup via FastAPI lifespan

The existing app deletes session audio on exit via `SessionAudioCache.__del__` / explicit cleanup. In FastAPI, use the `lifespan` context manager:

```python
# api/server.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing needed (models loaded lazily in deps.py)
    yield
    # Shutdown: clean up session audio cache
    audio_cache = app.state.audio_cache  # set during startup in deps.py
    if audio_cache is not None:
        audio_cache.cleanup()

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.mount("/api", api_router)
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
    return app
```

---

### A.11 All 6 language themes (verified from `gradio_app.py`)

The complete colour schemes to port to `styles/themes.css`. These were the values used in the Gradio `_LANGUAGE_THEMES` dict and should be replicated exactly to preserve the visual identity.

```css
:root[data-language="ja"] {
  --kc-primary: #c0392b;
  --kc-primary-light: #e74c3c;
  --kc-user-bubble: #fdf2f2;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #c0392b;
  --kc-waveform-active: #c0392b;
  --kc-waveform-inactive: #e0b0b0;
  --kc-correction-border: #e74c3c;
}

:root[data-language="fr"] {
  --kc-primary: #1565c0;
  --kc-primary-light: #1976d2;
  --kc-user-bubble: #f0f4ff;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #1565c0;
  --kc-waveform-active: #1565c0;
  --kc-waveform-inactive: #90b0d8;
  --kc-correction-border: #1976d2;
}

:root[data-language="en"] {
  --kc-primary: #1a5e20;
  --kc-primary-light: #2e7d32;
  --kc-user-bubble: #f1f8f1;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #1a5e20;
  --kc-waveform-active: #2e7d32;
  --kc-waveform-inactive: #a5c8a5;
  --kc-correction-border: #2e7d32;
}

:root[data-language="es"] {
  --kc-primary: #e65100;
  --kc-primary-light: #f57c00;
  --kc-user-bubble: #fff8f0;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #e65100;
  --kc-waveform-active: #f57c00;
  --kc-waveform-inactive: #f0c080;
  --kc-correction-border: #f57c00;
}

:root[data-language="it"] {
  --kc-primary: #006064;
  --kc-primary-light: #00838f;
  --kc-user-bubble: #f0fafa;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #006064;
  --kc-waveform-active: #00838f;
  --kc-waveform-inactive: #80c8cc;
  --kc-correction-border: #00838f;
}

:root[data-language="pt"] {
  --kc-primary: #4a148c;
  --kc-primary-light: #6a1b9a;
  --kc-user-bubble: #f8f0ff;
  --kc-bot-bubble: #f9f9f9;
  --kc-accent: #4a148c;
  --kc-waveform-active: #6a1b9a;
  --kc-waveform-inactive: #c090d8;
  --kc-correction-border: #6a1b9a;
}
```

> **Note:** Cross-check these values against `gradio_app.py`'s `_LANGUAGE_THEMES` dict before deleting that file. The values above are the intended palette; if the Gradio app has diverged, the Gradio app is the source of truth until it is deleted.

**Language codes** are defined in `src/kaiwacoach/constants.py` as `SUPPORTED_LANGUAGES`. Always import from there rather than hardcoding the list — it is the single source of truth.

---

### A.12 Svelte version

Specify **Svelte 4** when scaffolding. Svelte 5 (current default from `npm create svelte@latest`) uses a different reactivity model ("runes": `$state`, `$derived`, `$effect`) that is incompatible with the `writable`/`derived` store patterns used throughout this plan.

```bash
# Scaffold with Svelte 4 explicitly:
npm create vite@latest frontend -- --template svelte-ts
# Then in frontend/, pin Svelte 4:
npm install svelte@^4
```

If you prefer Svelte 5 runes, the store patterns in Section 5 must be rewritten — `writable` → `$state`, `derived` → `$derived`, `subscribe` side effects → `$effect`. Either is fine; pick one and be consistent.

---

### A.13 Development workflow

**Day-to-day (two terminals):**

```bash
# Terminal 1 — FastAPI backend (hot-reloads Python with --reload)
poetry run uvicorn kaiwacoach.api.server:app --reload --port 8000

# Terminal 2 — Vite dev server (hot module replacement for Svelte)
cd frontend && npm run dev
# Vite dev server listens on http://localhost:5173
# All /api/* requests are proxied to http://localhost:8000
```

**`frontend/vite.config.ts`** (required for the proxy):

```typescript
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
```

**Production (single process):**

```bash
cd frontend && npm run build   # outputs to frontend/dist/
poetry run python -m kaiwacoach.app  # FastAPI serves frontend/dist at /
```

**Run command** stays unchanged per `CLAUDE.md`: `poetry run python -m kaiwacoach.app`

---

### A.14 Required dependency additions

These must be added to `pyproject.toml` under `[tool.poetry.dependencies]`:

```toml
fastapi = ">=0.111"
uvicorn = {extras = ["standard"], version = ">=0.29"}
python-multipart = ">=0.0.9"    # required for UploadFile / form data
sse-starlette = ">=1.6"         # SSE EventSourceResponse
httpx = ">=0.27"                # for FastAPI test client (dev dependency)
```

`gradio` can be removed from `pyproject.toml` once Phase 9 is complete and all Gradio tests have been deleted or rewritten.

External system dependency (not a Python package):
- `ffmpeg` — install via `brew install ffmpeg`. Add to `scripts/setup_macos.sh` and `README.md`.
