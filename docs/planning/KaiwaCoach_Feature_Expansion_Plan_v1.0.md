# KaiwaCoach – Feature Expansion Plan

## Purpose of this document

This document defines the full scope of a planned feature expansion for KaiwaCoach. It is written to be self-contained: a developer or coding agent with no prior context should be able to read it and understand what to build, how to build it, and how to know when each feature is complete.

No code should be written until this document has been reviewed and approved.

---

## 1. Application Context

KaiwaCoach is an **offline-first language coaching app** for Apple Silicon macOS. It runs a local conversational loop using local ASR (speech-to-text), LLM (language model), and TTS (text-to-speech) models via MLX or Ollama. The backend is a **FastAPI** server; the frontend is a **Svelte 4 SPA** communicating via REST and Server-Sent Events (SSE).

### Key architectural rules (must not be violated)

- **No network calls at runtime.** All model inference is local.
- **Prompts only in `src/kaiwacoach/prompts/*.md`.** Never inline prompt text in Python.
- **All LLM outputs schema-validated** using Pydantic, one repair retry max, then fail safe.
- **Persist before side effects**: save to DB before TTS or other irreversible steps.
- **Single-writer SQLite queue**: all DB writes go through `SQLiteWriter`; reads use short-lived separate connections.
- **Bounded in-memory caches**: any `dict`-backed cache must be bounded.
- **Narrow exception handling**: never swallow errors with bare `except Exception`.
- **Module boundaries**: UI ↔ API ↔ Orchestrator ↔ Models/Storage. No cross-layer imports. The orchestrator must not import from `src/kaiwacoach/api/`.

### Relevant existing patterns

**Orchestrator language scoping**:  
`ConversationOrchestrator.__init__` takes `language: str` and stores it as `self._language`. It is a per-session singleton. All language-dependent operations (TTS normalisation, voice routing, `_user_level_for`) use `self._language`. New orchestrator methods **must not accept a `language` parameter** — they use `self._language` internally. If the frontend wants narration or monologue in a specific language, it must match the current session language, which is already enforced by the app's single-language-per-session design.

**SSE streaming** (used by turn routes):  
`_build_sse_generator()` is defined in `src/kaiwacoach/api/routes/turns.py` (not in utils.py). ML work runs in `_ML_EXECUTOR` (a single-threaded `ThreadPoolExecutor` defined in `src/kaiwacoach/api/utils.py`). Events are emitted as `{"event": "stage"|"complete"|"error", "data": {...}}`. Every error event must include a `request_id`.

**Important**: `_build_sse_generator` in `turns.py` has TTS-specific logic hardcoded into its `on_stage` callback (when `stage == "tts"` and `status == "complete"`, it converts the audio path to a URL). Monologue has no TTS stage, so the new monologue routes must **implement their own SSE generator** rather than reusing this function. Follow the same structural pattern but without the TTS path conversion.

**Prompt rendering**:  
`PromptLoader.render(name, variables)` in `src/kaiwacoach/prompts/loader.py`. Takes a dict of `{key: value}` substitutions and returns a `PromptRenderResult(text, sha256, path)`. Missing keys raise `KeyError`. Passing `None` as a value renders the literal string `"None"` — callers must convert `None` to a sensible string before passing. SHA256 is stored with each LLM call.

**LLM role calling**:  
`self._llm.generate_json(prompt, role)` dispatches to the correct wrapper. Each role has a named token cap in `LLMRoleCaps` (a frozen dataclass in `src/kaiwacoach/settings.py`). Adding a new role cap requires:

1. Adding the field to `LLMRoleCaps` with a default value
2. Updating `load_config()` to parse it from the merged config dict
3. Adding it to `AppConfig.to_dict()` if that method exists
4. Adding an env var mapping to `_apply_env_overrides` following the pattern `KAIWACOACH_LLM_ROLE_<ROLENAME>_MAX_NEW_TOKENS`

Each role also has a Pydantic schema in `src/kaiwacoach/models/json_enforcement.py`, registered in `ROLE_SCHEMAS`.

**Pydantic list schemas**:  
Existing schemas use `conlist(StrictStr, min_length=1)`. This is deprecated in Pydantic v2 (the preferred form is `Annotated[list[StrictStr], Field(min_length=1)]`). New schemas should match the existing `conlist` pattern for consistency. Do not mix styles within the file.

**`audio_path_to_url()`**:  
This function lives in `src/kaiwacoach/api/utils.py` and belongs to the API layer. Orchestrator methods must **not** call it. Orchestrator methods return raw audio paths; the route handler calls `audio_path_to_url(path, cache_root)` to produce the URL. The route handler needs access to `request.app.state.audio_cache.root_dir` for the cache root — see `regen.py` and `turns.py` for how this state is accessed.

**Testing patterns**:

- Use `MagicMock()` for orchestrator and model objects in API tests
- Use a real SQLite DB (`_setup_db(tmp_path)`) in orchestrator tests
- Parse SSE responses with the local `parse_sse(text)` helper (defined in the existing turn/regen test files)
- All SSE route tests must include a mid-stream failure case

---

## 2. What We Are Building

Four new features, plus shared navigation infrastructure:

| #   | Feature              | Summary                                                                         |
| --- | -------------------- | ------------------------------------------------------------------------------- |
| 0   | Tab bar navigation   | Prerequisite: introduces Chat / Monologue / Narration tabs + Settings gear icon |
| 1   | User Settings        | Name + per-language proficiency level; feeds into all LLM prompts               |
| 2   | Narration tab        | Paste text → TTS → preview + download; no LLM, stateless                        |
| 3   | Monologue mode       | Submit text/audio → corrections + improvement summary; persisted to DB          |
| 4   | Conversation Summary | On-demand summary of error patterns across a conversation; ephemeral            |

### Confirmed design decisions (do not relitigate)

- Navigation: horizontal tab strip below the top bar; Chat (default) · Monologue · Narration; gear icon opens Settings side panel; sidebar is **hidden** in the Narration tab
- Proficiency level affects **all LLM outputs**: conversation reply, correction detection, correction explanations; level is read fresh from DB on every turn
- Level defaults: **N5** for Japanese overall and kanji, **A1** for all other languages (CEFR); stored per-language so multilingual users have independent levels
- Japanese has **two independent level dimensions**: overall grammar/vocabulary (`ja`) and kanji reading (`ja_kanji`), both using the same scale: N5 → N4 → N3 → N2 → N1 → Native. "Native" represents the level of an educated Japanese adult — above JLPT N1 in both kanji breadth (~3000+ characters) and vocabulary. These two dimensions can differ (e.g. N3 grammar, N1 kanji).
- Settings changes take effect on the **next turn** in any conversation (no conversation restart required)
- Monologue sessions are **persisted to SQLite** (`conversation_type = 'monologue'`), appear in the sidebar under the Monologue tab; each session is **read-only** once submitted
- Monologue inputs: **text + mic + file upload** (all three)
- Monologue analysis pipeline: reuse existing `detect_and_correct` → `explain_and_native` pipeline, then add one new `monologue_summary` LLM call
- Narration: **one-shot**, session audio cache only, no DB record; uses session language (self.\_language); voice/speed from config defaults
- Conversation Summary: **ephemeral** (not persisted); displayed as a **collapsible inline panel** above the chat thread; capped at **20 most recent corrections** to stay within token budget
- Empty chat state (no conversation selected): **no change** from current behaviour

---

## 3. Build Order and Rationale

Build in this order — each step unblocks or improves what follows:

1. **Tab bar navigation** — frontend-only. Required before any new tab content can be built.
2. **User Settings** — backend + frontend. Per-language levels feed into prompts for Monologue and Conversation Summary; build this early so the data exists.
3. **Narration tab** — independent and low-risk. Good to ship early; no LLM, no schema changes.
4. **Monologue mode** — requires schema change (`conversation_type`), new LLM role, SSE route, file upload. Most complex feature.
5. **Conversation Summary** — reads existing corrections data and user level; one new LLM role; simpler once everything else is in place.

---

## 4. Feature 0: Tab Bar Navigation (Prerequisite)

### Goal

Introduce a horizontal tab bar below the top bar so the app can host Chat, Monologue, and Narration as peer views. Add a gear icon to open the Settings panel.

This step only changes the frontend. No backend changes.

### Changes

**`frontend/src/lib/stores/ui.ts`**  
Add:

```typescript
activeTab: "chat" | "monologue" | "narration"; // default: 'chat'
settingsOpen: boolean; // default: false
```

**New component: `frontend/src/components/TabBar.svelte`**  
A horizontal strip with three tab buttons: Chat · Monologue · Narration.  
Active tab is driven by `$uiStore.activeTab`.  
Clicking a tab updates the store.

**`frontend/src/App.svelte`**

- Add gear icon (⚙) in the top bar, left of the language selector. Click toggles `$uiStore.settingsOpen`.
- Render `<TabBar>` directly below the top bar.
- Conditionally render main panel content based on `$uiStore.activeTab`:
    - `'chat'` → current `<ChatThread>` + `<InputArea>` layout (no change)
    - `'monologue'` → `<MonologuePanel>` (stub initially)
    - `'narration'` → `<NarrationPanel>` (stub initially), sidebar hidden
- Render `<SettingsPanel>` when `$uiStore.settingsOpen` (stub initially, as overlay with same z-index pattern as `ShadowingPanel`)
- Extend the existing Esc key handler to also close the settings panel.

**Sidebar visibility (`frontend/src/components/Sidebar.svelte`)**:  
Sidebar is hidden when `$uiStore.activeTab === 'narration'`, giving the narration panel full width. Pass the activeTab down so Sidebar can either hide itself or be conditionally rendered by App.svelte.

### Definition of Done

- [x] Tab bar renders; clicking each tab updates the active tab without errors
- [x] Gear icon in top bar opens/closes a placeholder settings panel overlay
- [x] Esc key closes the settings overlay
- [x] Narration tab hides the sidebar and expands the main panel to full width
- [x] Switching tabs does not break the existing Chat flow (conversation list, chat, audio still work)
- [x] No TypeScript errors; `npm run build` succeeds

---

## 5. Feature 1: User Settings (Name + Per-Language Proficiency Levels)

### Goal

Store the user's display name and their proficiency level for each language. Inject these into every LLM prompt so responses and corrections are pitched at the appropriate level.

### Backend

#### Schema change: `src/kaiwacoach/storage/schema.sql`

Add a **singleton** user profile table after the `conversations` table definition:

```sql
CREATE TABLE IF NOT EXISTS user_profile (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  user_name TEXT,
  language_proficiency_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO user_profile (id) VALUES (1);
```

`language_proficiency_json` stores a JSON object keyed by language code:  
`{"ja": "N4", "fr": "B1", "en": "B2"}`. Missing keys mean the user is at the default (N5 / A1).

Bump `schema_version` to **v2**.

**Note on DB migration**: `SQLiteWriter.start()` re-applies the full `schema.sql` via `executescript` on every startup. The `CREATE TABLE IF NOT EXISTS` clause means the `user_profile` table will be created automatically on any existing v1 database without triggering a full reset. The `_schema_needs_reset()` check (which looks for required columns on the `conversations` table) does not need to be updated for this addition.

#### `src/kaiwacoach/storage/db.py`

- Add `"user_profile": {"user_name", "language_proficiency_json", "updated_at"}` to `_ALLOWED_UPDATE_COLUMNS`

#### Level validation constants

Add a `VALID_PROFICIENCY_LEVELS` dict in `src/kaiwacoach/constants.py` (where `SUPPORTED_LANGUAGES` already lives — the orchestrator already imports from this file):

```python
_JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1", "Native"]

VALID_PROFICIENCY_LEVELS: dict[str, list[str]] = {
    "ja":       _JLPT_LEVELS,   # overall grammar / vocabulary level
    "ja_kanji": _JLPT_LEVELS,   # kanji reading level, independent of grammar
    "fr": ["A1", "A2", "B1", "B2", "C1", "C2", "Native"],
    "en": ["A1", "A2", "B1", "B2", "C1", "C2", "Native"],
    "es": ["A1", "A2", "B1", "B2", "C1", "C2", "Native"],
    "it": ["A1", "A2", "B1", "B2", "C1", "C2", "Native"],
    "pt-br": ["A1", "A2", "B1", "B2", "C1", "C2", "Native"],
}
```

`"ja_kanji"` is treated as a valid proficiency key by the validation logic, even though it is not in `SUPPORTED_LANGUAGES`. The validation loop must check against `VALID_PROFICIENCY_LEVELS` keys, not `SUPPORTED_LANGUAGES` keys.

````

#### `src/kaiwacoach/orchestrator.py`
Add methods:
```python
def get_user_profile(self) -> dict:
    """Returns {"user_name": str | None, "language_proficiency": dict[str, str]}"""

def set_user_profile(self, user_name: str | None, language_proficiency: dict[str, str]) -> None:
    """Writes to user_profile via execute_update."""

def _user_level_for(self, language: str) -> str:
    """Returns the stored grammar/overall level for the given language.
    Defaults: 'N5' for 'ja'; 'A1' for all other SUPPORTED_LANGUAGES."""

def _user_kanji_level_for(self) -> str:
    """Returns the stored kanji reading level for Japanese ('ja_kanji' key).
    Falls back to _user_level_for('ja') if 'ja_kanji' is not set.
    Returns '' for non-Japanese sessions (callers pass it as empty string to prompts)."""

def _user_name_for_prompt(self) -> str:
    """Returns user_name if set, or empty string ''.
    Callers must not pass None to PromptLoader.render() — it renders as the literal 'None'."""
````

In `generate_reply()`, `run_corrections()`, and `run_explain_and_native()`:

- Call `_user_level_for(self._language)` to get the grammar/overall level string
- Call `_user_kanji_level_for()` to get the kanji level string (empty string for non-Japanese)
- Call `_user_name_for_prompt()` to get the name string (empty string if not set)
- Pass all three into `prompt_loader.render()` as `user_level`, `user_kanji_level`, and `user_name`

The prompts must be written to handle an empty `user_name` gracefully, and to handle an empty `user_kanji_level` gracefully (non-Japanese sessions: omit any kanji-specific guidance).

#### New route file: `src/kaiwacoach/api/routes/settings.py`

```
GET  /api/settings/profile   → 200 { user_name: str | null, language_proficiency: dict }
POST /api/settings/profile   → 204 (body: UserProfileRequest)
```

#### New Pydantic request schema

```python
class UserProfileRequest(BaseModel):
    user_name: str | None = None
    language_proficiency: dict[str, str] = {}
```

Level validation in the POST handler (in `settings.py`, not in the schema):  
For each `(language, level)` pair in `language_proficiency`, check against `VALID_PROFICIENCY_LEVELS`. If any entry is invalid, raise `HTTPException(status_code=422, detail=f"Invalid level '{level}' for language '{language}'")`. Unknown language keys in the dict are rejected with the same 422.

Register `settings_router` in `src/kaiwacoach/api/server.py`.

#### Prompts to update

Add `{user_name}`, `{user_level}`, and `{user_kanji_level}` variables to:

- `src/kaiwacoach/prompts/conversation.md` — address user by name where natural; pitch reply vocabulary and grammar complexity at `user_level`; for Japanese, also pitch kanji usage at `user_kanji_level` (ignore if empty)
- `src/kaiwacoach/prompts/detect_and_correct.md` — pitch correction detail at `user_level`; for Japanese, flag kanji errors relative to `user_kanji_level` (ignore if empty)
- `src/kaiwacoach/prompts/explain_and_native.md` — pitch explanation at `user_level`

After changing prompts, run the prompt test suite:

```bash
poetry run pytest -q tests/test_prompt_schemas.py tests/test_prompt_rendering_suite.py tests/test_prompt_loader.py
```

### Frontend

**New file: `frontend/src/lib/api/settings.ts`**

```typescript
getProfile(): Promise<UserProfile>
setProfile(profile: Partial<UserProfile>): Promise<void>
```

**`frontend/src/components/SettingsPanel.svelte`** (replace stub from Feature 0)

- Slide-in side panel on the right edge, controlled by `$uiStore.settingsOpen`
- Name input field (optional; leave blank to omit)
- Per-language level dropdowns:
    - `ja`: **two** dropdowns —
        - "Grammar level": N5, N4, N3, N2, N1, Native (N5 = beginner, displayed first)
        - "Kanji reading level": N5, N4, N3, N2, N1, Native (independent of grammar; defaults to N5)
    - All other languages: one CEFR dropdown — A1, A2, B1, B2, C1, C2, Native (A1 = beginner, displayed first)
    - "Native" label explanation: tooltip or subtext — "Native Japanese adult level (beyond JLPT N1)"
- Loads current profile from API on panel open
- Save button calls `setProfile()`; close on success

### Definition of Done

- [x] `poetry run pytest -q tests/test_storage_schema.py` — user_profile table, all columns, and default row asserted
- [x] `poetry run pytest -q tests/test_orchestrator_settings.py` (new file) — get/set profile round-trip with real DB; default N5 returned for missing `ja` key; default N5 returned for missing `ja_kanji` key (falls back to `ja` level if that is set); default A1 returned for missing `fr` key; `_user_name_for_prompt()` returns `""` when name is null; `_user_kanji_level_for()` returns `""` when `self._language` is not `"ja"`
- [x] `poetry run pytest -q tests/test_api_settings.py` (new file) — GET returns defaults on fresh DB; POST updates `ja` and `ja_kanji` independently; subsequent GET reflects both; POST with invalid level `{"ja": "fluent"}` returns 422; POST with `{"ja_kanji": "Native"}` succeeds; POST with unknown language key returns 422
- [x] `poetry run pytest -q tests/test_prompt_schemas.py tests/test_prompt_rendering_suite.py tests/test_prompt_loader.py` — all pass after prompt variable additions; rendering with empty string for `user_name` does not raise KeyError
- [x] Manual: set name to "Ashley" and level to N3 for Japanese; send a turn; verify the reply is pitched at N3 level
- [x] Manual: change level to N1 mid-conversation; next reply is noticeably more advanced without restarting
- [x] No TypeScript errors in SettingsPanel; `npm run build` succeeds

---

## 6. Feature 2: Narration Tab

### Goal

A dedicated tab where the user pastes text in the current session language and the app synthesises it to audio using the TTS model. The user can preview and download the result. No LLM call. No DB storage — audio lives in the session cache only.

**Language note**: Narration always uses `self._language` (the orchestrator's session language). The API route does not accept a language parameter — the orchestrator handles language internally.

### Backend

#### New route file: `src/kaiwacoach/api/routes/narration.py`

```
POST /api/narrate
Body: NarrationRequest { text: str }
Response: 200 { audio_url: str }
```

- Single JSON response (not SSE)
- ML work runs in `_ML_EXECUTOR` via `asyncio.get_event_loop().run_in_executor()`
- On empty text: return 400 with message `"Narration text is empty"`
- Route handler needs access to `request.app.state.audio_cache` — wire it the same way as `regen.py` and `turns.py`

#### New Pydantic schema

```python
class NarrationRequest(BaseModel):
    text: str
```

#### `src/kaiwacoach/orchestrator.py`

New method:

```python
def generate_narration(self, text: str) -> str:
    """Synthesises text to audio using self._language; returns raw audio path (not URL).
    The route handler is responsible for converting the path to a URL."""
    # 1. Call _normalise_for_tts(text, self._language)  [reuse existing]
    # 2. Call self._tts.synthesize(
    #        conversation_id="narrations",
    #        turn_id=str(uuid4()),
    #        text=normalised_text,
    #        voice=self._tts_voice,
    #        speed=self._tts_speed,
    #        language=self._language)
    # 3. Return result.audio_path  (raw path, NOT a URL)
```

The **route handler** (in `narration.py`) calls `audio_path_to_url(path, request.app.state.audio_cache.root_dir)` to produce the URL returned in the response. The orchestrator must not import from the API layer.

Reuses internally: `_normalise_for_tts()`, `self._tts.synthesize()`.

Register `narration_router` in `src/kaiwacoach/api/server.py`.

### Frontend

**New file: `frontend/src/lib/api/narration.ts`**

```typescript
generateNarration(text: string): Promise<{ audio_url: string }>
```

**New file: `frontend/src/components/NarrationPanel.svelte`**

```
┌──────────────────────────────────────────────────┐
│  Narrate in [current language]                   │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Paste or type text here...                │  │
│  │                                            │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  [Generate Audio]                                │
│                                                  │
│  ── Preview ────────────────────────────────     │
│  <AudioPlayer />          [↓ Download]           │
└──────────────────────────────────────────────────┘
```

- The current session language is shown as a label (read from `$sessionStore.language`); there is no separate language selector in this panel
- Shows a loading state on the Generate button while the request is in flight
- `AudioPlayer` appears only after audio is generated
- Blob URL management: revoke previous URL before creating a new one; revoke on `onDestroy`
- Download button uses the existing download pattern from `AssistantBubble.svelte`

**`frontend/src/App.svelte`**: sidebar hidden and main panel full-width when `$uiStore.activeTab === 'narration'` (set in Feature 0).

### Definition of Done

- [x] `poetry run pytest -q tests/test_orchestrator_narration.py` (new file) — `generate_narration` calls TTS with `self._language`; mocked TTS returns a path; method returns the raw path string (not a URL)
- [x] `poetry run pytest -q tests/test_api_narration.py` (new file) — POST with valid text returns `audio_url` (route calls `audio_path_to_url` with correct cache root); POST with empty text returns 400; mock orchestrator used; audio_cache state is wired correctly in the test fixture
- [x] Manual: switch to Narration tab; sidebar is gone, panel is full-width
- [x] Manual: paste a sentence in the target language; click Generate Audio; audio player appears and plays correctly
- [x] Manual: Download button saves a WAV file
- [x] Manual: generate a second narration; first blob URL is revoked (no memory leak)
- [x] No TypeScript errors; `npm run build` succeeds

---

## 7. Feature 3: Monologue Mode

### Goal

A dedicated practice mode where the user submits text or audio (mic recording or uploaded file). The app transcribes audio if needed, then analyses the input for language errors, provides corrections and explanations, and produces a summary of the main areas to work on. **No assistant reply is generated.** Sessions are saved to SQLite and appear in the sidebar under the Monologue tab.

**Language note**: All orchestrator operations (ASR, TTS normalisation, corrections, summary) use `self._language`. The API routes do not accept a language parameter — the orchestrator handles language internally.

### Backend

#### Schema change: `src/kaiwacoach/storage/schema.sql`

Add `conversation_type` column to `conversations` (in the `CREATE TABLE` definition):

```sql
conversation_type TEXT NOT NULL DEFAULT 'chat'
```

Bump `schema_version` to **v3**.

Because this adds a column to an existing table, `_schema_needs_reset()` (which checks required column names on `conversations`) **must** be updated. Add `'conversation_type'` to the `required_columns` set. An existing v1/v2 DB without this column will trigger an auto-reset (acceptable for single-user MVP; note this in the PR).

#### `src/kaiwacoach/storage/db.py`

- Add `'conversation_type'` to `_ALLOWED_UPDATE_COLUMNS["conversations"]`
- Add `'conversation_type'` to `required_columns` in `_schema_needs_reset()`

#### New LLM role: `"monologue_summary"`

**Token cap** — add to `LLMRoleCaps` in `src/kaiwacoach/settings.py`:

```python
monologue_summary: int = 256
```

Also update `load_config()` to parse this from the merged config, and add env var support:  
`KAIWACOACH_LLM_ROLE_MONOLOGUE_SUMMARY_MAX_NEW_TOKENS`  
(Follow the same pattern used for existing role caps in `_apply_env_overrides`.)

**Pydantic schema** in `src/kaiwacoach/models/json_enforcement.py`:

```python
class MonologueSummary(BaseModel):
    improvement_areas: conlist(StrictStr, min_length=1)
    overall_assessment: StrictStr
```

Add to `ROLE_SCHEMAS`. Use `conlist` to match existing code style (Pydantic v2 deprecation warning is acceptable for consistency).

**New prompt**: `src/kaiwacoach/prompts/monologue_summary.md`  
Variables: `{language}`, `{errors}`, `{corrected}`, `{explanation}`, `{user_level}`  
`{errors}` format: join the list of error strings with `"; "` — e.g. `"Wrong particle used; verb not conjugated"`. If the list is empty, pass `"No errors detected"`.  
Output schema: `{"improvement_areas": ["...", "..."], "overall_assessment": "..."}`  
Temperature: 0.0 (deterministic).

#### `src/kaiwacoach/orchestrator.py`

New frozen dataclass:

```python
@dataclass(frozen=True)
class MonologueTurnResult:
    conversation_id: str
    user_turn_id: str
    input_text: str
    asr_text: str | None          # None for text input
    asr_meta: dict | None         # None for text input
    corrections: dict             # {errors: list[str], corrected, native, explanation}
    summary: dict                 # {improvement_areas: list[str], overall_assessment}
```

New methods:

```python
def create_monologue_conversation(self) -> str:
    """Creates a conversation with conversation_type='monologue' and auto-title.
    Uses self._language. Title format: f"Monologue – {datetime.date.today().isoformat()}"
    Returns: conversation_id"""

def process_monologue_turn(
    self,
    conversation_id: str,
    text: str | None = None,
    pcm_bytes: bytes | None = None,
    audio_meta: AudioMeta | None = None,
    on_stage: Callable[[str, str, dict], None] | None = None,
) -> MonologueTurnResult:
    """Pipeline: ASR (if audio) → persist user_turn → run_corrections → monologue_summary.
    Uses self._language throughout. No assistant_turn row is created.
    Audio preparation (WebM→PCM conversion, AudioMeta construction) is done by the route
    handler before calling this method — follow the same pattern as POST /api/turns/audio
    in turns.py. The orchestrator receives post-conversion PCM bytes and AudioMeta."""
    # Stages emitted: 'asr' (audio only), 'corrections', 'summary'
```

Update `list_conversations()` to accept `conversation_type: str | None = None` and filter accordingly.

#### New API endpoint: create monologue conversation

Add to `src/kaiwacoach/api/routes/conversations.py` (or a new `monologue.py`):

```
POST /api/conversations/monologue
Body: (no body required — uses orchestrator's self._language)
Response: 201 { conversation_id: str }
```

The frontend calls this before submitting a turn. The returned `conversation_id` is then included in the subsequent turn submission.

#### New API routes: monologue turns

Add to a new `src/kaiwacoach/api/routes/monologue.py`:

```
POST /api/turns/monologue/text   Body: MonologueTextRequest (JSON)
POST /api/turns/monologue/audio  Body: multipart form
```

Both return SSE streams. These routes **implement their own SSE generator** — they do not reuse `_build_sse_generator` from `turns.py` because that function has TTS-specific logic in its `on_stage` callback. Follow the same structural pattern (queue + executor + thread) but without any TTS stage handling.

SSE event sequence:

1. `stage: asr, status: running` ← audio input only
2. `stage: asr, status: complete, data: {transcript: "..."}` ← audio only
3. `stage: corrections, status: running`
4. `stage: corrections, status: complete, data: {errors: [...], corrected, native, explanation}`
5. `stage: summary, status: running`
6. `stage: summary, status: complete, data: {improvement_areas: [...], overall_assessment}`
7. `complete: {conversation_id, user_turn_id, input_text, asr_text, corrections, summary}`

Request schemas:

```python
class MonologueTextRequest(BaseModel):
    conversation_id: str
    text: str
    # No language field — orchestrator uses self._language

# Audio route: multipart form with fields: conversation_id, audio (file)
# File size limit: enforce the same limit as /api/turns/audio (reuse the existing validation)
```

#### `GET /api/conversations` — default behaviour for the ?type param

- `?type=chat` → return only conversations with `conversation_type = 'chat'`
- `?type=monologue` → return only conversations with `conversation_type = 'monologue'`
- **No `?type` param** → return **all** conversations regardless of type

The chat sidebar must pass `?type=chat` explicitly to avoid showing monologue sessions. The monologue sidebar passes `?type=monologue`. Neither should rely on the unfiltered default.

#### Read-only monologue view: GET /api/conversations/{id} response shape

`GET /api/conversations/{id}` will be called to load a past monologue session. Monologue conversations have `user_turn` rows and `corrections` rows but **no `assistant_turn` rows**.

The existing `_build_turn_record()` in `conversations.py` joins across `assistant_turns` and expects a `reply_text`. It must be updated to handle a missing assistant turn gracefully:

- If no `assistant_turn` row exists for a `user_turn`, return a `TurnRecord` with `reply_text: null`, `assistant_turn_id: null`, and `has_audio: false`
- The `conversation_type` field should be included in the `ConversationDetail` response so the frontend can render the correct view

The frontend `MonologuePanel` reads `corrections` from `TurnRecord.correction` and renders them in the read-only results layout. The `reply_text` being null is expected and must not cause a render error.

### Frontend

**New file: `frontend/src/lib/api/monologue.ts`**

```typescript
createMonologueConversation(): Promise<{ conversation_id: string }>
submitMonologueText(params: { conversation_id: string; text: string }): AsyncGenerator<SSEEvent>
submitMonologueAudio(params: { conversation_id: string; audio: Blob }): AsyncGenerator<SSEEvent>
```

**Update `frontend/src/lib/api/conversations.ts`**  
Extend `listConversations()` to accept an optional `type?: 'chat' | 'monologue'` param and pass it as a `?type=` query string. When `type` is undefined, no query param is sent (returns all).

**`frontend/src/components/Sidebar.svelte`**  
When `$uiStore.activeTab === 'chat'`, calls `listConversations({type: 'chat'})`.  
When `$uiStore.activeTab === 'monologue'`, calls `listConversations({type: 'monologue'})`.  
This component already manages the conversation list — update the fetch call based on activeTab.

**New file: `frontend/src/components/MonologuePanel.svelte`**

```
┌─────────────────────────────────────────────────────┐
│  [Text] [Mic] [Upload]               [Analyse]      │
│  ┌───────────────────────────────────────────────┐  │
│  │  <textarea>  OR  <AudioRecorder>              │  │
│  │  OR  <file input accept="audio/*">            │  │
│  └───────────────────────────────────────────────┘  │
│  <PipelineProgress stages={asr?, corrections, summary}> │
│                                                     │
│  ── Results ──────────────────────────────────────  │
│  Transcript: "..."  (shown only for audio input)    │
│  Errors:    • ...  • ...                            │
│  Corrected: "..."                                   │
│  Native:    "..."                                   │
│  Explanation: "..."                                 │
│                                                     │
│  ── Summary ──────────────────────────────────────  │
│  Areas to focus on:                                 │
│  1. ...   2. ...                                    │
│  Overall: "..."                                     │
└─────────────────────────────────────────────────────┘
```

- Reuses `PipelineProgress` for stage display
- Reuses `AudioRecorder` for mic input
- File upload: `<input type="file" accept="audio/*">`; read as `ArrayBuffer` and sent as multipart
- On submit: call `createMonologueConversation()` first, then submit the turn with the returned `conversation_id`
- Input/output modes are mutually exclusive; switching clears the current input

**Sidebar in Monologue tab**:  
`Sidebar.svelte` fetches `listConversations({type: 'monologue'})` when `activeTab === 'monologue'`. Clicking a past session calls `getConversation(id)` and renders the results in a **read-only view** — the correction fields are displayed, the input form is hidden.

**State management for past session selection**:  
`MonologuePanel.svelte` manages its own local `selectedSessionId: string | null` (default `null`). It does **not** write to `sessionStore.conversationId` — that field belongs to the Chat tab and must not be polluted by monologue navigation. When `selectedSessionId` is non-null, `MonologuePanel` shows the read-only results view; when null, it shows the submission form. Switching away from the Monologue tab and back resets `selectedSessionId` to `null` (show the form, not a past session).

### Definition of Done

- [ ] `poetry run pytest -q tests/test_storage_schema.py` — `conversation_type` column asserted on conversations table; existing chat conversation tests still pass
- [ ] `poetry run pytest -q tests/test_orchestrator_monologue.py` (new file) — `process_monologue_turn` with mock LLM, real DB; assert `user_turn` row created; assert `corrections` row created; assert **no** `assistant_turn` row; assert `MonologueTurnResult` fields correct; `create_monologue_conversation` creates row with `conversation_type='monologue'`; `list_conversations(conversation_type='monologue')` returns only monologue conversations; `list_conversations(conversation_type='chat')` returns no monologue conversations
- [ ] `poetry run pytest -q tests/test_sse_monologue.py` (new file) — text route: stage events in correct order → complete; audio route: asr stage fires first; mid-stream failure case for both routes; error event includes `request_id`
- [ ] `poetry run pytest -q tests/test_api_monologue.py` (new file) — POST `/api/conversations/monologue` returns 201 with `conversation_id`; mock orchestrator used
- [ ] `poetry run pytest -q tests/test_prompt_schemas.py tests/test_prompt_rendering_suite.py` — monologue_summary prompt and schema covered
- [ ] `poetry run pytest -q -m "not slow"` — full fast suite passes; no regressions in existing chat flow; `GET /api/conversations?type=chat` does not return monologue conversations
- [ ] Manual (text input): type a sentence with deliberate errors; click Analyse; corrections + summary rendered
- [ ] Manual (mic input): record a sentence; submit; ASR stage fires; corrections + summary rendered
- [ ] Manual (file upload): upload a pre-recorded audio file; result displays correctly
- [ ] Manual: session appears in sidebar under Monologue tab; clicking it shows read-only results with no input form
- [ ] No TypeScript errors; `npm run build` succeeds

---

## 8. Feature 4: Conversation Summary

### Goal

An on-demand analysis of the main error patterns across a conversation. The user clicks a "Summarise" button in the conversation header; the app reads all stored corrections, calls the LLM, and displays the result as a collapsible panel above the chat thread. The result is **not persisted** — it is ephemeral and regenerated on each click.

### Backend

#### New LLM role: `"summarise_conversation"`

**Token cap** — add to `LLMRoleCaps` in `src/kaiwacoach/settings.py`:

```python
summarise_conversation: int = 512
```

Also update `load_config()` and add env var support:  
`KAIWACOACH_LLM_ROLE_SUMMARISE_CONVERSATION_MAX_NEW_TOKENS`

**Pydantic schema** in `src/kaiwacoach/models/json_enforcement.py`:

```python
class ConversationSummaryResult(BaseModel):
    top_error_patterns: conlist(StrictStr, min_length=1)
    priority_areas: conlist(StrictStr, min_length=1)
    overall_notes: StrictStr
```

Add to `ROLE_SCHEMAS`. Use `conlist` for consistency with existing code.

**New prompt**: `src/kaiwacoach/prompts/summarise_conversation.md`  
Variables: `{language}`, `{corrections_text}`, `{user_level}`  
Output schema: `{"top_error_patterns": [...], "priority_areas": [...], "overall_notes": "..."}`  
Temperature: 0.0.

**`{corrections_text}` serialisation**:  
Each correction record in the DB has `errors_json` (a JSON array of strings, possibly multiple entries) and `corrected_text`. Format each row as:

```
[N] Errors: <error_1>; <error_2>  |  Corrected: <corrected_text>
```

Multiple errors within a single row are joined with `"; "`. If `errors_json` is an empty array, omit that row. Use the most recent 20 correction records (ordered by `created_at DESC`, then take the 20 and re-order oldest-first for readability).

#### `src/kaiwacoach/orchestrator.py`

New method:

```python
def summarise_conversation(self, conversation_id: str) -> dict:
    """Reads up to 20 most recent corrections; calls summarise_conversation LLM role.
    Returns {top_error_patterns, priority_areas, overall_notes}.
    Returns a safe fallback dict on failure or if no corrections exist.
    No DB writes. Uses self._language for the prompt."""
    # DB query: corrections are linked to user_turns, not directly to conversations.
    # Use a JOIN to fetch them:
    #   SELECT c.errors_json, c.corrected_text, c.created_at
    #   FROM corrections c
    #   JOIN user_turns ut ON c.user_turn_id = ut.id
    #   WHERE ut.conversation_id = ?
    #   ORDER BY c.created_at DESC LIMIT 20
    # Add a new DB helper: get_corrections_for_conversation(conversation_id) -> list[dict]
    # This helper lives in orchestrator.py alongside get_conversation() and related reads,
    # using self._db.read_connection() (short-lived read connection, not the write queue).
```

#### New API endpoint (in `src/kaiwacoach/api/routes/conversations.py`)

```
POST /api/conversations/{conversation_id}/summarise
Response: 200 { top_error_patterns: [...], priority_areas: [...], overall_notes: "..." }
```

- Single JSON response (not SSE)
- 404 if conversation not found
- 400 with message `"No corrections to summarise"` if the conversation has no correction records
- ML work runs in `_ML_EXECUTOR` via `run_in_executor`; `_ML_EXECUTOR` is defined in `src/kaiwacoach/api/utils.py` and must be imported into `conversations.py`

### Frontend

**`frontend/src/lib/api/conversations.ts`**  
Add:

```typescript
summariseConversation(conversationId: string): Promise<ConversationSummaryResponse>
```

**`frontend/src/components/ConversationHeader.svelte`**

- Add a **"Summarise"** button, shown only when `turns.length > 0`
- Button shows a loading spinner while the request is in flight
- On response: toggle `summaryOpen` local bool and store `summaryData` locally; re-clicking collapses the panel

**New file: `frontend/src/components/ConversationSummaryPanel.svelte`**

- Collapsible section rendered **above `<ChatThread>`**, below `<ConversationHeader>`
- Animated slide-in/out via CSS transition
- Three sections: "Top error patterns" · "Priority areas" · "Overall notes"
- Collapse/close button in panel header
- All state is local to the parent component (no store); no persistence

### Definition of Done

- [ ] `poetry run pytest -q tests/test_orchestrator_summary.py` (new file) — `summarise_conversation` with mock LLM, real DB with correction rows; assert correct corrections read (most recent 20); assert multi-error rows are joined with `"; "`; assert LLM called with correctly formatted `corrections_text`; assert return dict has all three fields; assert conversation with no corrections returns a safe fallback dict (no exception)
- [ ] `poetry run pytest -q tests/test_api_summary.py` (new file) — POST returns 200 with all three summary fields; 404 for unknown conversation; 400 for conversation with no corrections; mock orchestrator used
- [ ] `poetry run pytest -q tests/test_prompt_schemas.py tests/test_prompt_rendering_suite.py` — summarise_conversation prompt and schema covered
- [ ] `poetry run pytest -q -m "not slow"` — full fast suite passes
- [ ] Manual: hold a multi-turn conversation with deliberate errors; click Summarise; collapsible panel appears above the chat with relevant patterns and areas to work on
- [ ] Manual: collapse the panel; it animates closed; chat thread is fully visible
- [ ] Manual: click Summarise again; panel re-opens with freshly generated summary
- [ ] No TypeScript errors; `npm run build` succeeds

---

## 9. Schema Version History

| Version | Change                                                                                                                                                                             |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1      | Original schema: conversations, user_turns, assistant_turns, corrections, artifacts                                                                                                |
| v2      | Add `user_profile` singleton table (Feature 1); created via `CREATE TABLE IF NOT EXISTS` on startup — no reset required for existing DBs                                           |
| v3      | Add `conversation_type TEXT NOT NULL DEFAULT 'chat'` column to `conversations` (Feature 3); triggers auto-reset on existing DBs because `_schema_needs_reset()` checks this column |

---

## 10. New Files Summary

| File                                                      | Feature | Type |
| --------------------------------------------------------- | ------- | ---- |
| `src/kaiwacoach/api/routes/settings.py`                   | 1       | New  |
| `src/kaiwacoach/api/routes/narration.py`                  | 2       | New  |
| `src/kaiwacoach/api/routes/monologue.py`                  | 3       | New  |
| `src/kaiwacoach/prompts/monologue_summary.md`             | 3       | New  |
| `src/kaiwacoach/prompts/summarise_conversation.md`        | 4       | New  |
| `tests/test_orchestrator_settings.py`                     | 1       | New  |
| `tests/test_api_settings.py`                              | 1       | New  |
| `tests/test_orchestrator_narration.py`                    | 2       | New  |
| `tests/test_api_narration.py`                             | 2       | New  |
| `tests/test_orchestrator_monologue.py`                    | 3       | New  |
| `tests/test_sse_monologue.py`                             | 3       | New  |
| `tests/test_api_monologue.py`                             | 3       | New  |
| `tests/test_orchestrator_summary.py`                      | 4       | New  |
| `tests/test_api_summary.py`                               | 4       | New  |
| `frontend/src/components/TabBar.svelte`                   | 0       | New  |
| `frontend/src/components/SettingsPanel.svelte`            | 0/1     | New  |
| `frontend/src/components/NarrationPanel.svelte`           | 2       | New  |
| `frontend/src/components/MonologuePanel.svelte`           | 3       | New  |
| `frontend/src/components/ConversationSummaryPanel.svelte` | 4       | New  |
| `frontend/src/lib/api/settings.ts`                        | 1       | New  |
| `frontend/src/lib/api/narration.ts`                       | 2       | New  |
| `frontend/src/lib/api/monologue.ts`                       | 3       | New  |

---

## 11. Files Modified (not created)

| File                                                | Features   | What changes                                                                                          |
| --------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| `src/kaiwacoach/storage/schema.sql`                 | 1, 3       | Add user_profile table; add conversation_type column; bump version                                    |
| `src/kaiwacoach/storage/db.py`                      | 1, 3       | \_ALLOWED_UPDATE_COLUMNS; \_schema_needs_reset required_columns                                       |
| `src/kaiwacoach/orchestrator.py`                    | 1, 2, 3, 4 | New methods; prompt variable injection                                                                |
| `src/kaiwacoach/models/json_enforcement.py`         | 3, 4       | New schemas; ROLE_SCHEMAS entries                                                                     |
| `src/kaiwacoach/constants.py`                       | 1          | Add VALID_PROFICIENCY_LEVELS dict                                                                     |
| `src/kaiwacoach/settings.py`                        | 3, 4       | LLMRoleCaps new fields; load_config parsing; env var wiring                                           |
| `src/kaiwacoach/api/server.py`                      | 1, 2, 3, 4 | Router registration                                                                                   |
| `src/kaiwacoach/api/routes/conversations.py`        | 3, 4       | Monologue conversation creation; summarise endpoint; \_build_turn_record null assistant_turn handling |
| `src/kaiwacoach/api/schemas/conversation.py`        | 3          | Add conversation_type field to ConversationDetail response schema                                     |
| `src/kaiwacoach/prompts/conversation.md`            | 1          | Add user_name, user_level variables                                                                   |
| `src/kaiwacoach/prompts/detect_and_correct.md`      | 1          | Add user_level variable                                                                               |
| `src/kaiwacoach/prompts/explain_and_native.md`      | 1          | Add user_level variable                                                                               |
| `tests/test_storage_schema.py`                      | 1, 3       | Assert new table and column                                                                           |
| `frontend/src/App.svelte`                           | 0, 2, 3    | Tab bar; settings overlay; conditional panel rendering                                                |
| `frontend/src/lib/stores/ui.ts`                     | 0          | activeTab; settingsOpen                                                                               |
| `frontend/src/lib/api/conversations.ts`             | 3, 4       | type filter param; summarise function                                                                 |
| `frontend/src/components/Sidebar.svelte`            | 3          | Conditional fetch based on activeTab                                                                  |
| `frontend/src/components/ConversationHeader.svelte` | 4          | Summarise button                                                                                      |

---

## 12. Implementation / Branching Plan

### Guiding principles

- One feature per PR. PRs are reviewable in isolation.
- Every branch is cut from **main** (never from another feature branch), except where an explicit dependency is noted.
- Each PR must pass `poetry run pytest -q -m "not slow"` and `npm run build` before merging.
- Schema migrations ship in the same PR as the code that requires them (no split-migration PRs).
- Branches with a frontend dependency on Feature 0 (tab bar) are cut from main but should not be merged until `feature/tab-bar-navigation` is merged — the overlapping `App.svelte` edits will conflict otherwise. Rebase onto main after that merge lands.

### PR sequence

| Order | Status | Branch                             | Feature                                                                                         | Dependencies before merge |
| ----- | ------ | ---------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------- |
| 1     | ✅     | `feature/new-features-exploration` | Planning docs only (this branch)                                                                | None — merge immediately  |
| 2     | ✅     | `feature/tab-bar-navigation`       | Feature 0: tab bar + settings stub + narration/monologue stubs                                  | PR 1 merged               |
| 3     | ✅     | `feature/user-settings`            | Feature 1: user_profile schema, orchestrator methods, settings API routes, SettingsPanel        | PR 2 merged               |
| 4     | ✅     | `feature/narration-tab`            | Feature 2: narration orchestrator method, API route, NarrationPanel                             | PR 2 merged               |
| 5     | ⬜     | `feature/monologue-mode`           | Feature 3: conversation_type schema, monologue orchestrator methods, SSE routes, MonologuePanel | PRs 2 + 3 merged          |
| 6     | ⬜     | `feature/conversation-summary`     | Feature 4: summarise_conversation role + route, ConversationSummaryPanel                        | PRs 2 + 3 merged          |

### Per-branch scope

**PR 1 — `feature/new-features-exploration`** (current; commit and open PR immediately)

- `docs/planning/KaiwaCoach_Feature_Expansion_Plan_v1.0.md` (this file)
- `docs/planning/KaiwaCoach_Implementation_Checklist_v1.1.md` (updated post-MVP items to ✅)
- No code changes. CI trivially passes.

**PR 2 — `feature/tab-bar-navigation`**

- `frontend/src/lib/stores/ui.ts` — add `activeTab`, `settingsOpen`
- `frontend/src/components/TabBar.svelte` — new
- `frontend/src/components/SettingsPanel.svelte` — stub (no API calls yet)
- `frontend/src/components/NarrationPanel.svelte` — stub (placeholder text)
- `frontend/src/components/MonologuePanel.svelte` — stub (placeholder text)
- `frontend/src/App.svelte` — gear icon, tab bar render, conditional panel, Esc handler
- `frontend/src/components/Sidebar.svelte` — hide when narration tab active
- Definition of done: §4 checklist.

**PR 3 — `feature/user-settings`** (cut from main; rebase after PR 2 merges)

- `src/kaiwacoach/storage/schema.sql` — v2: `user_profile` table
- `src/kaiwacoach/storage/db.py` — `_ALLOWED_UPDATE_COLUMNS` update
- `src/kaiwacoach/constants.py` — `VALID_PROFICIENCY_LEVELS`
- `src/kaiwacoach/orchestrator.py` — profile methods; inject `user_level`/`user_kanji_level`/`user_name` into `generate_reply`, `run_corrections`, `run_explain_and_native`
- `src/kaiwacoach/prompts/conversation.md`, `detect_and_correct.md`, `explain_and_native.md` — add prompt variables
- `src/kaiwacoach/api/routes/settings.py` — new; GET + POST `/api/settings/profile`
- `src/kaiwacoach/api/server.py` — register settings router
- `frontend/src/lib/api/settings.ts` — new
- `frontend/src/components/SettingsPanel.svelte` — replace stub with real form
- `tests/test_storage_schema.py`, `tests/test_orchestrator_settings.py`, `tests/test_api_settings.py` — new/updated
- Definition of done: §5 checklist.

**PR 4 — `feature/narration-tab`** (cut from main; rebase after PR 2 merges)

- `src/kaiwacoach/orchestrator.py` — `generate_narration`
- `src/kaiwacoach/api/routes/narration.py` — new; POST `/api/narrate`
- `src/kaiwacoach/api/server.py` — register narration router
- `frontend/src/lib/api/narration.ts` — new
- `frontend/src/components/NarrationPanel.svelte` — replace stub with real panel
- `tests/test_orchestrator_narration.py`, `tests/test_api_narration.py` — new
- Definition of done: §6 checklist.

**PR 5 — `feature/monologue-mode`** (cut from main; rebase after PRs 2 + 3 merge)

- `src/kaiwacoach/storage/schema.sql` — v3: `conversation_type` column on `conversations`
- `src/kaiwacoach/storage/db.py` — `_ALLOWED_UPDATE_COLUMNS`, `_schema_needs_reset` `required_columns`
- `src/kaiwacoach/settings.py` — `LLMRoleCaps.monologue_summary`; `load_config`; env var
- `src/kaiwacoach/models/json_enforcement.py` — `MonologueSummary` schema; `ROLE_SCHEMAS`
- `src/kaiwacoach/prompts/monologue_summary.md` — new
- `src/kaiwacoach/orchestrator.py` — `MonologueTurnResult`, `create_monologue_conversation`, `process_monologue_turn`; `list_conversations` `conversation_type` filter
- `src/kaiwacoach/api/routes/conversations.py` — POST `/api/conversations/monologue`; null assistant_turn handling; `conversation_type` filter on `GET /api/conversations`
- `src/kaiwacoach/api/routes/monologue.py` — new SSE routes
- `src/kaiwacoach/api/schemas/conversation.py` — add `conversation_type` to `ConversationDetail`
- `src/kaiwacoach/api/server.py` — register monologue router
- `frontend/src/lib/api/conversations.ts`, `monologue.ts` — updated/new
- `frontend/src/components/Sidebar.svelte`, `MonologuePanel.svelte` — updated/replaced
- `tests/test_storage_schema.py`, `tests/test_orchestrator_monologue.py`, `tests/test_sse_monologue.py`, `tests/test_api_monologue.py` — new/updated
- Definition of done: §7 checklist.

**PR 6 — `feature/conversation-summary`** (cut from main; rebase after PRs 2 + 3 merge)

- `src/kaiwacoach/settings.py` — `LLMRoleCaps.summarise_conversation`; `load_config`; env var
- `src/kaiwacoach/models/json_enforcement.py` — `ConversationSummaryResult` schema; `ROLE_SCHEMAS`
- `src/kaiwacoach/prompts/summarise_conversation.md` — new
- `src/kaiwacoach/orchestrator.py` — `summarise_conversation`; `get_corrections_for_conversation`
- `src/kaiwacoach/api/routes/conversations.py` — POST `/api/conversations/{id}/summarise`
- `frontend/src/lib/api/conversations.ts` — `summariseConversation`
- `frontend/src/components/ConversationHeader.svelte`, `ConversationSummaryPanel.svelte` — updated/new
- `tests/test_orchestrator_summary.py`, `tests/test_api_summary.py` — new
- Definition of done: §8 checklist.

### Parallelism

PRs 3, 4, and 6 can be developed concurrently (no shared file conflicts aside from `orchestrator.py` — coordinate if pairing). PRs 5 and 6 share `json_enforcement.py` and `settings.py` but the additions are additive; merge order does not matter as long as both rebase onto main before opening PRs.
