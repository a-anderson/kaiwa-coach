# VoiceVox TTS Integration Plan

**Branch:** `feature/voicevox-tts`  
**Approach:** Test-Driven Development — tests are written before implementation in each phase.

---

## Context

KaiwaCoach uses Kokoro (MLX Audio) for all TTS. This plan adds VoiceVox as a higher-quality Japanese TTS option that routes Japanese synthesis through VoiceVox when the local server is running, with automatic per-call fallback to Kokoro when it is not.

VoiceVox runs as a local HTTP server (`http://localhost:50021`) and is offline-first, consistent with the project's runtime constraints.

The `tts` config section is restructured from flat keys to backend-nested keys to support future per-language backend routing (e.g. voicevox for ja, kokoro for fr, chatterbox for en).

---

## Architecture Summary

```
build_tts() in factory.py
├── always builds KokoroTTS (fallback)
├── checks VoiceVoxBackend.check_available(url)
│   ├── available → builds VoiceVoxTTS + returns LanguageDispatchTTS
│   └── not available → returns KokoroTTS (same as today)
│
LanguageDispatchTTS.synthesize()
├── language == "ja" → try VoiceVoxTTS
│   └── OSError → warn + fall back to KokoroTTS
└── other → KokoroTTS directly
```

---

## Config Design

### `config.example.yaml`

```yaml
tts:
  kokoro:
    voice: "default"
    speed: 1.0
  voicevox:
    url: "http://localhost:50021"
    speaker_id: 74   # 琴詠ニヤ — see voicevox.hiroshiba.jp for full list
    speed: 1.0       # maps to VoiceVox speedScale
```

### Env vars

| Old | New |
|-----|-----|
| `KAIWACOACH_TTS_VOICE` | `KAIWACOACH_TTS_KOKORO_VOICE` |
| `KAIWACOACH_TTS_SPEED` | `KAIWACOACH_TTS_KOKORO_SPEED` |
| *(new)* | `KAIWACOACH_TTS_VOICEVOX_URL` |
| *(new)* | `KAIWACOACH_TTS_VOICEVOX_SPEAKER_ID` |
| *(new)* | `KAIWACOACH_TTS_VOICEVOX_SPEED` |

---

## Phase 0 — Branch Setup

- [x] Create branch `feature/voicevox-tts` from `main`

**Definition of Done:** Branch exists and is checked out.

---

## Phase 1 — Config Restructure (TDD) ✅

### 1a. Write failing tests first

- [x] Created `tests/test_settings_tts_config.py` with 15 tests covering defaults, env var overrides, validation, and to_dict round-trip
- [x] Confirmed all 15 tests failed before implementation

### 1b. Implement

- [x] Added `VOICEVOX_DEFAULT_URL` and `VOICEVOX_DEFAULT_SPEAKER_ID` to `src/kaiwacoach/config/models.py`
- [x] Replaced flat `TTSConfig` in `src/kaiwacoach/settings.py` with nested `KokoroConfig` / `VoiceVoxConfig` / `TTSConfig`
- [x] Updated env var mappings (renamed Kokoro vars, added VoiceVox vars)
- [x] Updated YAML loading for nested structure
- [x] Updated `to_dict()` in `AppConfig`
- [x] Moved speed validation: `tts.speed > 0` → `tts.kokoro.speed > 0` AND `tts.voicevox.speed > 0`
- [x] Updated `config.example.yaml` with nested `tts:` structure

### 1c. Update broken callsites

- [x] `src/kaiwacoach/app.py`: `config.tts.voice` → `config.tts.kokoro.voice`, `config.tts.speed` → `config.tts.kokoro.speed`
- [x] `tests/test_model_integration.py`: `config.tts.speed` → `config.tts.kokoro.speed`
- [x] `tests/test_app_startup.py`: updated both SimpleNamespace stubs to use nested structure

**Post-review fixes applied:**
- [x] `VOICEVOX_DEFAULT_URL` and `VOICEVOX_DEFAULT_SPEAKER_ID` now imported and used in `VoiceVoxConfig` defaults and `load_config` defaults dict (removed dead literal duplication)
- [x] Added `_warn_stale_tts_keys()` and `_warn_stale_tts_env_vars()` — startup warnings when old flat `tts.voice`/`tts.speed` keys or `KAIWACOACH_TTS_VOICE`/`KAIWACOACH_TTS_SPEED` env vars are detected
- [x] Added migration note to `config.example.yaml`
- [x] Added 10 new tests: YAML round-trip (4), partial override (1), stale key warnings (5) — total 25 tests

**Definition of Done:** ✅
- All 25 config tests pass
- `grep -rn "config\.tts\.voice\|config\.tts\.speed" src/ tests/` returns zero hits
- `429 passed` in full non-slow suite, no regressions

---

## Phase 2 — VoiceVox Backend (TDD) ✅

### 2a. Write failing tests first

- [x] Created `tests/test_tts_voicevox.py` with tests for:

  **`VoiceVoxBackend`**
  - `check_available()` returns `True` when GET `/version` succeeds (mock `urllib.request.urlopen`)
  - `check_available()` returns `False` on `OSError`
  - `synthesize()` sends POST to `/audio_query` with correct `text` and `speaker` query params
  - `synthesize()` sets `speedScale` in the AudioQuery before sending to `/synthesis`
  - `synthesize()` sends POST to `/synthesis` with correct `speaker` query param
  - `synthesize()` uses `wave` module to parse returned WAV; returns correct `AudioMeta` sample rate
  - `synthesize()` returns PCM bytes (not raw WAV bytes)

  **`VoiceVoxTTS`**
  - `model_id` returns `"voicevox:speaker=74"` (or configured speaker)
  - `synthesize()` returns cached result on cache hit (backend not called)
  - `synthesize()` calls backend and stores result on cache miss
  - `synthesize()` ignores `voice` and `lang_code` params (uses constructor `speaker_id`)
  - `synthesize()` ignores `speed` param from call (uses constructor `speed`)

  **`LanguageDispatchTTS`**
  - Routes `language="ja"` to `japanese_tts`
  - Routes `language="fr"` to `default_tts`
  - Routes `language=None` to `default_tts`
  - Falls back to `default_tts` when `japanese_tts` raises `OSError` (mid-session VoiceVox closure)
  - Logs warning on fallback including language and first 40 chars of text
  - `model_id` includes both backend IDs

- [x] Confirmed all 20 tests failed before implementation

### 2b. Implement

- [x] Created `src/kaiwacoach/models/tts_voicevox.py` with `VoiceVoxBackend`, `VoiceVoxTTS`, `LanguageDispatchTTS`
- [x] WAV parsing uses `wave` module
- [x] `speedScale` set in AudioQuery before POST to `/synthesis`
- [x] Cache key: `SHA256(f"voicevox|{text}|{speaker_id}|{speed}")`
- [x] `_cache_index` uses `BoundedDict(maxsize=512)` (not a plain dict)

**Definition of Done:** ✅
- All 20 tests in `test_tts_voicevox.py` pass
- `449 passed` in full non-slow suite, no regressions

---

## Phase 3 — Factory Routing (TDD) ✅

### 3a. Write failing tests first

- [x] In `tests/test_model_factory.py`, added 7 tests:
  - `build_tts()` returns `LanguageDispatchTTS` when `VoiceVoxBackend.check_available` is mocked to return `True`
  - `build_tts()` returns `KokoroTTS` when `VoiceVoxBackend.check_available` is mocked to return `False`
  - `build_tts()` wires `speaker_id`, `url`, and `speed` from `config.tts.voicevox` to `VoiceVoxTTS`
  - `build_tts()` logs `"VoiceVox available"` with speaker ID when available
  - `build_tts()` logs `"VoiceVox not available"` when unavailable
- [x] Confirmed 6 of 7 tests failed before implementation (7th passed trivially since `build_tts` already returned `KokoroTTS`)

### 3b. Implement

- [x] Updated `build_tts()` in `src/kaiwacoach/models/factory.py` to check `VoiceVoxBackend.check_available()` and return `LanguageDispatchTTS` or `KokoroTTS`
- [x] Added `import logging` and `_log = logging.getLogger(__name__)` to `factory.py`
- [x] Imported `VoiceVoxBackend`, `VoiceVoxTTS`, `LanguageDispatchTTS` from `tts_voicevox`

**Definition of Done:** ✅
- All 7 factory routing tests pass
- `456 passed` in full non-slow suite, no regressions

---

## Phase 4 — Startup and Remaining Test Updates ✅

- [x] Updated `tests/test_app_startup.py`:
  - Config stubs already used nested structure (updated in Phase 1); no path fixes needed
  - Added `test_main_logs_voicevox_available_when_server_reachable`: lets `build_tts` run for real (stubs model constructors in `factory_module`), patches `VoiceVoxBackend.check_available` → `True`, asserts `"VoiceVox available"` and speaker ID 74 appear in factory log
  - Added `test_main_logs_voicevox_unavailable_when_server_not_reachable`: patches `check_available` → `False`, asserts `"VoiceVox not available"` appears
  - Extracted `_voicevox_startup_config()` helper and `_StartupNoop` class at module level to avoid duplication between the two new tests
- [x] `tests/test_tts_kokoro.py`: no stale config references found
- [x] `grep -rn "config\.tts\.voice\|config\.tts\.speed" src/ tests/` returns zero hits

**Definition of Done:** ✅
- `459 passed` in full non-slow suite, zero failures
- No remaining references to old flat config paths in src/ or tests/
- README test count updated to 459

---

## Phase 5 — Integration Verification ✅

- [x] Run full non-slow test suite: `poetry run pytest -q -m "not slow"` — 459 passed
- [x] Update test count snapshot in `README.md` — updated to 459
- [x] Manual smoke test with VoiceVox running:
  - [x] Open VoiceVox app
  - [x] Start backend: `poetry run python -m kaiwacoach.app`
  - [x] Confirm startup log: `"VoiceVox available — Japanese TTS will use speaker 74"`
  - [x] Japanese session: send a chat message; verify VoiceVox audio plays
  - [x] French session: confirm Kokoro audio plays (different voice character)
  - [x] Close VoiceVox mid-session; send a Japanese message; verify warning logged and Kokoro fallback audio plays
  - [x] Japanese session: test Narration tab; verify VoiceVox audio
  - [x] Restart with VoiceVox closed; confirm log: `"VoiceVox not available — using Kokoro for all languages"`

**Definition of Done:**
- All manual smoke tests pass
- No console errors during normal operation
- README test count updated ✅

---

## Completion Criteria

The feature is complete when:

1. All automated tests pass: `poetry run pytest -q -m "not slow"`
2. No old flat TTS config references remain: `grep -rn "config\.tts\.voice\|config\.tts\.speed" src/ tests/` returns zero hits
3. All manual smoke test items above are checked off
4. README test count is updated
