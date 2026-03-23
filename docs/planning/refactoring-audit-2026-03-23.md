# KaiwaCoach Refactoring Audit — 2026-03-23

## Context

This document is the output of a staff-engineer-level code audit of the KaiwaCoach codebase (post-v2, main branch as of 2026-03-23). The goal is to address bugs, CLAUDE.md convention violations, technical debt, and test coverage gaps accumulated during rapid feature development. Issues are ordered **most to least important**: correctness bugs first, then memory/reliability risks, then code quality and convention violations, then test gaps and minor cleanup.

This document is self-contained — any engineer or coding agent can pick up tasks individually. Each task identifies the exact files and lines involved, what to change, and why.

**Branch:** `refactor/staff-engineer-audit`

---

## Task List

### P0 — Correctness Bugs (fix before anything else)

---

#### TASK-01: Fix broken regex patterns in `protected_spans.py`

**File:** `src/kaiwacoach/textnorm/protected_spans.py`
**Lines:** 10–17
**Severity:** Bug — all protected-span masking is silently non-functional

**Problem:**
The regex patterns use `\\` inside raw strings (`r"..."`), which produces literal backslash characters instead of regex metacharacters. For example:

- `r"https?://[^\\s)]+"` matches literal `\s`, not whitespace — URLs are never masked.
- `` r"```[\\s\\S]*?```" `` matches literal `\s\S`, not any character — code blocks are never masked.
- `r"\\[[^\\]]+\\]\\([^\\)]+\\)"` double-escapes brackets unnecessarily.

The masking loop in `mask_protected_spans()` silently produces no substitutions because nothing matches, so Japanese text that coincidentally contains URL-like or code-like substrings passes through unprotected.

**Fix:**
Replace each pattern with its correctly escaped form:

```python
_URL_RE           = re.compile(r"https?://[^\s)]+")
_EMAIL_RE         = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CODE_BLOCK_RE    = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE   = re.compile(r"`[^`]+`")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\([^\)]+\)")
_FILE_PATH_RE     = re.compile(r"(?:/[^\s]+)+|(?:[A-Za-z]:\\[^\s]+)|(?:~/(?:[^\s]+)*)")
```

Also remove the old-style `from typing import Dict, List, Tuple` import — replace uses with built-in `dict`, `list`, `tuple`.

**Verification:**
```bash
poetry run pytest -q tests/test_invariants.py tests/test_jp_katakana.py \
  tests/test_jp_normalisation_golden.py tests/test_jp_tts_normalisation.py \
  tests/test_protected_spans.py
```
Confirm tests in `tests/test_protected_spans.py` verify that URLs, code blocks, and markdown links are actually masked (placeholder token appears in the output).

---

#### TASK-02: Fix unbounded LLM cache in `llm_qwen.py`

**File:** `src/kaiwacoach/models/llm_qwen.py`
**Line:** 91
**Severity:** Memory leak — violates CLAUDE.md "Bounded in-memory caches" rule

**Problem:**
```python
self._cache: Dict[Tuple[str, str, int], LLMResult] = {}
```
This is an unbounded plain `dict`. Every unique prompt hash + role + token-cap combination adds an entry permanently. In a long-running session (many turns, many roles), this grows without bound. The `_BoundedDict` utility already exists in `orchestrator.py` specifically to address this pattern.

**Fix:**
TASK-09 must be done first (move `_BoundedDict` to `src/kaiwacoach/utils.py`). Then replace the plain dict:

```python
_LLM_CACHE_MAX = 256  # Named constant

self._cache: _BoundedDict[tuple[str, str, int], LLMResult] = _BoundedDict(_LLM_CACHE_MAX)
```

Also update typing imports to Python 3.9+ built-ins as part of this change (see TASK-05).

**Verification:**
```bash
poetry run pytest -q tests/test_llm_qwen.py
```

---

#### TASK-03: Add logging to the silent TTS exception handler in `orchestrator.py`

**File:** `src/kaiwacoach/orchestrator.py`
**Lines:** 869–871
**Severity:** Observability bug — TTS failures are invisible in logs

**Problem:**
```python
except Exception:
    return None
```
When TTS synthesis raises (e.g., model OOM, bad audio path), the exception is swallowed with no log entry. This violates the CLAUDE.md narrow exception handling rule.

**Fix:**
```python
except Exception:
    _logger.exception(
        "TTS synthesis failed for turn_id=%s text_len=%d language=%s",
        assistant_turn_id,
        len(normalised_text),
        self._language,
    )
    return None
```

**Verification:**
Add a test that mocks the TTS backend to raise, then asserts `run_tts()` returns `None` and a log record at `ERROR` level was emitted (use `caplog`).

---

#### TASK-04: Remove misleading `# pragma: no cover` in `loader.py`

**File:** `src/kaiwacoach/prompts/loader.py`
**Line:** 55
**Severity:** CLAUDE.md violation — "No `# pragma: no cover` on reachable error paths"

**Problem:**
```python
def __missing__(self, key: str) -> str:  # pragma: no cover - raised in format_map
    raise KeyError(key)
```
This method IS reachable and IS tested (in `test_prompt_loader.py`). The pragma is incorrect.

**Fix:** Remove `# pragma: no cover` from the line.

**Verification:**
```bash
poetry run pytest -q tests/test_prompt_loader.py
```

---

### P1 — Memory / Reliability Risks

---

#### TASK-05: Update all type hints to Python 3.9+ built-ins

**Files:**
- `src/kaiwacoach/models/llm_qwen.py` — `Dict`, `Optional`, `Tuple`
- `src/kaiwacoach/models/tts_kokoro.py` — `Dict`, `Optional`
- `src/kaiwacoach/models/asr_whisper.py` — `Dict`, `Tuple`
- `src/kaiwacoach/models/protocols.py` — `Dict`, `Optional`
- `src/kaiwacoach/models/json_enforcement.py` — `Dict`, `Optional`
- `src/kaiwacoach/textnorm/protected_spans.py` — `Dict`, `List`, `Tuple` (handled by TASK-01)
- Other textnorm files — audit for `typing` imports

**Fix:** Replace `Dict[K,V]` → `dict[K,V]`, `List[T]` → `list[T]`, `Tuple[A,B]` → `tuple[A,B]`, `Optional[X]` → `X | None`. Keep `Any`, `Callable`, `Protocol`, `Mapping`.

**Verification:**
```bash
poetry run pytest -q -m "not slow"
```

---

#### TASK-06: Extract duplicate ASR caching logic in `orchestrator.py`

**File:** `src/kaiwacoach/orchestrator.py`
**Lines:** ~240–288 and ~435–478
**Severity:** Maintenance risk — duplicated ~40-line blocks will diverge

**Problem:** `process_audio_turn()` and `prepare_audio_turn()` each contain an independent copy of the same ASR cache lookup, cache miss path, and persistence logic.

**Fix:** Extract a private method `_run_asr_with_caching(audio_bytes: bytes, request_id: str | None) -> ASRResult` that encapsulates the shared logic. Both callers use it.

**Verification:**
```bash
poetry run pytest -q tests/test_orchestrator_audio_flow.py
```

---

#### TASK-07: Replace private attribute `getattr` access with public API in `orchestrator.py`

**File:** `src/kaiwacoach/orchestrator.py`
**Lines:** ~243–244, ~436–437, ~1125
**Severity:** Brittle coupling — reaches into ASR wrapper internals

**Problem:** Three locations use:
```python
model_id = getattr(self._asr, "model_id", None) or getattr(self._asr, "_model_id", "unknown")
language = getattr(self._asr, "language", None) or getattr(self._asr, "_language", self._language)
```

**Fix (choose one):**
1. Add `model_id: str` and `language: str` as properties to `ASRProtocol` in `protocols.py`, implement on `WhisperASR`.
2. Extract a single helper `_asr_cache_key(asr, fallback_language)` that centralises the getattr pattern.

Option 1 preferred; Option 2 if ASRProtocol changes are risky.

**Verification:**
```bash
poetry run pytest -q tests/test_orchestrator_audio_flow.py tests/test_orchestrator_text_flow.py
```

---

### P2 — Code Quality & Convention Violations

---

#### TASK-08: Replace silent catch blocks in `LanguageSelector.svelte`

**File:** `frontend/src/components/LanguageSelector.svelte`
**Lines:** ~25–27, ~43–46
**Severity:** CLAUDE.md violation

**Fix:**
```typescript
} catch (e) {
  if (import.meta.env.DEV) {
    console.warn('[LanguageSelector] language change failed:', e)
  }
}
```

**Verification:** Manual — toggle language with backend stopped in dev mode; confirm warning in browser console.

---

#### TASK-09: Move `_BoundedDict` to a shared utility module

**Current location:** `src/kaiwacoach/orchestrator.py`
**Severity:** Required before TASK-02 to avoid circular imports

**Fix:** Move `_BoundedDict` to `src/kaiwacoach/utils.py`. Update `orchestrator.py` to import from there. `llm_qwen.py` can then also import from `utils.py` without a circular dependency.

**Verification:**
```bash
poetry run pytest -q -m "not slow"
```

---

#### TASK-10: Fix inconsistency in regen SSE `complete` event encoding

**File:** `src/kaiwacoach/api/routes/regen.py`
**Line:** 105

**Problem:** `yield {"event": "complete", "data": "{}"}` passes a raw string instead of `json.dumps({})` like all other SSE routes.

**Fix:** `yield {"event": "complete", "data": json.dumps({})}`

**Verification:**
```bash
poetry run pytest -q tests/test_api_regen.py
```

---

#### TASK-11: Narrow broad exception handler for JSON parse error in `orchestrator.py`

**File:** `src/kaiwacoach/orchestrator.py`
**Line:** ~964

**Problem:** `except Exception: return []` silently swallows JSON parse errors, making data corruption indistinguishable from "no corrections."

**Fix:**
```python
except json.JSONDecodeError:
    _logger.warning("Failed to parse corrections JSON for turn_id=%s", turn_id)
    return []
```

**Verification:** Add a test that stores invalid JSON in corrections and asserts the warning was logged.

---

### P3 — Test Coverage Gaps

---

#### TASK-12: Verify `test_protected_spans.py` covers actual masking (post TASK-01)

After TASK-01, confirm tests assert non-empty masking results:
- A URL in input produces a placeholder token in masked output
- Restoring the masked output reproduces the original text exactly

```bash
poetry run pytest -q tests/test_protected_spans.py
```

---

#### TASK-13: Add regression test for bounded LLM cache (post TASK-02)

**File:** `tests/test_llm_qwen.py`

Test: create `QwenLLM` with mock backend, generate `_LLM_CACHE_MAX + 10` unique prompts, assert `len(instance._cache) <= _LLM_CACHE_MAX`.

---

#### TASK-14: Add regression test for TTS exception logging (post TASK-03)

**File:** `tests/test_orchestrator_text_flow.py`

Test: mock TTS backend to raise `RuntimeError`, call `run_tts()`, assert return is `None` and `caplog` captured an `ERROR` level entry.

---

## Recommended Execution Order

Dependencies constrain order for some tasks:

1. **TASK-09** (move `_BoundedDict`) — must precede TASK-02
2. **TASK-01** (regex fix) and **TASK-04** (pragma) — independent, do in any order
3. **TASK-02** (bound LLM cache) — after TASK-09
4. **TASK-03** (TTS logging) — independent
5. **TASK-05** (type hints) — can be done alongside others
6. **TASK-06**, **TASK-07** (orchestrator refactors) — independent of each other
7. **TASK-08**, **TASK-10**, **TASK-11** — independent
8. **TASK-12**, **TASK-13**, **TASK-14** — after their respective P0 fixes

## Verification (Full Suite)

```bash
# Full non-slow suite
poetry run pytest -q -m "not slow"

# Targeted suites
poetry run pytest -q tests/test_invariants.py tests/test_jp_katakana.py \
  tests/test_jp_normalisation_golden.py tests/test_jp_tts_normalisation.py \
  tests/test_protected_spans.py

poetry run pytest -q tests/test_llm_qwen.py tests/test_tts_kokoro.py

poetry run pytest -q tests/test_orchestrator_text_flow.py tests/test_orchestrator_audio_flow.py

poetry run pytest -q tests/test_api_regen.py tests/test_api_turns.py
```

Update the passing test count in `README.md` when all tasks are complete.

---

## Summary Table

| Task | Area | Description | Priority |
|------|------|-------------|----------|
| TASK-01 | textnorm | Fix broken regex escaping in `protected_spans.py` | P0 |
| TASK-02 | models | Bound the LLM cache in `llm_qwen.py` | P0 |
| TASK-03 | orchestrator | Log the silent TTS exception | P0 |
| TASK-04 | prompts | Remove incorrect `# pragma: no cover` in `loader.py` | P0 |
| TASK-05 | models/textnorm | Update all type hints to Python 3.9+ built-ins | P1 |
| TASK-06 | orchestrator | Extract duplicate ASR caching logic | P1 |
| TASK-07 | orchestrator | Replace private-attr `getattr` with public API | P1 |
| TASK-08 | frontend | Replace silent catch blocks in `LanguageSelector.svelte` | P2 |
| TASK-09 | utils | Move `_BoundedDict` to shared `utils.py` | P2 |
| TASK-10 | api | Fix regen SSE `complete` event encoding inconsistency | P2 |
| TASK-11 | orchestrator | Narrow JSON parse exception handler | P2 |
| TASK-12 | tests | Verify protected-span masking tests assert non-empty output | P3 |
| TASK-13 | tests | Add regression test for bounded LLM cache | P3 |
| TASK-14 | tests | Add regression test for TTS exception logging | P3 |
