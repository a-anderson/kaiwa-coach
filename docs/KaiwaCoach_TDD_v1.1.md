# KaiwaCoach – Technical Design Document (TDD v1)
**Sits underneath PRD v2**

## 0. Purpose

**Version note:** v1.1 incorporates validated MLX implementation details (sampler-based generation, robust JSON extraction, Japanese TTS dependency preflight).

This TDD describes the implementation approach for KaiwaCoach as specified in PRD v2, focusing on module boundaries, data contracts, pipeline orchestration, configuration, and test strategy. It is intentionally short and pragmatic for a personal offline project.

---

## 1. Scope and Assumptions

### In scope
- Offline local app on Apple Silicon macOS
- Python 3.11+
- Gradio UI
- ASR: `mlx-community/whisper-large-v3-turbo-asr-fp16`
- LLM: `mlx-community/Qwen3-14B-8bit` (default), `mlx-community/Qwen3-14B-bf16` (optional)
- TTS: `mlx-community/Kokoro-82M-bf16`
- Persistent storage: SQLite (text + metadata). Audio is session-only cache.

Implementation note (ASR):
- Use `mlx-whisper` for the ASR backend to keep the dependency surface minimal and maintain
  explicit control over decoding and language forcing.

### Assumptions
- Single-user process (no multi-user server)
- Single active conversation session in UI at a time
- Batch inference per turn (no streaming)

---

## 2. Repository Layout

```
kaiwacoach/
  app.py                      # Gradio entrypoint
  settings.py                 # Config loading + defaults
  orchestrator.py             # Turn pipeline coordinator
  storage/
    db.py                     # SQLite access layer (single-writer)
    schema.sql                # Schema migrations
    models.py                 # Pydantic DB models
    blobs.py                  # Session audio cache IO, paths, hashing
  models/
    asr_whisper.py            # Whisper MLX wrapper
    llm_qwen.py               # MLX-LM wrapper + schema enforcement
    tts_kokoro.py             # Kokoro wrapper
  prompts/
    conversation.md
    detect_errors.md
    correct_sentence.md
    native_rewrite.md
    explain.md
    jp_tts_normalise.md
    repair_json.md
  textnorm/
    protected_spans.py        # Mask/restore URLs/code
    jp_katakana.py            # LLM-based katakana normalisation calls
    tts_punctuation.py        # Pause/sentence break normalisation
    invariants.py             # Byte-identical JP checks (planned mitigation hooks)
  tests/
    test_prompt_schemas.py
    test_jp_normalisation_golden.py
    test_storage_roundtrip.py
    fixtures/
      jp_normalisation_cases.json
      prompts/
  tools/
    export_conversation.py     # Export markdown/json (audio regenerated on demand)
```

---

## 3. Configuration

### 3.1 Config file
- `config.yaml` (optional) loaded at startup
- Environment variables override config
- Shared constants (e.g., `SUPPORTED_LANGUAGES`) live in `src/kaiwacoach/constants.py`.

### 3.2 Key settings
- `session.language`: `"ja"` or `"fr"` (forced ASR language)
- `models.asr_id`: default Whisper turbo model id
- `models.llm_id`: default Qwen3-14B-8bit
- `models.tts_id`: default Kokoro
- `llm.max_context_tokens`: global cap
- Per-role caps (see Section 5)
- `storage.root_dir`: location for DB (audio cache stored in a session temp dir)
- `tts.voice`, `tts.speed`

---

## 4. Storage Design

### 4.1 SQLite schema (conceptual)
Tables:
- `conversations(id, title, language, created_at, model_metadata_json)`
- `user_turns(id, conversation_id, created_at, input_text, input_audio_path, asr_text, asr_meta_json)`
- `assistant_turns(id, user_turn_id, created_at, reply_text, reply_audio_path, llm_meta_json)`
- `corrections(id, user_turn_id, created_at, errors_json, corrected_text, native_text, explanation_text, prompt_hash)`
- `artifacts(id, conversation_id, kind, path, meta_json)`
Note: `*_audio_path` fields are optional and point to **session-only** cache files.

### 4.2 Audio blobs (Session-only)
- Stored as WAV files under a **session temp dir** (not persisted across restarts).
- Filenames include hash for caching during the session:
  - `user_{sha256}.wav`
  - `assistant_{sha256}.wav`
  - `tts_{sha256}.wav`
- Session cache is deleted on app exit.

### 4.3 DB concurrency
- Single-process app, but Gradio callbacks can overlap.
- Implement a **single-writer queue** in `storage/db.py`:
  - One background worker thread owns the SQLite connection.
  - UI callbacks submit write tasks and await completion.
  - Reads can use separate short-lived connections (or also go through the worker for simplicity).

---

## 5. Orchestrator Pipeline

### 5.1 Turn lifecycle (audio input)
1. Cache raw user audio blob (WAV) in **session audio cache**
2. ASR transcribe (language forced) → `asr_text`
3. Persist `asr_text` + ASR meta
4. LLM conversation reply (schema: `{reply}`)
5. Persist assistant reply text + LLM meta
6. LLM correction suite (errors/corrected/native/explanation) OR combined structured call (implementation choice)
7. Persist correction entity linked to `user_turn_id`
8. If session language is Japanese (or reply contains JP), run JP TTS normalisation for assistant reply:
   - Protected-span masking
   - LLM rewrite (temp=0)
   - (Planned mitigation) JP substring invariant check
   - Punctuation/pauses normalisation
   - Persist normalised text and meta
9. TTS generate assistant audio (cached by text hash in session cache)
10. Return UI update with text + audio controls (user + assistant) + correction pane content

Regeneration:
- When viewing history, audio can be regenerated on demand for a **single turn** or **entire conversation**.
- Regenerated audio is written to the session cache only.

Prompt hashing:
- Record the SHA256 hash of each rendered prompt alongside LLM metadata for reproducibility.

### 5.2 Turn lifecycle (text input)
- Same as audio input but skip steps 1–3; `input_text` is stored and used as-is.

### 5.3 Contracts
Each step is a function with explicit inputs/outputs:
- `transcribe(audio_path, language) -> (asr_text, asr_meta)`
- `llm_converse(history, user_text, language) -> (reply, llm_meta)`
- `llm_analyse(user_text, language) -> (errors, corrected, native, explanation, llm_meta)`
- `jp_normalise_for_tts(text) -> (tts_text, norm_meta)`
- `tts_synth(text, voice, speed) -> (wav_path, tts_meta)`  # wav_path is session-only

### 5.4 Context assembly rules
- Conversation prompt history is truncated to:
  - last N turns or last M tokens (configurable)
- Analysis prompts include:
  - only current user turn by default
  - optional: last assistant turn for context when ambiguous

### 5.5 Per-role token caps (initial)
- Conversation: `max_new_tokens=256`
- Error detection: `max_new_tokens=128`
- Correction: `max_new_tokens=128`
- Native reformulation: `max_new_tokens=128`
- Explanation: `max_new_tokens=192`
- JP TTS normalisation: `max_new_tokens=192`

These are conservative defaults; adjust after profiling.

### 5.6 JSON enforcement
- Use Pydantic schemas for each role output
- Fast-fail parse with **one retry max**
- If still invalid:
  - Return a safe fallback (text-only reply) and log raw output

---

## 6. Prompt and Schema Management

- Prompts live in `prompts/*.md` only, never inline.
- A loader reads prompts and interpolates variables.
- Outputs validated via Pydantic models per role.
- Prompt hashes (SHA256 of rendered prompt) stored with each LLM call for reproducibility.

LLM backend integration:
- Implement MLX-LM integration inside `models/llm_qwen.py` (model load + generation hooks).
- Define a lightweight backend protocol (prompt + max_tokens -> text) to keep the
  wrapper testable and decouple model loading from generation.

---

## 7. Japanese TTS Normalisation Details

### 7.1 Protected spans
Mask and restore:
- URLs
- file paths
- code blocks
- email addresses
- markdown links

### 7.2 Planned invariant check (mitigation hook)
- Identify Japanese substrings in original
- Verify they appear byte-identical in output
- If violated:
  - fall back to original text
  - log the diff

### 7.3 Punctuation/pause normalisation
- Insert pauses after Japanese sentence endings (。, ！, ？)
- Normalize repeated punctuation
- Optionally split very long sentences for Kokoro stability

---

## 8. Testing Strategy

### 8.1 Unit tests (early)
- Schema validation tests for all LLM phases
- Prompt rendering tests (no missing variables)
- Storage round-trip tests (turn insert/read)

### 8.2 Golden-file tests (high priority for JP)
- `tests/fixtures/jp_normalisation_cases.json`
- Each case includes:
  - input text
  - expected normalised output OR expected invariants behaviour
- Run on every change to prompts or normalisation logic

### 8.3 Smoke tests
- End-to-end single-turn pipeline (text)
- End-to-end single-turn pipeline (audio) with a short fixture audio sample (session-only audio cache)

---

## 9. Performance and Profiling

- Record per-step timings:
  - ASR
  - LLM converse
  - LLM analyse
  - normalise
  - TTS
- Cache hits/misses for:
  - ASR (audio hash, session-only)
  - LLM (prompt hash)
  - TTS (text hash, session-only)

Target UX (initial):
- Text turn: < 4s total on M1 Pro (best-effort)
- Audio turn: < 7s total on M1 Pro (best-effort)

---

## 10. Implementation Milestones

### Milestone A (MVP functional)
- Storage layer + schema
- ASR wrapper working with forced language
- LLM wrapper with JSON enforcement
- Kokoro TTS wrapper + session-only caching
- Basic orchestrator + minimal Gradio UI

### Milestone B (MVP quality)
- Prompt set stabilised
- JP normalisation with protected spans
- Golden-file tests for JP normalisation

### Milestone C (robustness)
- Single-writer DB queue
- Memory / context truncation rules
- Logging and export tool

---

## 11. Open Decisions (defer until implementation)
- Whether to run analysis as one structured call vs multiple calls
- Whether to normalise assistant reply text only, or also normalise user JP text for replay
- Which fields to include in `model_metadata_json` (ids, revision hashes, quant bits, etc.)
