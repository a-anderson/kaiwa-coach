# KaiwaCoach – Implementation Checklist (Derived from PRD v2)

This checklist translates **PRD v2** into concrete, actionable implementation tasks.
It is intended for a **solo developer** working locally on Apple Silicon and is also suitable for use by an LLM coding assistant.

---

## Legend

- ⬜ Not started
- ⏳ In progress
- ✅ Done
- 🔁 Revisit later

---

## 0. Project Setup (Foundational)

✅ Create repository structure per TDD  
✅ Set Python version to **3.11.x (tested baseline)**  
✅ Create virtual environment
✅ Run setup script to install UniDic assets (`python -m unidic download`)  
✅ Add dependency management (`pyproject.toml` or `requirements.txt`) including:

- `mlx`
- `mlx-lm`
- MLX Whisper package
- MLX Kokoro package
- `gradio`
- `pydantic`
- `soundfile` / `scipy`
- stdlib `sqlite3`

✅ Add `.gitignore` (models, audio blobs, DB files)  
✅ Add `README.md` referencing PRD v2 and TDD

---

## 1. Configuration Layer

✅ Implement `config.py`  
✅ Define defaults:

- Session language (`ja` / `fr`)
- ASR model ID
- LLM model ID (default 8-bit)
- TTS model ID
- Storage root path
- TTS voice and speed
- Per-role token limits

✅ Support environment variable overrides  
✅ Validate configuration at startup

---

## 2. Storage Layer

### 2.1 Database Schema

⬜ Create `schema.sql`  
⬜ Tables:

- `conversations`
- `user_turns`
- `assistant_turns`
- `corrections`
- `artifacts`

⬜ Add schema versioning

---

### 2.2 SQLite Access

⬜ Implement `storage/db.py`  
⬜ Enforce single-writer queue  
⬜ Ensure all writes go through one connection  
⬜ Safe concurrent reads for Gradio callbacks

---

### 2.3 Audio Blob Storage

⬜ Implement `storage/blobs.py`  
⬜ Deterministic paths:

- per conversation
- per turn
- hash-based filenames

⬜ WAV save/load helpers  
⬜ Enforce sample rate consistency

---

## 3. ASR Module

⬜ Implement `models/asr_whisper.py`  
⬜ Load `mlx-community/whisper-large-v3-turbo-asr-fp16`  
⬜ Force language per session  
⬜ Preserve English words in mixed-language utterances

⬜ Return:

- transcript
- ASR metadata

⬜ Cache ASR results by audio hash  
⬜ (Planned) Log confidence proxies

---

## 4. LLM Core Wrapper

### 4.1 Base Wrapper

⬜ Implement `models/llm_qwen.py`  
⬜ Load `mlx-community/Qwen3-14B-8bit` by default (memory-safe, latency-optimised)  
⬜ Support optional BF16 mode  
⬜ Enforce:

- max context tokens
- per-call max tokens

⬜ Capture timing and metadata

---

### 4.2 JSON Schema Enforcement

⬜ Implement first-valid-object JSON extraction
⬜ Ignore/log trailing content

⬜ Define Pydantic schemas for:

- Conversation reply
- Error detection
- Corrected sentence
- Native reformulation
- Explanation
- JP TTS normalisation

⬜ Strict JSON parsing  
⬜ One retry max via repair prompt  
⬜ Safe fallback on failure

---

## 5. Prompt Management

⬜ Create `prompts/` directory  
⬜ Add prompt files:

- `conversation.md`
- `detect_errors.md`
- `correct_sentence.md`
- `native_rewrite.md`
- `explain.md`
- `jp_tts_normalise.md`
- `repair_json.md`

⬜ Implement prompt loader:

- markdown read
- variable interpolation
- SHA256 hash generation

⬜ Store prompt hash per LLM call

---

## 6. Japanese TTS Normalisation

### 6.1 Protected Spans

⬜ Implement masking for:

- URLs
- file paths
- emails
- code blocks
- markdown links

---

### 6.2 Katakana Conversion

⬜ Implement LLM-based rewrite (temp = 0)  
⬜ Rewrite only non-Japanese spans

---

### 6.3 Invariant Mitigation Hooks

⬜ Detect Japanese substrings  
⬜ Verify byte-identical preservation  
⬜ Fallback + log on violation

---

### 6.4 Punctuation / Pause Normalisation

⬜ Normalize sentence breaks  
⬜ Normalize repeated punctuation  
⬜ Insert pauses for Kokoro

---

## 7. TTS Module

⬜ Implement `models/tts_kokoro.py`  
⬜ Load `mlx-community/Kokoro-82M-bf16`  
⬜ Generate WAV output  
⬜ Cache TTS by `(text, voice, speed)`

---

## 8. Conversation Orchestrator

⬜ Implement `orchestrator.py`

### Text Turn Flow

⬜ Persist `UserTurn`  
⬜ Generate assistant reply  
⬜ Persist `AssistantTurn`  
⬜ Generate corrections  
⬜ Persist `Correction`  
⬜ Normalise for TTS (JP)  
⬜ Generate TTS  
⬜ Persist audio

---

### Audio Turn Flow

⬜ Persist raw audio  
⬜ Run ASR  
⬜ Persist transcript  
⬜ Continue text flow

---

### Orchestrator Rules

⬜ Schema validation at every step  
⬜ Persist intermediates before side-effects  
⬜ Graceful degradation on failure

---

## 9. Gradio UI

⬜ Implement `app.py`  
⬜ UI elements:

- Chat transcript
- Text input
- Microphone input
- Send button
- Per-turn audio playback
- Corrections panel

⬜ Session reset support  
⬜ Safe interaction with DB queue

---

## 10. Caching and Performance

⬜ ASR cache  
⬜ LLM output cache  
⬜ TTS cache

⬜ Per-step timing logs

---

## 11. Testing

### Schema Tests

⬜ Validate all LLM schemas  
⬜ Test repair prompt

---

### Japanese Normalisation Golden Tests (High Priority)

⬜ Create JP/EN mixed fixtures  
⬜ Assert katakana conversion or invariant fallback

---

### Storage Tests

⬜ DB insert/read round-trip  
⬜ Audio save/load tests

---

## 12. Resource and Stability

⬜ Enforce context truncation  
⬜ Enforce token caps per role  
⬜ Periodic memory logging

⬜ Manual session reset control

---

## 13. MVP Exit Criteria

⬜ Spoken JP conversation works end-to-end  
⬜ Corrections and native phrasing displayed  
⬜ Kokoro pronounces mixed JP/EN correctly  
⬜ Conversations persist across restarts  
⬜ Stable operation within ~22–26 GB RAM

---

## 14. Post-MVP (Optional)

🔁 Add support for other languages
🔁 ASR confidence-based UX  
🔁 Pronunciation scoring  
🔁 Shadowing mode  
🔁 Anki export  
🔁 Desktop packaging
