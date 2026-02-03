# KaiwaCoach – Product Requirements Document (PRD v2)

**Revised after senior ML review**

## Summary / Purpose

**Version note:** v2.1 incorporates validated implementation learnings (Python 3.11 environment, MLX-LM sampler usage, robust JSON extraction, Japanese TTS dependency preflight, and default 8-bit LLM selection).

**KaiwaCoach** is a fully offline, local-first conversational language coaching application for **Japanese and French**, designed to help learners improve **spoken fluency, correctness, and naturalness** through real dialogue, automatic speech recognition, structured error detection, correction, native reformulation, and high-quality text-to-speech playback.

The application runs entirely on **Apple Silicon macOS**, requires **no internet connection**, and prioritises **high-quality Japanese handling**, including accurate pronunciation of loanwords (外来語 / gairaigo), acronyms, and mixed-language utterances.

---

## 1. Goals and Non-Goals

### 1.1 Goals (Non-Negotiable)

- Fully offline operation on macOS (Apple Silicon)
- Python **3.11.x (recommended, tested)**
- High-quality **Japanese** ASR and TTS
- Support for **text and voice** conversations
- Persistent conversation history including:
    - Raw user text input
    - Raw user audio input
    - ASR transcript
    - Assistant reply text
    - Assistant reply audio
    - Corrections and native reformulations
- Explicit feedback per user turn:
    - Errors (if any)
    - Corrected form
    - More natural / native phrasing
    - Concise explanation
- Deterministic, debuggable pipelines
- Simple Python-native UI (Gradio)
- Modular, maintainable architecture suitable for open-source use

### 1.2 Non-Goals

- Cloud inference or SaaS deployment
- Real-time streaming ASR/TTS (batch per turn is acceptable)
- Gamification or curriculum design
- Broad device support beyond Apple Silicon macOS

---

## 2. Target User

- Primary: technically proficient individual language learner (developer / data scientist)
- Secondary: advanced learners prioritising **natural spoken Japanese**
- Usage pattern:
    - Short daily conversations
    - Review and replay of past turns
    - Occasional deep dives into corrections

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────┐
│              UI (Gradio)                 │
│  - Chat transcript                       │
│  - Text input                            │
│  - Microphone input                      │
│  - Per-turn audio playback               │
│  - Corrections / Native phrasing panel   │
└───────────────────┬──────────────────────┘
                    │
┌───────────────────▼──────────────────────┐
│        Conversation Orchestrator         │
│  - Turn lifecycle                        │
│  - Context assembly (per task)           │
│  - Model routing (logical roles)         │
│  - Caching & retries                     │
│  - Failure handling                      │
└───────────────┬───────────────┬──────────┘
                │               │
┌───────────────▼────────┐ ┌────▼────────────────────┐
│ ASR Engine             │ │ Core LLM Engine         │
│ Whisper large-v3 turbo │ │ Qwen3-14B (8bit default)│
│ - Language forced      │ │ - Conversation          │
│ - Confidence proxies   │ │ - Error detection       │
└───────────────┬────────┘ │ - Correction            │
                │          │ - Native reformulation  │
                │          │ - JP TTS normalisation  │
                │          └──────┬──────────────────┘
                │                 │
                │          ┌──────▼───────────────────────┐
                │          │ Text Pre-TTS Normalisation   │
                │          │ - Katakana conversion (JP)   │
                │          │ - Punctuation / pauses       │
                │          │ - Invariant checks           │
                │          └──────┬───────────────────────┘
                │                 │
┌───────────────▼─────────────────▼───────────────────────┐
│                 TTS Engine                              │
│         Kokoro-82M-bf16 (JP / FR)                       │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│                 Storage Layer                           │
│  - SQLite (turns, corrections, metadata)                │
│  - Filesystem (WAV audio blobs)                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Core Technology Stack

### 4.1 Speech-to-Text (ASR)

**Environment note (v2.1):**

- ASR/TTS stack relies on `mlx-audio`.
- Japanese TTS requires `phonemizer-fork` (not `phonemizer`) and UniDic assets installed via `python -m unidic download`.

**Model**

- `mlx-community/whisper-large-v3-turbo-asr-fp16`

**Rationale**

- Strong Japanese accuracy
- Much lower latency than full large-v3
- MLX-native, no PyTorch overhead

**Language Handling**

- Language is **forced per session** (`ja` or `fr`)
- Users may freely include **English words** if they do not know the target-language term
- Code-switching is preserved in ASR output

**Planned (non-MVP)**

- Logging of confidence proxies (avg logprob, compression ratio)
- Low-confidence ASR path influencing correction tone

---

### 4.2 Core LLM (Text Processing)

**Implementation note (v2.1):**

- LLM generation uses sampler-based control (`mlx_lm.sample_utils.make_sampler`).
- JSON outputs are extracted via first-valid-object parsing (`JSONDecoder.raw_decode`).
- Trailing content or multiple objects are tolerated and logged; one repair retry max.

**Default Model**

- `mlx-community/Qwen3-14B-8bit`

**Optional High-Accuracy Mode**

- `mlx-community/Qwen3-14B-bf16`

**Library**

- `mlx-lm`

**Design Choice**

- Single loaded model, logically split into multiple roles
- Explicit token limits and context windows per role

---

### 4.3 Single-Model, Multi-Role LLM Usage

| Role                 | Characteristics                     |
| -------------------- | ----------------------------------- |
| Conversation         | Moderate temperature, short context |
| Error detection      | Low temperature, minimal context    |
| Correction           | Deterministic, minimal output       |
| Native reformulation | Slightly creative, bounded          |
| JP TTS normalisation | Fully deterministic                 |

The model is accessed via **adapters**, making future replacement possible without architectural changes.

---

## 5. Japanese TTS Normalisation

### Purpose

Ensure Kokoro TTS pronounces Japanese text correctly when Latin words, acronyms, or numbers appear.

### Invariant

All Japanese substrings **must remain byte-identical** after normalisation.

### Planned Mitigations

- Diff-based invariant checks
- Fallback to original text on violation
- Future two-stage span-based rewrite

---

## 6. Text-to-Speech (TTS)

**Model**

- `mlx-community/Kokoro-82M-bf16`

**Features**

- High-quality Japanese pronunciation
- Offline MLX inference
- Configurable voices and speeds

**Pre-TTS Normalisation**

- Katakana conversion (JP)
- Punctuation and pause handling

---

## 7. Conversation Orchestrator

The orchestrator:

- Coordinates pipeline steps
- Assembles context per task
- Enforces schema-validated I/O
- Persists intermediate artefacts before side effects

Each step is treated as a **pure function**.

---

## 8. Data Model (Revised)

- **UserTurn**: raw input, ASR transcript
- **AssistantTurn**: reply text and audio
- **Correction**: errors, corrected form, native form, explanation

Corrections are first-class entities linked to user turns.

---

## 9. Failure Modes and Mitigations

### Language correctness

- Preserve original user text permanently
- Track politeness level per conversation
- Do not over-normalise English nouns

### Resource stability

- Default to 8-bit LLM
- Context truncation rules
- Planned model reload hooks

---

## 10. Caching Strategy

- ASR cache by audio hash
- LLM output cache by (prompt, role)
- TTS cache by (text, voice, speed)

---

## 11. Resource Envelope (Apple Silicon)

| Component      | Approx RAM |
| -------------- | ---------- |
| Qwen3-14B-8bit | ~15–16 GB  |
| Whisper turbo  | ~2–3 GB    |
| Kokoro         | ~0.5 GB    |
| UI / OS / DB   | ~4–6 GB    |
| **Total**      | ~22–26 GB  |

---

## 12. Development Phases

Phase 1 – Essential  
Phase 2 – Medium  
Phase 3 – Low  
Phase 4 – Future

---

## 13. Guiding Principles

- Japanese correctness over heuristics
- Determinism where possible
- Offline-first and privacy-preserving
- Simplicity over premature generality

---

## 14. Final Statement

This v2.1 revision preserves the full intent and structure of PRD v2 while integrating concrete lessons from early spike implementations, ensuring the document reflects a _known-working_ offline ML stack rather than theoretical assumptions.

KaiwaCoach is a serious, local, conversation-driven language coach designed with discipline around correctness, determinism, and maintainability. This v2 PRD incorporates targeted senior ML feedback while remaining appropriate for a personal, offline-first project.
