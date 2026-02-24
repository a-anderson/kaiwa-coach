# Model Integration Learnings

## Purpose

A second goal for this project was to learn how to integrate multiple AI models into a local application in a way that is reliable, testable, and understandable.

This project combines:

- ASR (speech-to-text)
- LLM generation and analysis roles
- TTS synthesis

The focus was not just “make the models run”, but “make the system behave predictably”.

## System Shape (What Was Actually Integrated)

At a high level, each turn follows a pipeline:

1. Capture user input (text or audio) in the UI
2. Persist the raw input and turn metadata
3. Run the LLM conversation reply
4. Run optional correction-related LLM roles
5. Run TTS for the assistant reply
6. Store artefacts and timing information
7. Update the UI with replayable outputs

The project also supports:

- conversation persistence and resume flows
- language switching
- multilingual UI theming
- Japanese TTS normalisation safeguards

## Key Integration Challenges

### 1. LLM outputs are not reliably well-formed

Even with clear prompts, model outputs can include:

- invalid JSON
- extra text (including reasoning-style text) before or after JSON
- repeated JSON objects
- empty responses

This required schema validation and bounded repair/fallback behaviour to prevent silent downstream failures.

### 2. One model, multiple roles

Using one loaded LLM for different tasks (conversation, error detection, correction, explanation, native rewrite, Japanese normalisation) is memory-efficient, but it means each role needs:

- clear prompt boundaries
- role-specific token caps
- schema-specific parsing
- predictable fallbacks

### 3. Audio input is messy in practice

Browser/Gradio audio input sample rates vary (for example 44.1 kHz and 48 kHz), while downstream processing expects a fixed sample rate. This required:

- resampling for processing
- preserving the original raw audio
- handling errors without breaking the user flow

### 4. UI state synchronisation is easy to break

Loading conversations, switching languages, and updating theme/logo/audio widgets all touch shared state. This made callback output ordering and handler sequencing an important part of system correctness.

## What Worked Well

### Schema-first LLM role handling

The most important reliability improvement was validating every role output against an explicit schema. This made bugs visible early and prevented malformed outputs from quietly propagating.
My implementation is not perfect, and occasional output errors still occur, but schema enforcement and fallbacks make them much easier to contain and debug.

### Bounded repair and fallback behaviour

A single repair attempt (rather than unbounded retries) kept the system predictable and easier to reason about.

### Orchestrator-owned turn lifecycle

Keeping sequencing logic in the orchestrator (instead of scattering logic across UI and model wrappers) made it easier to test the system and track behaviour changes.

### Timing instrumentation

Per-stage timing logs made performance work much more concrete. Instead of guessing where latency came from, changes could be made and measured.

### Strong test coverage on integration edges

Many bugs were caused by interactions between components rather than isolated functions. Tests that covered these edges (UI callback output shape, correction pipelines, conversation loading, audio flows) made bugs much easier to identify and resolve.

## Failures and Fixes That Were Particularly Useful Learning Experiences

### Native rewrite output silently missing

Issue:

- correction and explanation outputs worked
- native rewrite often appeared blank in the UI

Cause:

- native rewrite path was less resilient to malformed/prefixed JSON than other correction roles

Fix:

- switched native rewrite to use the more robust generation/parsing path with repair/fallback handling
- added regression coverage

Lesson:

- consistency across similar role pipelines matters
- small differences in parsing logic can create “works in some tiles but not others” bugs

### Audio submission failures due to sample rate mismatch

Issue:

- recording and playback worked in the UI
- sending audio failed with sample-rate validation warnings

Fix:

- support resample path for processing
- persist raw input audio separately
- add tests for raw-audio persistence and rate handling

Lesson:

- model integration problems are often input-format problems rather than model problems

### Conversation load + language sync edge cases

Issue:

- conversation load sometimes required multiple clicks when conversation language differed from current session language

Fix:

- tighten callback sequencing and language sync logic
- add test coverage for load and language state interactions

Lesson:

- UI state transitions deserve the same engineering discipline as backend logic

## Performance Learnings

### Multi-role correction pipelines dominate latency

In practice, a single user turn can trigger several LLM calls. The conversation reply may feel fast enough, while correction roles add significant delay.

This made two strategies especially valuable:

- reducing token caps for correction roles
- making correction features optional/toggleable in the UI

### Instrumentation changes the quality of optimisation work

Once timing logs were available by stage, performance conversations became much clearer:

- which stage is slow?
- is this a model issue or a pipeline issue?
- did the token-cap change actually help?

This was much better than optimising based on intuition.

## Determinism and Safety Learnings

### “Mostly works” is not enough for model pipelines

If model outputs drive further automation, validation and fallbacks are essential. Small parsing failures otherwise become intermittent product bugs.

### Preserving raw inputs is useful beyond debugging

Keeping raw user text and raw audio supports:

- reproducibility
- debugging
- future analysis
- safer reprocessing paths

### Japanese TTS needs explicit safeguards

Japanese normalisation is not just formatting. The invariant checks and fallback behaviour were important to avoid accidental corruption of user text before TTS.

## Testing Strategy That Worked in This Project

The most useful testing split was:

- unit tests for helpers and schema parsing
- integration tests for orchestrator/model paths
- UI callback tests for Gradio wiring and output ordering
- smoke tests for real model availability and basic runtime behaviour

This mix caught:

- logic bugs
- wiring bugs
- framework version issues
- regressions introduced by refactors

## What I Would Improve Next

- Add clearer automated evaluation snapshots for quality/latency across releases
- Expand model comparison notes (quality vs latency vs memory)
- Add a lightweight benchmarking script for repeatable latency checks
- Continue simplifying UI callback wiring as features grow

## Summary

The main lesson from integrating ASR, LLM, and TTS in one local app is that reliability comes less from any single model and more from the engineering around them: schema validation, explicit fallbacks, persistent storage, clear orchestration, and tests that target integration boundaries.
