# AGENTS.md

## Non‑Negotiable Engineering Rules

- Offline‑first: no network calls at runtime.
- macOS Apple Silicon only; Python 3.11.x target.
- Deterministic pipelines by default; randomness must be explicit, logged, and bounded.
- Persist intermediates before side effects (e.g., save text before TTS).
- Prompts live only in `prompts/*.md` and are never inlined in code.
- All LLM outputs must be schema‑validated; one repair retry max.
- Preserve raw user input (text + audio) permanently.
- Japanese TTS normalisation must not alter Japanese substrings (byte‑identical invariant with fallback).

## Architectural Boundaries

- UI (Gradio) only orchestrates user interaction; no model logic inside UI code.
- Orchestrator owns turn lifecycle and sequencing; steps are pure functions.
- Model wrappers (`models/*`) expose typed contracts only; no DB or UI dependencies.
- Storage layer (SQLite + WAV blobs) is the single source of truth; no hidden caches.
- Single‑writer DB queue is mandatory for writes; reads may be separate connections.
- Text normalisation lives in `textnorm/*`; no normalisation inside model wrappers.

## LLM, Prompts, Schemas

- One loaded LLM, multiple logical roles with explicit token caps per role.
- Outputs are JSON objects validated with Pydantic models per role.
- JSON extraction uses first‑valid‑object parsing; trailing content is ignored and logged.
- On invalid JSON: run at most one repair prompt; then fail safe with a minimal reply.
- Store prompt hashes (SHA256 of rendered prompt) with each LLM call for reproducibility.
- Prompts are rendered via a loader that verifies required variables.

## Environment & Dependency Management (Strict)

- The Python environment is authoritative and pre‑created by the human maintainer.
- The project uses Poetry with Python 3.11.x as the known‑good baseline.
- Do not create, delete, or recreate virtual environments.
- Do not change the Python version.
- Do not run `poetry env use`, `python -m venv`, `pyenv`, or similar commands.
- Do not modify `pyproject.toml`, `poetry.lock`, or dependency versions.
- Do not install or uninstall system packages or Python packages.
- Do not attempt to “fix” dependency issues by altering the environment.
- Assume the Poetry environment already exists.
- Assume all required dependencies are installed.
- Any environment issues will be handled manually by the human maintainer.
- Only modify the environment if the human explicitly requests it using clear language such as:
- “Update the Poetry environment.”
- “Change dependencies.”
- “Recreate the virtual environment.”
- If environment or dependency issues are detected: stop, report the issue clearly, and do not apply changes autonomously.

## Performance and Determinism

- Cache by hash: ASR uses audio hash.
- Cache by hash: LLM uses prompt hash + role.
- Cache by hash: TTS uses text + voice + speed.
- Enforce context truncation and per‑role max token limits.
- Record per‑step timings (ASR, LLM converse/analyse, normalise, TTS).
- Target UX (best‑effort): text turn < 4s, audio turn < 7s on M1 Pro.

## Testing Expectations

- Schema tests for every LLM role and repair prompt path.
- Prompt rendering tests ensure no missing variables.
- Storage round‑trip tests for DB + audio blobs.
- Japanese normalisation golden tests on every change to prompts or normalisation logic.
- End‑to‑end smoke tests for single text turn and single audio turn.
