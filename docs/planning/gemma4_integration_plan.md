# Gemma 4 Integration Plan

**Status**: Planning only — no implementation started  
**Branch**: `feat/gemma4-integration-planning`  
**Date**: 2026-04-11

## Goal

Add Gemma 4 (via MLX-LM) as an alternative LLM backend to kaiwa-coach, either replacing or coexisting with the existing Qwen3 models. The Qwen3 implementation must remain working.

An Ollama backend was also discussed as a separate option; see the [Ollama Backend](#ollama-backend-option) section.

---

## The Gemma 4 Model Family (mlx-community)

All models are available at: https://huggingface.co/collections/mlx-community/gemma-4

Four size tiers, each with IT (instruction-tuned) and base variants, across multiple quantisations (bf16, 8bit, 6bit, 5bit, 4bit, mxfp8, mxfp4, nvfp4):

| Variant | Architecture | Total params | Active params (inference) |
|---|---|---|---|
| `gemma-4-e2b` | Dense | 2B | 2B |
| `gemma-4-e4b` | Dense | 4B | 4B |
| `gemma-4-26b-a4b` | MoE | 25.2B | 3.8B |
| `gemma-4-31b` | Dense | 31B | 31B |

Only `-it` (instruction-tuned) variants are relevant for this project.

**Recommended starting point**: `gemma-4-e4b-it` — lowest implementation risk (see below).  
**Higher quality option**: `gemma-4-26b-a4b-it` — requires additional output handling (see below).

---

## How the Current LLM Stack Works

Understanding this is essential before making changes.

### Stack layers (bottom to top)

1. **`MlxLmBackend`** (`models/llm_qwen.py`): Loads any model via `mlx_lm.load()`, generates via `mlx_lm.generate()`. Completely model-agnostic — no Gemma-specific changes needed here.

2. **`QwenLLM`** (`models/llm_qwen.py`): Wraps `MlxLmBackend` with context-limit enforcement, per-role token caps, hash-based in-memory caching, and JSON enforcement. Contains Qwen3-specific logic (see below).

3. **`factory.py`** (`models/factory.py`): Routes config model IDs to the correct wrapper. The comment explicitly says "add routing branches here as new backends are integrated."

4. **`json_enforcement.py`** (`models/json_enforcement.py`): Extracts and validates JSON from raw LLM output. Contains a regex to strip `<think>...</think>` tags before parsing.

5. **`config/models.py`** (`config/models.py`): Defines model ID constants and `SUPPORTED_MODELS` allowlist. Any new model ID must be added here or `_validate_config()` in `settings.py` will reject it at startup.

### Qwen3-specific logic in `QwenLLM`

Two pieces of logic are specific to Qwen3 and do not apply to Gemma 4:

**`_NO_THINK_ROLES`** (line ~235): Appends `<think>\n\n</think>` to the prompt for certain roles. This signals to Qwen3 that the reasoning phase is complete and it should output the answer directly, avoiding wasted token budget. Gemma 4 uses a completely different mechanism — this suffix does nothing useful for Gemma 4 and should not be applied.

**`extra_eos_tokens = ["}"]`** (line ~257): For the `conversation` role only, `}` is added as a stop token to reduce Qwen3's tendency to generate trailing content after the JSON object. May or may not be needed for Gemma 4 — requires empirical testing.

### How prompts are passed to models

Prompts in `prompts/*.md` are plain text (no chat template tokens). They are passed as raw strings to `mlx_lm.generate()`, which tokenises them directly — the chat template is **not** applied in Python code. This means models are prompted in completion style, not chat style. This currently works for Qwen3 and should work for Gemma 4, but JSON output reliability needs empirical validation.

### JSON extraction

`extract_first_json_object()` in `json_enforcement.py`:
```python
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(cleaned)  # must start with valid JSON
    ...
```

`raw_decode` requires the string to **begin** with valid JSON after stripping. This is critical for the Gemma 4 26B-A4B model (see below).

---

## Key Gemma 4 Implementation Findings

### The 26B-A4B model always generates thought tags

Source: https://huggingface.co/google/gemma-4-26B-A4B-it

The model page explicitly states: *"Thinking disabled still generates tags (26B A4B behavior)"*

The thought output format is:
```
<|channel>thought
...reasoning content...
<channel|>
{"reply": "..."}
```

This **breaks `extract_first_json_object()`** because:
1. `_THINK_TAG_RE` matches `<think>...</think>` (Qwen3 format) — it does not match `<|channel>thought...<channel|>`
2. After stripping (nothing), `raw_decode` hits `<|channel>` at position 0 and fails
3. Every LLM call falls through to the repair path; repair also fails; turn fails safe

**Fix required**: Add a Gemma-specific pattern to strip `<|channel>thought...<channel|>` blocks before `raw_decode`. This change is in `json_enforcement.py`.

### The e4b model likely does not have this problem

The model page phrases the always-on thought tags as specific to the 26B-A4B model. `e4b-it` likely produces clean output. This makes `e4b-it` the lower-risk starting point.

> **Action needed**: Verify by checking https://huggingface.co/google/gemma-4-e4b-it before implementing.

### Recommended temperature differs

Google recommends `temperature=1.0, top_p=0.95, top_k=64` for Gemma 4. The app defaults to `conversation_temperature=0.7` (from Qwen3 tuning). Not a breaking issue but worth documenting in `config.example.yaml` as a per-model note.

---

## Recommended Implementation Approach

### Option chosen: New `GemmaLLM` wrapper (not generalising `QwenLLM`)

Keep `llm_qwen.py` entirely untouched. Write a new `llm_gemma.py` with a `GemmaLLM` class that reuses `MlxLmBackend` but omits Qwen3-specific logic. Route in `factory.py` based on model ID prefix.

**Rationale**: Zero risk to the working Qwen3 path. Duplication is minimal since `MlxLmBackend` handles all actual generation. Fits the existing factory pattern. Alternative (generalising `QwenLLM` into `MlxLLM`) was considered but rejected — the risk/reward for touching working code is poor.

### Files to change

| File | Change |
|---|---|
| `models/llm_gemma.py` | **New file**: `GemmaLLM` class using `MlxLmBackend`, no `_NO_THINK_ROLES`, Gemma-appropriate EOS handling |
| `models/json_enforcement.py` | Add Gemma 4 thought-tag stripping pattern (required for 26B-A4B; harmless for e4b) |
| `models/factory.py` | Import `GemmaLLM`; add routing branch — if `llm_id` starts with `mlx-community/gemma-4`, return `GemmaLLM(...)` |
| `config/models.py` | Add `GEMMA4_*` model ID constants; add to `SUPPORTED_MODELS["llm"]` |
| `tests/test_app_startup.py` | Config wiring tests: Gemma 4 model IDs load cleanly; unsupported IDs still rejected |
| `README.md` | Add Gemma 4 to the LLM variant table |
| `config.example.yaml` | Add Gemma 4 model ID options with temperature note |

### What does NOT change

- `MlxLmBackend` — already model-agnostic
- `prompts/*.md` — stable unless prompt reliability testing reveals issues
- `orchestrator.py`, storage, ASR, TTS — untouched
- `settings.py` config loading — works as-is once `SUPPORTED_MODELS` is updated

---

## Remaining Unknowns (require a running model to resolve)

1. **JSON output reliability**: Does Gemma 4 IT reliably return bare JSON with no preamble or markdown fences when prompted in completion style? If not, prompts in `prompts/*.md` may need adjustments (stronger JSON-only directives). This is the highest-risk unknown.

2. **Token cap calibration**: `LLMRoleCaps` defaults (`conversation=256`, `jp_tts_normalisation=192`, `detect_and_correct=96`, `explain_and_native=144`) were tuned for Qwen3's tokenizer. Gemma 4 tokenises differently — these may be tight or loose and should be verified empirically.

3. **e4b thought tag behaviour**: Confirm the e4b model does not generate `<|channel>thought...<channel|>` output before treating `json_enforcement.py` changes as optional for that model.

4. **`}` EOS token heuristic**: Whether `extra_eos_tokens=["}"]` helps or hurts Gemma 4 output quality is unknown. Test with and without.

---

## Ollama Backend Option

A separate (independent) option discussed: replace `MlxLmBackend` with an `OllamaBackend` that calls Ollama's local HTTP API (`http://localhost:11434/api/generate`).

### Why it's viable

The `LLMBackend` protocol is minimal:
```python
class LLMBackend(Protocol):
    def generate(self, prompt: str, max_tokens: int,
                 extra_eos_tokens: list[str] | None = None,
                 temperature: float = 0.0) -> str: ...
```
An `OllamaBackend` just implements this with an HTTP POST. Everything above it (context limits, caching, JSON enforcement, orchestrator) works unchanged.

Ollama handles the chat template internally, so raw prompt strings should work.

`extra_eos_tokens` maps to Ollama's `stop` sequences parameter — functionally equivalent.

### Offline-first constraint

The project's "offline-first" rule is about privacy (no cloud APIs, no external data). Calling `localhost:11434` is acceptable under this interpretation — confirmed by the project owner.

### Performance trade-off vs MLX-LM

**MLX-LM is faster on Apple Silicon** (~20–40% more tokens/sec). Key reasons:
- MLX was built by Apple specifically for Apple Silicon's unified memory architecture
- Ollama uses llama.cpp + Metal — fast but cross-platform, not Apple-Silicon-specific
- Ollama adds per-call HTTP/IPC overhead; with this app's short outputs (96–256 token caps), fixed overhead is a larger fraction of total time
- Each turn can involve up to 3 LLM calls (conversation + detect_and_correct + explain_and_native), so the gap compounds

**Conclusion**: Ollama backend is viable if sharing a single model installation across multiple tools is a priority. MLX-LM is the better fit if turn latency matters.

### Ollama implementation scope (if pursued)

| File | Change |
|---|---|
| `models/llm_ollama.py` | New file: `OllamaBackend` implementing `LLMBackend` via HTTP; ~50 lines |
| `models/factory.py` | Add routing branch for Ollama model IDs (e.g. `ollama:gemma4:27b` naming convention) |
| `config/models.py` | Add Ollama model ID constants; expand `SUPPORTED_MODELS["llm"]` |
| `tests/test_app_startup.py` | Config wiring tests |

Operational dependency: Ollama daemon must be running before the app starts. A startup health-check (GET `http://localhost:11434`) would give a clear error rather than a silent failure.

---

## Suggested Next Steps

1. Check the `e4b-it` model page to confirm thought-tag behaviour
2. Start with `gemma-4-e4b-it-8bit` as the lowest-risk first target
3. Implement `GemmaLLM` + factory routing + config changes
4. Run smoke tests against a live model — focus on JSON output reliability and token cap calibration
5. If 26B-A4B is also wanted: add thought-tag stripping to `json_enforcement.py`, then validate
6. Decide separately whether to pursue the Ollama backend
