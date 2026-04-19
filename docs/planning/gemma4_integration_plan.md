# Gemma 4 Integration Plan

**Status**: ✅ Both PRs fully implemented and manually verified  
**Branch**: `feat/gemma4-integration-planning`  
**Last updated**: 2026-04-19

---

## Goal

Add Gemma 4 (via MLX-LM and/or Ollama) as an alternative LLM backend to kaiwa-coach, while keeping the existing Qwen3 path working identically.

The work was split into two sequential pull requests:

1. **PR 1 — Backend/model separation** ✅ **Done**: Refactored the LLM stack so the execution backend (MLX, Ollama) is a first-class config concept, separate from the model family (Qwen3, Gemma 4). No behaviour change — Qwen3+MLX default works identically.
2. **PR 2 — Gemma 4** ✅ **Done**: Added `GemmaLLM` wrapper, implemented `OllamaBackend`, added Gemma 4 model IDs, added thought-tag suppression for 26B-A4B. Qwen3 path untouched.

---

## Codebase Context

Read this section before touching any code. It explains the parts of the codebase that this work touches.

### Relevant files (post-implementation)

| File | Role |
|---|---|
| `src/kaiwacoach/models/llm_backends.py` | `LLMBackend` protocol, `MlxLmBackend`, `OllamaBackend` — all implemented |
| `src/kaiwacoach/models/llm_qwen.py` | `QwenLLM` wrapper — imports `LLMBackend`/`MlxLmBackend` from `llm_backends.py` |
| `src/kaiwacoach/models/llm_gemma.py` | `GemmaLLM` wrapper — implemented, no Qwen3-specific logic |
| `src/kaiwacoach/models/factory.py` | `build_llm()` — backend routing + family auto-detection, both families wired |
| `src/kaiwacoach/models/json_enforcement.py` | `_GEMMA_CHANNEL_RE` added for Gemma channel-tag stripping |
| `src/kaiwacoach/models/protocols.py` | `LLMProtocol` — unchanged |
| `src/kaiwacoach/config/models.py` | Gemma 4 MLX constants + `SUPPORTED_BACKENDS` + `SUPPORTED_LLM_MODELS` |
| `src/kaiwacoach/settings.py` | `ModelsConfig.llm_backend`, `KAIWACOACH_MODELS_LLM_BACKEND` env var |
| `config.example.yaml` | `llm_backend` field + Gemma 4 / Ollama examples documented |
| `README.md` | Gemma 4 model table, Ollama backend instructions added |

### The LLM stack (post-implementation)

```
factory.py
  └─ build_llm(config)
       1. detect_family(llm_id)            → "qwen3" or "gemma4"
       2. backend_name == "mlx"  → MlxLmBackend(llm_id)
          backend_name == "ollama"
            → OllamaBackend.check_available()
            → OllamaBackend(llm_id, suppress_thinking=(family == "gemma4"))
       3. family == "qwen3"  → QwenLLM(...)
          family == "gemma4" → GemmaLLM(..., backend_label="mlx_lm"|"ollama")
```

### Family auto-detection

Model family is inferred from `llm_id` prefix. Explicit `llm_family` config key was considered and rejected — auto-detection keeps config minimal and the prefix rules are unambiguous for the supported models. Unknown prefixes raise `ValueError` at startup.

| `llm_id` prefix | Detected family |
|---|---|
| `mlx-community/Qwen3-` | `qwen3` |
| `qwen3:` | `qwen3` |
| `mlx-community/gemma-4-` | `gemma4` |
| `gemma4:` | `gemma4` |

### Model ID validation strategy

MLX model IDs are validated against an explicit allowlist at startup. Ollama model IDs are **not** validated — Ollama gives a clear error at model-load time if the model isn't installed.

### Key design decisions

- **`suppress_thinking`**: `OllamaBackend(suppress_thinking=True)` adds `"think": false` to the Ollama payload. Set for all Gemma 4 Ollama models — required for 26B-A4B (see Resolved Unknowns #3), no-op for e4b.
- **No `_NO_THINK_ROLES`** in `GemmaLLM`: Qwen3's `<think>\n\n</think>` suffix is meaningless for Gemma 4.
- **No `extra_eos=["}"]`** in `GemmaLLM`: Present in `QwenLLM` for Qwen3's trailing-content tendency; not yet tested for Gemma 4 (see Open Unknowns #4).
- **`backend_label`** param on `GemmaLLM`: `"mlx_lm"` or `"ollama"` — passed explicitly from factory rather than inspecting the backend type.
- **No inheritance**: `GemmaLLM` does not inherit from `QwenLLM`. Shared logic is small enough that duplication is preferable to coupling.

---

## Resolved Unknowns

1. ✅ **JSON output reliability** (resolved 2026-04-19, `gemma4:e4b` via Ollama): All four roles (`conversation`, `detect_and_correct`, `explain_and_native`, `jp_tts_normalisation`) produce clean JSON with no preamble or markdown fences. Prompts require no changes.

2. **Token cap calibration**: Defaults (`conversation=256`, `jp_tts_normalisation=192`, `detect_and_correct=96`, `explain_and_native=144`) not formally verified against the Gemma 4 tokenizer. Empirically confirmed working for both `gemma4:e4b` and `gemma4:26b` via Ollama (2026-04-19) — all roles produce complete, valid responses at these caps.

3. ✅ **Thought-tag behaviour** (resolved 2026-04-19):
   - `gemma4:e4b`: No `<|channel>thought...<channel|>` tags produced — output is clean. `_GEMMA_CHANNEL_RE` is harmless (never matches).
   - `gemma4:26b`: Ollama does **not** strip thought blocks. They consume the entire token budget before the JSON answer is generated, causing blank responses. Fix: `"think": false` in the Ollama payload via `suppress_thinking=True`. `_GEMMA_CHANNEL_RE` is not the fix for this — it requires the closing `<channel|>` tag which never appears in truncated output. All roles confirmed working after fix.

4. ✅ **Ollama startup health check** (resolved): `OllamaBackend.check_available()` implemented; raises `RuntimeError` with a clear message if the Ollama daemon is not running. Called from `build_llm()` before any model is instantiated.

5. ✅ **`}` EOS token for Gemma 4** (resolved 2026-04-19, `gemma4:e4b` and `gemma4:26b` via Ollama): No trailing content or non-Japanese punctuation observed. `GemmaLLM` correctly omits `extra_eos=["}"]` — the stop token is Qwen3-specific and not needed for Gemma 4.

---

## Open Unknowns

None — all unknowns resolved.

---

## Resolved Unknowns (all)

5. ✅ **Recommended temperature** (resolved 2026-04-19): Google recommends `temperature=1.0` for Gemma 4. Default `0.7` produces natural responses with no observed quality issues for either `gemma4:e4b` or `gemma4:26b`. No change to the default — if users want to experiment, `config.example.yaml` documents the `conversation_temperature` option.

---

## Config shape

```yaml
models:
  llm_backend: "mlx"                        # "mlx" or "ollama"; default "mlx"
  llm_id: "mlx-community/Qwen3-14B-8bit"   # backend-specific model path/name
```

Ollama examples:
```yaml
models:
  llm_backend: "ollama"
  llm_id: "gemma4:e4b"    # or "gemma4:26b", "qwen3:14b", etc.
```

Env vars:
- `KAIWACOACH_MODELS_LLM_BACKEND` — `"mlx"` or `"ollama"`
- `KAIWACOACH_MODELS_LLM_ID` — existing, unchanged

---

## What Does NOT Change

- `MlxLmBackend` internals — already model-agnostic
- `QwenLLM` — untouched by both PRs
- `prompts/*.md` — stable; empirically validated for Gemma 4
- `orchestrator.py`, storage, ASR, TTS — untouched
- The `LLMProtocol` in `protocols.py` — unchanged
- `settings.py` config loading logic (three-layer merge) — only new fields added
