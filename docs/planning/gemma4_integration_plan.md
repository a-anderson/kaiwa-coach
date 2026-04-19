# Gemma 4 Integration Plan

**Status**: Planning complete — implementation not started  
**Branch**: `feat/gemma4-integration-planning`  
**Last updated**: 2026-04-19

---

## Goal

Add Gemma 4 (via MLX-LM and/or Ollama) as an alternative LLM backend to kaiwa-coach, while keeping the existing Qwen3 path working identically.

The work is split into two sequential pull requests:

1. **PR 1 — Backend/model separation**: Refactor the LLM stack so the execution backend (MLX, Ollama) is a first-class config concept, separate from the model family (Qwen3, Gemma 4). No behaviour change — Qwen3+MLX stays the default and must work identically after this PR.
2. **PR 2 — Gemma 4**: Add `GemmaLLM` wrapper and Gemma 4 model IDs on top of the clean structure from PR 1.

---

## Codebase Context

Read this section before touching any code. It explains the parts of the codebase that this work touches.

### Relevant files

| File | Role in this work |
|---|---|
| `src/kaiwacoach/models/llm_qwen.py` | Contains `LLMBackend` protocol, `MlxLmBackend`, and `QwenLLM`. PR 1 extracts the first two. |
| `src/kaiwacoach/models/factory.py` | `build_llm()` instantiates the LLM. This is where backend routing and family detection live after PR 1. |
| `src/kaiwacoach/models/json_enforcement.py` | Extracts and validates JSON from raw LLM output. Contains `<think>` tag stripping for Qwen3. PR 2 adds Gemma 4 thought-tag stripping here. |
| `src/kaiwacoach/models/protocols.py` | `LLMProtocol` — the interface that all LLM wrappers must satisfy. Do not change this. |
| `src/kaiwacoach/config/models.py` | Model ID constants and `SUPPORTED_MODELS` allowlist. Both PRs add entries here. |
| `src/kaiwacoach/settings.py` | Config loading, `ModelsConfig` dataclass, env var mapping, `_validate_config`. PR 1 adds `llm_backend` here. |
| `config.example.yaml` | User-facing config template. Both PRs add documented options here. |
| `tests/test_app_startup.py` | Config wiring tests. Both PRs add test cases here. |

### The current LLM stack (before PR 1)

```
factory.py
  └─ build_llm(config)
       ├─ MlxLmBackend(config.models.llm_id)   # from llm_qwen.py
       └─ QwenLLM(..., backend=backend)         # from llm_qwen.py
```

`MlxLmBackend` loads any model via `mlx_lm.load()` and generates via `mlx_lm.generate()`. It is completely model-agnostic.

`QwenLLM` wraps any `LLMBackend` with: context-limit enforcement, per-role token caps, hash-based in-memory caching, and JSON enforcement. It also contains two pieces of Qwen3-specific logic:

- **`_NO_THINK_ROLES`** (line 235): Appends `<think>\n\n</think>` to suppress Qwen3's reasoning phase for deterministic roles. This suffix is meaningless for Gemma 4 — do not apply it in `GemmaLLM`.
- **`extra_eos = ["}"]`** (line 257): For the `conversation` role, adds `}` as a stop token to reduce Qwen3's tendency to generate trailing content. Whether this helps Gemma 4 is unknown — leave it out initially and test empirically.

### How prompts reach the model

Prompts in `prompts/*.md` are rendered as plain text and passed directly to `mlx_lm.generate()` (completion style, no chat template). This works for Qwen3 and should work for Gemma 4, but JSON output reliability with Gemma 4 needs empirical validation (see Remaining Unknowns).

### JSON extraction

`extract_first_json_object()` in `json_enforcement.py`:

```python
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(cleaned)   # must start with valid JSON
    ...
```

`raw_decode` requires the string to **begin** with valid JSON after stripping. The Gemma 4 26B-A4B model always produces thought tags in a different format (`<|channel>thought...<channel|>`) that the current regex does not match — see PR 2 for the fix.

### Config loading

`load_config()` in `settings.py` merges three layers: hardcoded defaults → `config.yaml` → env vars (`KAIWACOACH_*`). Env var names follow `KAIWACOACH_<SECTION>_<KEY>`. After PR 1, `KAIWACOACH_MODELS_LLM_BACKEND` is the new env var.

`_validate_config()` checks model IDs against `SUPPORTED_MODELS` in `config/models.py` and raises `ValueError` at startup for any invalid config. Unknown model IDs are caught early, before any model is loaded.

---

## Architecture Design

### The three-layer model

The refactored design introduces a clean three-way separation:

| Layer | What it represents | Examples |
|---|---|---|
| **Backend** | How inference runs | `MlxLmBackend`, `OllamaBackend` |
| **Model family** | Model-specific output quirks (think tags, EOS tokens, temperature) | `QwenLLM`, `GemmaLLM` |
| **Checkpoint** | The specific weights path/name for the backend | `mlx-community/Qwen3-14B-8bit`, `gemma4:e4b` |

After PR 1, `factory.py` orchestrates all three:

```
factory.build_llm(config)
  1. read config.models.llm_backend  → pick backend class
  2. detect_family(config.models.llm_id) → pick wrapper class
  3. instantiate backend with llm_id
  4. instantiate wrapper with backend
```

### Family auto-detection

Model family is inferred from `llm_id` prefix. Explicit `llm_family` config key was considered and rejected — auto-detection keeps config minimal and the prefix rules are unambiguous for the supported models. Unknown prefixes raise `ValueError` at startup with a clear message.

After both PRs, the detection table is:

| `llm_id` prefix | Detected family |
|---|---|
| `mlx-community/Qwen3-` | `qwen3` |
| `qwen3:` | `qwen3` |
| `mlx-community/gemma-4-` | `gemma4` |
| `gemma4:` | `gemma4` |

PR 1 only needs the Qwen3 rules. Gemma 4 rules are added in PR 2.

### Model ID validation strategy

MLX model IDs are validated against an explicit allowlist at startup (finite, curated set). Ollama model IDs are **not** validated — Ollama gives a clear error at model-load time if the model isn't installed. This is the right tradeoff: MLX checkpoints are curated, Ollama models are user-managed.

```python
# config/models.py after PR 1
SUPPORTED_LLM_MODELS: dict[str, frozenset[str] | None] = {
    "mlx": frozenset({LLM_MODEL_ID_4BIT, LLM_MODEL_ID_8BIT, LLM_MODEL_ID_BF16}),
    "ollama": None,  # any model ID accepted; Ollama validates at load time
}
```

### Config shape after both PRs

```yaml
models:
  llm_backend: "mlx"                        # NEW — "mlx" or "ollama"; default "mlx"
  llm_id: "mlx-community/Qwen3-14B-8bit"   # existing — backend-specific model path/name
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

## Remaining Unknowns (require a running model to resolve)

These cannot be answered without empirical testing against a live Gemma 4 model. They are **not** blockers for PR 1.

1. **JSON output reliability (highest risk)**: Does Gemma 4 IT reliably produce bare JSON with no preamble or markdown fences when prompted in completion style? If not, prompts in `prompts/*.md` may need stronger JSON-only directives. Test every role: `conversation`, `detect_and_correct`, `explain_and_native`, `jp_tts_normalisation`.

2. **Token cap calibration**: `LLMRoleCaps` defaults (`conversation=256`, `jp_tts_normalisation=192`, `detect_and_correct=96`, `explain_and_native=144`) were tuned for Qwen3's tokenizer. Gemma 4 tokenises differently — verify these are not too tight or too loose.

3. **e4b thought-tag behaviour**: The 26B-A4B model always generates `<|channel>thought...<channel|>` tags. Verify whether `e4b-it` does the same. If it does not, the `json_enforcement.py` change in PR 2 is still harmless (the pattern simply never matches).

4. **`}` EOS token**: Whether `extra_eos_tokens=["}"]` helps or hurts Gemma 4 conversation output is unknown. Test with and without.

5. **Recommended temperature**: Google recommends `temperature=1.0, top_p=0.95, top_k=64` for Gemma 4. The app default is `0.7` (Qwen3-tuned). Not a breaking issue but worth noting in `config.example.yaml`.

6. **Ollama startup health check**: If Ollama backend is configured but the Ollama daemon is not running, the failure should surface as a clear error at startup rather than a cryptic HTTP connection error mid-turn. PR 2 should add a startup health-check ping to `http://localhost:11434` when `llm_backend = "ollama"`.

---

## PR 1 — Backend / Model Separation

**Scope**: Pure refactor. No new model support. Qwen3+MLX behaviour is identical before and after.  
**Test gate**: `poetry run pytest -q -m "not slow"` must pass with no changes to test count.

### Step 1 — Create `models/llm_backends.py`

Create `src/kaiwacoach/models/llm_backends.py`. Move `LLMBackend` (protocol) and `MlxLmBackend` (class) from `llm_qwen.py` into this file verbatim. Add an `OllamaBackend` stub:

```python
class OllamaBackend:
    """Ollama HTTP backend stub — not yet implemented."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        extra_eos_tokens: list[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError("OllamaBackend is not yet implemented")

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError("OllamaBackend is not yet implemented")
```

`OllamaBackend` satisfies the `LLMBackend` protocol so factory routing can be wired now. It will raise immediately if called, which is the correct behaviour while it is a stub.

### Step 2 — Update `models/llm_qwen.py`

Replace the `LLMBackend` and `MlxLmBackend` definitions with imports:

```python
from kaiwacoach.models.llm_backends import LLMBackend, MlxLmBackend
```

No other changes. `QwenLLM` is untouched.

### Step 3 — Update `config/models.py`

Add `SUPPORTED_BACKENDS` and `SUPPORTED_LLM_MODELS`. Keep existing `SUPPORTED_MODELS` for ASR and TTS validation (those have not changed):

```python
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"mlx", "ollama"})

SUPPORTED_LLM_MODELS: dict[str, frozenset[str] | None] = {
    "mlx": frozenset({LLM_MODEL_ID_4BIT, LLM_MODEL_ID_8BIT, LLM_MODEL_ID_BF16}),
    "ollama": None,   # pass-through; Ollama validates model availability at runtime
}
```

Remove `"llm"` from the existing `SUPPORTED_MODELS` dict (ASR and TTS remain). Update any references to `SUPPORTED_MODELS["llm"]` in `settings.py` to use `SUPPORTED_LLM_MODELS` instead.

### Step 4 — Update `settings.py`

Four changes:

**a. Add `llm_backend` to `ModelsConfig`:**
```python
@dataclass(frozen=True)
class ModelsConfig:
    asr_id: str
    llm_id: str
    llm_backend: str    # NEW — "mlx" or "ollama"
    tts_id: str
```

**b. Update `to_dict()`** to include `"llm_backend": self.models.llm_backend` in the `"models"` section.

**c. Add env var to `_apply_env_overrides`:**
```python
"KAIWACOACH_MODELS_LLM_BACKEND": (("models", "llm_backend"), _to_lower_str),
```

**d. Update `load_config` defaults:**
```python
"models": {
    "asr_id": model_defaults["asr"],
    "llm_id": model_defaults["llm"],
    "llm_backend": "mlx",    # NEW
    "tts_id": model_defaults["tts"],
},
```

**e. Update `_validate_config`** to replace the `llm_id` check with backend-aware validation:

```python
from kaiwacoach.config.models import SUPPORTED_BACKENDS, SUPPORTED_LLM_MODELS

if config.models.llm_backend not in SUPPORTED_BACKENDS:
    raise ValueError(
        f"Unsupported models.llm_backend: {config.models.llm_backend!r}. "
        f"Must be one of {sorted(SUPPORTED_BACKENDS)}."
    )
allowed_ids = SUPPORTED_LLM_MODELS.get(config.models.llm_backend)
if allowed_ids is not None and config.models.llm_id not in allowed_ids:
    raise ValueError(
        f"Unsupported models.llm_id {config.models.llm_id!r} for backend "
        f"{config.models.llm_backend!r}. Must be one of {sorted(allowed_ids)}."
    )
```

### Step 5 — Update `models/factory.py`

Replace `build_llm` with backend-aware routing and family auto-detection:

```python
from kaiwacoach.models.llm_backends import MlxLmBackend, OllamaBackend
from kaiwacoach.models.llm_qwen import QwenLLM

# Ordered list — first matching prefix wins.
_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("mlx-community/Qwen3-", "qwen3"),
    ("qwen3:", "qwen3"),
    # Gemma 4 rules added in PR 2
]

def _detect_family(llm_id: str) -> str:
    for prefix, family in _FAMILY_PREFIXES:
        if llm_id.startswith(prefix):
            return family
    raise ValueError(
        f"Cannot determine LLM family for model ID {llm_id!r}. "
        f"Supported prefixes: {[p for p, _ in _FAMILY_PREFIXES]}"
    )

def build_llm(config: AppConfig) -> LLMProtocol:
    backend_name = config.models.llm_backend
    llm_id = config.models.llm_id

    if backend_name == "mlx":
        backend = MlxLmBackend(llm_id)
        token_counter = backend.count_tokens
    elif backend_name == "ollama":
        backend = OllamaBackend(llm_id)
        token_counter = None   # OllamaBackend does not yet provide a token counter
    else:
        raise ValueError(f"Unknown llm_backend: {backend_name!r}")

    family = _detect_family(llm_id)

    if family == "qwen3":
        return QwenLLM(
            model_id=llm_id,
            max_context_tokens=config.llm.max_context_tokens,
            role_max_new_tokens=dataclasses.asdict(config.llm.role_max_new_tokens),
            backend=backend,
            token_counter=token_counter,
            conversation_temperature=config.llm.conversation_temperature,
        )
    # GemmaLLM added in PR 2
    raise ValueError(f"No wrapper implemented for LLM family {family!r}")
```

### Step 6 — Update `config.example.yaml`

Add `llm_backend` with a comment:

```yaml
models:
  llm_backend: "mlx"   # "mlx" (default, Apple Silicon only) or "ollama" (requires Ollama daemon)
  asr_id: "mlx-community/whisper-large-v3-mlx"
  llm_id: "mlx-community/Qwen3-14B-8bit"
  tts_id: "mlx-community/Kokoro-82M-bf16"
```

### Step 7 — Update `tests/test_app_startup.py`

Add tests covering:
- Default config loads with `llm_backend = "mlx"` (regression — confirm existing Qwen3 path unchanged)
- `llm_backend: "ollama"` with a valid Ollama-style model ID loads without error (validation passes)
- `llm_backend: "ollama"` with any model ID passes (None allowlist = pass-through)
- `llm_backend: "invalid"` raises `ValueError` at startup
- `llm_backend: "mlx"` with an unsupported `llm_id` raises `ValueError` at startup
- `KAIWACOACH_MODELS_LLM_BACKEND` env var is applied correctly

---

## PR 2 — Gemma 4

**Prerequisite**: PR 1 merged.  
**Scope**: Add `GemmaLLM` wrapper, implement `OllamaBackend`, add Gemma 4 model IDs, add thought-tag stripping for the 26B-A4B model. Qwen3 path must remain untouched.

### The Gemma 4 model family

Available via MLX (`mlx-community` on Hugging Face) and Ollama:

| Variant | Architecture | Active params | Ollama tag | MLX prefix |
|---|---|---|---|---|
| e2b-it | Dense 2B | 2B | `gemma4:e2b` | `mlx-community/gemma-4-e2b-it-*` |
| e4b-it | Dense 4B | 4B | `gemma4:e4b` | `mlx-community/gemma-4-e4b-it-*` |
| 26b-a4b-it | MoE 25B | ~4B active | `gemma4:26b` | `mlx-community/gemma-4-26b-a4b-it-*` |
| 31b-it | Dense 31B | 31B | `gemma4:31b` | `mlx-community/gemma-4-31b-it-*` |

**Recommended starting point**: `gemma4:e4b` or `mlx-community/gemma-4-e4b-it-8bit` — lowest implementation risk.  
**Higher quality**: `gemma4:26b` / `mlx-community/gemma-4-26b-a4b-it-*` — requires thought-tag stripping (see below).

### Step 1 — Add Gemma 4 thought-tag stripping to `json_enforcement.py`

The Gemma 4 26B-A4B model always generates thought tags before its answer:

```
<|channel>thought
...reasoning content...
<channel|>
{"reply": "..."}
```

The current `_THINK_TAG_RE` only matches `<think>...</think>` (Qwen3 format) and does not strip these. `raw_decode` then fails because the string does not start with `{`.

Add a second regex and apply both in sequence:

```python
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_GEMMA_CHANNEL_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)

def extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = _THINK_TAG_RE.sub("", text)
    cleaned = _GEMMA_CHANNEL_RE.sub("", cleaned).strip()
    ...
```

This change is harmless for Qwen3 and for e4b (if e4b does not produce channel tags, the regex simply never matches).

### Step 2 — Create `models/llm_gemma.py`

New file. `GemmaLLM` mirrors `QwenLLM` but omits all Qwen3-specific logic:

- No `_NO_THINK_ROLES` — do not append `<think>\n\n</think>` to any prompt
- No `extra_eos = ["}"]` initially — test empirically and add only if it helps
- The `_default_generator` metadata should report `"backend": "mlx_lm"` for MLX, `"backend": "ollama"` for Ollama (read from the backend instance type or a label passed at construction)

Everything else (`generate`, `generate_json`, context limit enforcement, caching, per-role caps) is identical to `QwenLLM`. Do not inherit from `QwenLLM` — the shared logic is small enough that duplication is preferable to coupling. If future maintenance reveals significant drift, extract a shared base at that point.

### Step 3 — Implement `OllamaBackend` in `llm_backends.py`

Replace the stub with a real implementation. The Ollama HTTP API endpoint is `POST http://localhost:11434/api/generate`.

Key mapping:

| `LLMBackend` parameter | Ollama API field |
|---|---|
| `prompt` | `prompt` |
| `max_tokens` | `options.num_predict` |
| `extra_eos_tokens` | `options.stop` |
| `temperature` | `options.temperature` |

Set `"stream": false` to get a single response object. Parse `response["response"]` as the generated text.

`count_tokens` for Ollama: use `POST /api/embed` with the text and read `prompt_eval_count` from the response, or return `None` if this proves unreliable. If `token_counter=None`, context limit enforcement in the wrapper is skipped (see `QwenLLM.generate` — the `token_counter is None` path already handles this gracefully).

**Startup health check**: Add a `check_available()` class method that does a `GET http://localhost:11434` and raises `RuntimeError` with a clear message if it fails. Call this from `build_llm()` in `factory.py` when `llm_backend = "ollama"`, before instantiating any model.

**Offline-first note**: Calling `localhost:11434` is acceptable — the project's offline-first rule is about no cloud/external API calls, not localhost services. This was confirmed by the project owner during planning.

### Step 4 — Update `models/factory.py`

Import `GemmaLLM`. Add Gemma 4 detection rules to `_FAMILY_PREFIXES`:

```python
_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("mlx-community/Qwen3-", "qwen3"),
    ("qwen3:", "qwen3"),
    ("mlx-community/gemma-4-", "gemma4"),   # NEW
    ("gemma4:", "gemma4"),                   # NEW
]
```

Add `GemmaLLM` routing in `build_llm`:

```python
if family == "gemma4":
    return GemmaLLM(
        model_id=llm_id,
        max_context_tokens=config.llm.max_context_tokens,
        role_max_new_tokens=dataclasses.asdict(config.llm.role_max_new_tokens),
        backend=backend,
        token_counter=token_counter,
        conversation_temperature=config.llm.conversation_temperature,
    )
```

Add the Ollama health check call:

```python
if backend_name == "ollama":
    OllamaBackend.check_available()   # raises RuntimeError if daemon not running
    backend = OllamaBackend(llm_id)
```

### Step 5 — Update `config/models.py`

Add Gemma 4 constants and expand `SUPPORTED_LLM_MODELS["mlx"]`:

```python
# Gemma 4 MLX variants (instruction-tuned only)
GEMMA4_E4B_8BIT = "mlx-community/gemma-4-e4b-it-8bit"
GEMMA4_26B_8BIT = "mlx-community/gemma-4-26b-a4b-it-8bit"
# Add additional quantisations as needed

SUPPORTED_LLM_MODELS: dict[str, frozenset[str] | None] = {
    "mlx": frozenset({
        LLM_MODEL_ID_4BIT, LLM_MODEL_ID_8BIT, LLM_MODEL_ID_BF16,   # Qwen3
        GEMMA4_E4B_8BIT, GEMMA4_26B_8BIT,                           # Gemma 4
    }),
    "ollama": None,
}
```

Verify that the exact MLX model IDs exist on `mlx-community` on Hugging Face before adding them. The naming convention is `mlx-community/gemma-4-<variant>-it-<quantisation>` but confirm before hardcoding.

### Step 6 — Update `tests/test_app_startup.py`

Add tests:
- Gemma 4 MLX model ID with `llm_backend: "mlx"` loads cleanly
- Gemma 4 Ollama model ID (`gemma4:e4b`) with `llm_backend: "ollama"` passes validation
- Unsupported Gemma 4 MLX ID (e.g. typo) with `llm_backend: "mlx"` raises `ValueError`

### Step 7 — Update `README.md`

Add Gemma 4 to the LLM variant table under "LLM model variant". Add a note that Ollama backend requires a running Ollama daemon.

### Step 8 — Update `config.example.yaml`

Add Gemma 4 options with notes on temperature and Ollama:

```yaml
# Gemma 4 via MLX (Apple Silicon):
#   llm_backend: "mlx"
#   llm_id: "mlx-community/gemma-4-e4b-it-8bit"
#   Note: Google recommends temperature=1.0 for Gemma 4; default 0.7 is conservative.

# Gemma 4 via Ollama (requires Ollama daemon: https://ollama.com):
#   llm_backend: "ollama"
#   llm_id: "gemma4:e4b"    # or gemma4:26b
```

---

## What Does NOT Change

- `MlxLmBackend` internals — already model-agnostic
- `QwenLLM` — untouched by both PRs
- `prompts/*.md` — stable unless empirical testing reveals JSON output issues
- `orchestrator.py`, storage, ASR, TTS — untouched
- The `LLMProtocol` in `protocols.py` — do not change this interface
- `settings.py` config loading logic (three-layer merge) — only new fields are added
