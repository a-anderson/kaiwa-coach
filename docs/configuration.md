# Configuration Guide

This document describes how KaiwaCoach configuration is loaded and which settings you can override.

## Load Order

Configuration is resolved in this order:

1. Built-in defaults
2. Optional config file (`config.example.yaml` shape)
3. Environment variable overrides

Implementation reference: `src/kaiwacoach/settings.py`.

## Core Configuration Areas

- `session`
  - `language`
- `models`
  - `asr_id`
  - `llm_id`
  - `tts_id`
- `llm`
  - `max_context_tokens`
  - `role_max_new_tokens`
- `storage`
  - `root_dir`
  - `expected_sample_rate`
- `tts`
  - `voice`
  - `speed`
- `logging`
  - `timing_logs`

## Environment Variables

### Session

- `KAIWACOACH_SESSION_LANGUAGE`

### Models

- `KAIWACOACH_MODELS_ASR_ID`
- `KAIWACOACH_MODELS_LLM_ID`
- `KAIWACOACH_MODELS_TTS_ID`

### LLM Limits

- `KAIWACOACH_LLM_MAX_CONTEXT_TOKENS`
- `KAIWACOACH_LLM_ROLE_CONVERSATION_MAX_NEW_TOKENS`
- `KAIWACOACH_LLM_ROLE_ERROR_DETECTION_MAX_NEW_TOKENS`
- `KAIWACOACH_LLM_ROLE_CORRECTION_MAX_NEW_TOKENS`
- `KAIWACOACH_LLM_ROLE_NATIVE_REFORMULATION_MAX_NEW_TOKENS`
- `KAIWACOACH_LLM_ROLE_EXPLANATION_MAX_NEW_TOKENS`
- `KAIWACOACH_LLM_ROLE_JP_TTS_NORMALISATION_MAX_NEW_TOKENS`

### Storage

- `KAIWACOACH_STORAGE_ROOT_DIR`
- `KAIWACOACH_STORAGE_EXPECTED_SAMPLE_RATE`

### TTS

- `KAIWACOACH_TTS_VOICE`
- `KAIWACOACH_TTS_SPEED`

## File-Based Configuration

Use `config.example.yaml` as the canonical template for file-based configuration.

## Practical Tips

- Keep `expected_sample_rate` aligned with ASR expectations unless you are intentionally testing resampling.
- Lower role token caps first if you need latency reductions.
- Keep model IDs explicit per environment to avoid accidental model swaps.
