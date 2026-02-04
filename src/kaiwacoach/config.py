"""Application configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LANGUAGE = "ja"
SUPPORTED_LANGUAGES = {"ja", "fr"}


def _load_model_defaults() -> dict[str, str]:
    """Load model defaults from `config/models.py` via package import.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, str]
        Mapping of model roles to default model IDs.

    Raises
    ------
    ImportError
        If the defaults module cannot be imported.
    """
    from kaiwacoach.config import models as model_module  # type: ignore

    return {
        "asr": getattr(model_module, "ASR_MODEL_ID"),
        "llm": getattr(model_module, "LLM_MODEL_ID"),
        "tts": getattr(model_module, "TTS_MODEL_ID"),
    }


def _find_repo_root() -> Path:
    """Locate the repo root by searching upward for `pyproject.toml`.

    Parameters
    ----------
    None

    Returns
    -------
    pathlib.Path
        The repository root path.
    """
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root: missing pyproject.toml.")


@dataclass(frozen=True)
class SessionConfig:
    language: str = DEFAULT_LANGUAGE


@dataclass(frozen=True)
class ModelsConfig:
    asr_id: str
    llm_id: str
    tts_id: str


@dataclass(frozen=True)
class LLMRoleCaps:
    conversation: int = 256
    error_detection: int = 128
    correction: int = 128
    native_reformulation: int = 128
    explanation: int = 192
    jp_tts_normalisation: int = 192


@dataclass(frozen=True)
class LLMConfig:
    max_context_tokens: int = 4096
    role_max_new_tokens: LLMRoleCaps = field(default_factory=LLMRoleCaps)


@dataclass(frozen=True)
class StorageConfig:
    root_dir: str


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "default"
    speed: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    session: SessionConfig
    models: ModelsConfig
    llm: LLMConfig
    storage: StorageConfig
    tts: TTSConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": {"language": self.session.language},
            "models": {
                "asr_id": self.models.asr_id,
                "llm_id": self.models.llm_id,
                "tts_id": self.models.tts_id,
            },
            "llm": {
                "max_context_tokens": self.llm.max_context_tokens,
                "role_max_new_tokens": {
                    "conversation": self.llm.role_max_new_tokens.conversation,
                    "error_detection": self.llm.role_max_new_tokens.error_detection,
                    "correction": self.llm.role_max_new_tokens.correction,
                    "native_reformulation": self.llm.role_max_new_tokens.native_reformulation,
                    "explanation": self.llm.role_max_new_tokens.explanation,
                    "jp_tts_normalisation": self.llm.role_max_new_tokens.jp_tts_normalisation,
                },
            },
            "storage": {"root_dir": self.storage.root_dir},
            "tts": {"voice": self.tts.voice, "speed": self.tts.speed},
        }


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into `base`, returning a new dictionary.

    Parameters
    ----------
    base : dict[str, Any]
        Baseline configuration dictionary.
    overlay : Mapping[str, Any]
        Overlay values that take precedence.

    Returns
    -------
    dict[str, Any]
        The merged configuration.
    """
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_config_file(path: Path | None) -> dict[str, Any]:
    """Parse an optional YAML config file into a dictionary.

    Parameters
    ----------
    path : pathlib.Path | None
        Path to the YAML file, or None to skip parsing.

    Returns
    -------
    dict[str, Any]
        Parsed configuration data.

    Raises
    ------
    RuntimeError
        If a config file exists but PyYAML is unavailable.
    ValueError
        If the YAML root is not a mapping.
    """
    if path is None:
        return {}
    if not path.exists():
        return {}
    if path.is_dir():
        raise ValueError(f"Config path is a directory, expected a file: {path}")

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Config file detected but PyYAML is not available. "
            "Install dependencies or remove the config file."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("Config file must contain a mapping at the top level.")
    return dict(data)


def _apply_env_overrides(config: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Override config values using supported environment variables.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary to update.
    env : Mapping[str, str]
        Environment mapping (typically `os.environ`).

    Returns
    -------
    dict[str, Any]
        Updated configuration with applied overrides.
    """
    def _to_str(value: str, _env_key: str) -> str:
        return str(value)

    def _to_lower_str(value: str, _env_key: str) -> str:
        return str(value).strip().lower()

    def _to_int(value: str, env_key: str) -> int:
        return _coerce_int(value, f"env override {env_key}")

    def _to_float(value: str, env_key: str) -> float:
        return _coerce_float(value, f"env override {env_key}")

    mapping: dict[str, tuple[tuple[str, ...], Any]] = {
        "KAIWACOACH_SESSION_LANGUAGE": (("session", "language"), _to_lower_str),
        "KAIWACOACH_MODELS_ASR_ID": (("models", "asr_id"), _to_str),
        "KAIWACOACH_MODELS_LLM_ID": (("models", "llm_id"), _to_str),
        "KAIWACOACH_MODELS_TTS_ID": (("models", "tts_id"), _to_str),
        "KAIWACOACH_LLM_MAX_CONTEXT_TOKENS": (("llm", "max_context_tokens"), _to_int),
        "KAIWACOACH_LLM_ROLE_CONVERSATION_MAX_NEW_TOKENS": (
            ("llm", "role_max_new_tokens", "conversation"),
            _to_int,
        ),
        "KAIWACOACH_LLM_ROLE_ERROR_DETECTION_MAX_NEW_TOKENS": (
            ("llm", "role_max_new_tokens", "error_detection"),
            _to_int,
        ),
        "KAIWACOACH_LLM_ROLE_CORRECTION_MAX_NEW_TOKENS": (
            ("llm", "role_max_new_tokens", "correction"),
            _to_int,
        ),
        "KAIWACOACH_LLM_ROLE_NATIVE_REFORMULATION_MAX_NEW_TOKENS": (
            ("llm", "role_max_new_tokens", "native_reformulation"),
            _to_int,
        ),
        "KAIWACOACH_LLM_ROLE_EXPLANATION_MAX_NEW_TOKENS": (
            ("llm", "role_max_new_tokens", "explanation"),
            _to_int,
        ),
        "KAIWACOACH_LLM_ROLE_JP_TTS_NORMALISATION_MAX_NEW_TOKENS": (
            ("llm", "role_max_new_tokens", "jp_tts_normalisation"),
            _to_int,
        ),
        "KAIWACOACH_STORAGE_ROOT_DIR": (("storage", "root_dir"), _to_str),
        "KAIWACOACH_TTS_VOICE": (("tts", "voice"), _to_str),
        "KAIWACOACH_TTS_SPEED": (("tts", "speed"), _to_float),
    }

    result = dict(config)
    for env_key, (path, caster) in mapping.items():
        if env_key not in env:
            continue
        raw_value = env[env_key]
        typed_value = caster(raw_value, env_key)
        target = result
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = typed_value
    return result


def _coerce_int(value: Any, field: str) -> int:
    """Convert a value to int with a field-specific error message.

    Parameters
    ----------
    value : Any
        Input value to convert.
    field : str
        Field name for error reporting.

    Returns
    -------
    int
        Converted integer value.

    Raises
    ------
    ValueError
        If the value cannot be converted.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _coerce_float(value: Any, field: str) -> float:
    """Convert a value to float with a field-specific error message.

    Parameters
    ----------
    value : Any
        Input value to convert.
    field : str
        Field name for error reporting.

    Returns
    -------
    float
        Converted float value.

    Raises
    ------
    ValueError
        If the value cannot be converted.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a float") from exc


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from defaults, optional file, and environment overrides.

    Parameters
    ----------
    config_path : str | pathlib.Path | None
        Optional path to the YAML config file. If None, `config.yaml` in the repo root is used.

    Returns
    -------
    AppConfig
        Validated application configuration.

    Raises
    ------
    ValueError
        If any configuration values are invalid.
    RuntimeError
        If YAML parsing is required but unavailable.
    """
    repo_root = _find_repo_root()
    resolved_path = Path(config_path) if config_path else repo_root / "config.yaml"

    model_defaults = _load_model_defaults()
    defaults = {
        "session": {"language": DEFAULT_LANGUAGE},
        "models": {
            "asr_id": model_defaults["asr"],
            "llm_id": model_defaults["llm"],
            "tts_id": model_defaults["tts"],
        },
        "llm": {
            "max_context_tokens": 4096,
            "role_max_new_tokens": {
                "conversation": 256,
                "error_detection": 128,
                "correction": 128,
                "native_reformulation": 128,
                "explanation": 192,
                "jp_tts_normalisation": 192,
            },
        },
        "storage": {"root_dir": str(repo_root / "storage")},
        "tts": {"voice": "default", "speed": 1.0},
    }
    file_data = _parse_config_file(resolved_path)
    merged = _deep_merge(defaults, file_data)
    merged = _apply_env_overrides(merged, os.environ)

    language = str(merged["session"]["language"]).strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}.")

    llm_caps = merged["llm"]["role_max_new_tokens"]

    config = AppConfig(
        session=SessionConfig(language=language),
        models=ModelsConfig(
            asr_id=str(merged["models"]["asr_id"]),
            llm_id=str(merged["models"]["llm_id"]),
            tts_id=str(merged["models"]["tts_id"]),
        ),
        llm=LLMConfig(
            max_context_tokens=_coerce_int(merged["llm"]["max_context_tokens"], "llm.max_context_tokens"),
            role_max_new_tokens=LLMRoleCaps(
                conversation=_coerce_int(llm_caps["conversation"], "llm.role_max_new_tokens.conversation"),
                error_detection=_coerce_int(
                    llm_caps["error_detection"], "llm.role_max_new_tokens.error_detection"
                ),
                correction=_coerce_int(llm_caps["correction"], "llm.role_max_new_tokens.correction"),
                native_reformulation=_coerce_int(
                    llm_caps["native_reformulation"], "llm.role_max_new_tokens.native_reformulation"
                ),
                explanation=_coerce_int(llm_caps["explanation"], "llm.role_max_new_tokens.explanation"),
                jp_tts_normalisation=_coerce_int(
                    llm_caps["jp_tts_normalisation"], "llm.role_max_new_tokens.jp_tts_normalisation"
                ),
            ),
        ),
        storage=StorageConfig(root_dir=str(merged["storage"]["root_dir"])),
        tts=TTSConfig(
            voice=str(merged["tts"]["voice"]),
            speed=_coerce_float(merged["tts"]["speed"], "tts.speed"),
        ),
    )

    _validate_config(config)
    return config


def _validate_config(config: AppConfig) -> None:
    """Validate configuration values and raise ValueError on invalid settings.

    Parameters
    ----------
    config : AppConfig
        Configuration to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any configuration values are invalid.
    """
    if not config.models.asr_id:
        raise ValueError("models.asr_id must be set")
    if not config.models.llm_id:
        raise ValueError("models.llm_id must be set")
    if not config.models.tts_id:
        raise ValueError("models.tts_id must be set")
    if config.llm.max_context_tokens <= 0:
        raise ValueError("llm.max_context_tokens must be > 0")
    if config.tts.speed <= 0:
        raise ValueError("tts.speed must be > 0")
    if not config.storage.root_dir:
        raise ValueError("storage.root_dir must be set")
