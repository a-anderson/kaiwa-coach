"""Shared constants for KaiwaCoach."""

DEFAULT_LANGUAGE = "ja"

# Maps the English name (stored in DB and passed to LLM prompts) to the
# native-script display name shown in the UI.
SUPPORTED_TRANSLATION_LANGUAGES: dict[str, str] = {
    "English": "English",
    "Spanish": "Español",
    "French": "Français",
    "German": "Deutsch",
    "Italian": "Italiano",
    "Brazilian Portuguese": "Português (Brasil)",
    "Korean": "한국어",
    "Simplified Chinese": "中文（简体）",
    "Traditional Chinese": "中文（繁體）",
    "Hindi": "हिन्दी",
    "Japanese": "日本語",
}

DEFAULT_TRANSLATION_LANGUAGE = "English"
SUPPORTED_LANGUAGES = {"ja", "fr", "en", "es", "it", "pt-br"}

DEFAULT_VOICES = {
    "en": "bf_emma",
    "fr": "ff_siwis",
    "ja": "jf_alpha",
    "es": "ef_dora",
    "it": "if_sara",
    "pt": "pf_dora",
    "pt-br": "pf_dora",
}

_JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1", "Native"]
_CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2", "Native"]

# Valid proficiency levels per language key.
# "ja_kanji" is a valid key (independent kanji reading level for Japanese) even
# though it is not in SUPPORTED_LANGUAGES. Validate against this dict, not SUPPORTED_LANGUAGES.
VALID_PROFICIENCY_LEVELS: dict[str, list[str]] = {
    "ja": _JLPT_LEVELS,
    "ja_kanji": _JLPT_LEVELS,
    "fr": _CEFR_LEVELS,
    "en": _CEFR_LEVELS,
    "es": _CEFR_LEVELS,
    "it": _CEFR_LEVELS,
    "pt-br": _CEFR_LEVELS,
}
