/** Languages supported by the KaiwaCoach backend. Must stay in sync with constants.py. */
export const SUPPORTED_LANGUAGES = ['ja', 'fr', 'en', 'es', 'it', 'pt-br'] as const
export type SupportedLanguage = typeof SUPPORTED_LANGUAGES[number]

export function isSupportedLanguage(lang: string): lang is SupportedLanguage {
  return (SUPPORTED_LANGUAGES as readonly string[]).includes(lang)
}

/** Native-language display name for each supported language code. */
export const LANGUAGE_NATIVE_NAMES: Record<string, string> = {
  ja: '日本語',
  fr: 'Français',
  en: 'English',
  es: 'Español',
  it: 'Italiano',
  'pt-br': 'Português',
}

/** English display name for each supported language code. */
export const LANGUAGE_ENGLISH_NAMES: Record<string, string> = {
  ja: 'Japanese',
  fr: 'French',
  en: 'English',
  es: 'Spanish',
  it: 'Italian',
  'pt-br': 'Portuguese (Brazil)',
}

/**
 * Translation target languages. `value` is the English name stored in the DB
 * and passed directly to the LLM prompt. `label` is the native-script display
 * name shown in the UI. Must stay in sync with SUPPORTED_TRANSLATION_LANGUAGES
 * in constants.py.
 */
export const TRANSLATION_LANGUAGES = [
  { value: 'English', label: 'English' },
  { value: 'Spanish', label: 'Español' },
  { value: 'French', label: 'Français' },
  { value: 'German', label: 'Deutsch' },
  { value: 'Italian', label: 'Italiano' },
  { value: 'Brazilian Portuguese', label: 'Português (Brasil)' },
  { value: 'Korean', label: '한국어' },
  { value: 'Simplified Chinese', label: '中文（简体）' },
  { value: 'Traditional Chinese', label: '中文（繁體）' },
  { value: 'Hindi', label: 'हिन्दी' },
  { value: 'Japanese', label: '日本語' },
] as const

export type TranslationLanguage = typeof TRANSLATION_LANGUAGES[number]['value']

export const DEFAULT_TRANSLATION_LANGUAGE: TranslationLanguage = 'English'
