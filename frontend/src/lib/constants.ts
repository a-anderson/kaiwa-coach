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
