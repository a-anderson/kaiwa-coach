/** Languages supported by the KaiwaCoach backend. Must stay in sync with constants.py. */
export const SUPPORTED_LANGUAGES = ['ja', 'fr', 'en', 'es', 'it', 'pt-br'] as const
export type SupportedLanguage = typeof SUPPORTED_LANGUAGES[number]

export function isSupportedLanguage(lang: string): lang is SupportedLanguage {
  return (SUPPORTED_LANGUAGES as readonly string[]).includes(lang)
}
