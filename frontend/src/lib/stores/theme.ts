import { derived } from 'svelte/store'
import { sessionStore } from './session'

/**
 * Derived store: current language code, used as the CSS theme key.
 *
 * Also applies data-language to <html> so that themes.css can target it.
 * The subscription runs immediately and on every language change.
 */
export const themeStore = derived(sessionStore, ($session) => $session.language)

// Side-effect: keep data-language attribute in sync with the store.
// Guard for SSR contexts where document is unavailable.
if (typeof document !== 'undefined') {
  themeStore.subscribe((language) => {
    document.documentElement.setAttribute('data-language', language)
  })
}
