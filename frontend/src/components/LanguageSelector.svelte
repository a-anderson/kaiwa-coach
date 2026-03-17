<script lang="ts">
  import { sessionStore } from '../lib/stores/session'

  // Mirrors SUPPORTED_LANGUAGES in src/kaiwacoach/constants.py
  const LANGUAGE_OPTIONS: { code: string; flag: string; label: string }[] = [
    { code: 'ja', flag: '🇯🇵', label: '日本語 (Japanese)' },
    { code: 'fr', flag: '🇫🇷', label: 'Français (French)' },
    { code: 'en', flag: '🇬🇧', label: 'English' },
    { code: 'es', flag: '🇪🇸', label: 'Español (Spanish)' },
    { code: 'it', flag: '🇮🇹', label: 'Italiano (Italian)' },
    { code: 'pt', flag: '🇧🇷', label: 'Português (Brazilian Portuguese)' },
  ]

  async function onLanguageChange(event: Event) {
    const lang = (event.target as HTMLSelectElement).value
    sessionStore.update((s) => ({ ...s, language: lang }))
    try {
      await fetch('/api/session/language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: lang }),
      })
    } catch {
      // Local app — network errors are unexpected; silently ignore for now.
    }
  }
</script>

<select
  class="language-selector"
  value={$sessionStore.language}
  on:change={onLanguageChange}
  aria-label="Select language"
>
  {#each LANGUAGE_OPTIONS as { code, flag, label }}
    <option value={code}>{flag} {label}</option>
  {/each}
</select>

<style>
  .language-selector {
    appearance: none;
    background: transparent;
    border: 1px solid var(--kc-primary, #333);
    border-radius: 4px;
    color: var(--kc-primary, #333);
    cursor: pointer;
    font-size: 0.8rem;
    padding: 4px 24px 4px 8px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23666' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 6px center;
  }

  .language-selector:focus {
    outline: 2px solid var(--kc-primary-light, #555);
    outline-offset: 1px;
  }
</style>
