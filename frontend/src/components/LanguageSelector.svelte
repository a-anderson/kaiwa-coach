<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { sessionStore } from '../lib/stores/session'
  import { setSessionLanguage, createConversation, getConversation, deleteConversation } from '../lib/api/conversations'
  import { LANGUAGE_NATIVE_NAMES, LANGUAGE_ENGLISH_NAMES } from '../lib/constants'

  const dispatch = createEventDispatcher<{ newconversation: void }>()

  // Mirrors SUPPORTED_LANGUAGES in src/kaiwacoach/constants.py
  const LANGUAGE_OPTIONS: { code: string; flag: string; label: string }[] = [
    { code: 'ja',    flag: '🇯🇵', label: `${LANGUAGE_NATIVE_NAMES['ja']} (${LANGUAGE_ENGLISH_NAMES['ja']})` },
    { code: 'fr',    flag: '🇫🇷', label: `${LANGUAGE_NATIVE_NAMES['fr']} (${LANGUAGE_ENGLISH_NAMES['fr']})` },
    { code: 'en',    flag: '🇬🇧', label: LANGUAGE_NATIVE_NAMES['en'] },
    { code: 'es',    flag: '🇪🇸', label: `${LANGUAGE_NATIVE_NAMES['es']} (${LANGUAGE_ENGLISH_NAMES['es']})` },
    { code: 'it',    flag: '🇮🇹', label: `${LANGUAGE_NATIVE_NAMES['it']} (${LANGUAGE_ENGLISH_NAMES['it']})` },
    { code: 'pt-br', flag: '🇧🇷', label: `${LANGUAGE_NATIVE_NAMES['pt-br']} (${LANGUAGE_ENGLISH_NAMES['pt-br']})` },
  ]

  async function onLanguageChange(event: Event) {
    const lang = (event.target as HTMLSelectElement).value
    const { conversationId: prevId, turns: prevTurns } = $sessionStore

    sessionStore.update((s) => ({ ...s, language: lang }))
    try {
      await setSessionLanguage(lang)
    } catch (e) {
      // Local app — network errors are unexpected; silently ignore for now.
      if (import.meta.env.DEV) {
        console.warn('[LanguageSelector] language change failed:', e)
      }
    }

    if (prevId) {
      try {
        const summary = await createConversation(lang)
        if (prevTurns.length === 0) {
          await deleteConversation(prevId)
        }
        const convo = await getConversation(summary.id)
        sessionStore.update((s) => ({
          ...s,
          conversationId: convo.id,
          language: convo.language,
          turns: convo.turns,
        }))
        dispatch('newconversation')
      } catch (e) {
        // If creation fails, just clear the active conversation.
        if (import.meta.env.DEV) {
          console.warn('[LanguageSelector] conversation creation failed:', e)
        }
        sessionStore.update((s) => ({ ...s, conversationId: null, turns: [] }))
      }
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
