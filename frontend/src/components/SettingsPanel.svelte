<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte'
  import { fly } from 'svelte/transition'
  import { uiStore } from '../lib/stores/ui'
  import { getProfile, setProfile } from '../lib/api/settings'
  import { SUPPORTED_LANGUAGES as LANGUAGE_CODES, LANGUAGE_NATIVE_NAMES } from '../lib/constants'

  const SUPPORTED_LANGUAGES = LANGUAGE_CODES.map((code) => ({
    code,
    label: LANGUAGE_NATIVE_NAMES[code] ?? code,
  }))
  const JLPT_LEVELS = ['N5', 'N4', 'N3', 'N2', 'N1', 'Native']
  const CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'Native']

  let panelEl: HTMLElement
  let previousFocus: HTMLElement | null = null

  let userName = ''
  let proficiency: Record<string, string> = {}
  let loading = false
  let saving = false
  let error = ''

  const FOCUSABLE =
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

  function close() {
    previousFocus?.focus()
    uiStore.update((s) => ({ ...s, settingsOpen: false }))
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key !== 'Tab') return
    const focusable = Array.from(panelEl.querySelectorAll<HTMLElement>(FOCUSABLE))
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault()
        last.focus()
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
  }

  async function load() {
    loading = true
    error = ''
    try {
      const profile = await getProfile()
      userName = profile.user_name ?? ''
      proficiency = { ...profile.language_proficiency }
    } catch (e) {
      if (import.meta.env.DEV) console.warn('[SettingsPanel] load failed', e)
      error = 'Failed to load settings.'
    } finally {
      loading = false
    }
  }

  async function save() {
    saving = true
    error = ''
    try {
      await setProfile({
        user_name: userName.trim() || null,
        language_proficiency: proficiency,
      })
      close()
    } catch (e) {
      if (import.meta.env.DEV) console.warn('[SettingsPanel] save failed', e)
      error = 'Failed to save settings.'
    } finally {
      saving = false
    }
  }

  function levelFor(code: string): string {
    return proficiency[code] ?? (code === 'ja' || code === 'ja_kanji' ? 'N5' : 'A1')
  }

  function setLevel(code: string, value: string) {
    proficiency = { ...proficiency, [code]: value }
  }

  onMount(async () => {
    previousFocus = document.activeElement as HTMLElement | null
    await load()
    await tick()
    const first = panelEl.querySelector<HTMLElement>(FOCUSABLE)
    first?.focus()
    window.addEventListener('keydown', handleKeydown)
  })

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown)
    previousFocus?.focus()
  })
</script>

<div class="overlay" role="dialog" aria-label="Settings" aria-modal="true">
  <button class="backdrop" on:click={close} aria-label="Close settings" tabindex="-1" />
  <div class="panel" bind:this={panelEl} transition:fly={{ x: 320, duration: 200 }}>
    <div class="panel-header">
      <span class="panel-title">Settings</span>
      <button class="close-btn" on:click={close} aria-label="Close settings">✕</button>
    </div>

    <div class="panel-body">
      {#if loading}
        <p class="status-text">Loading…</p>
      {:else}
        {#if error}
          <p class="error-text">{error}</p>
        {/if}

        <section class="field-group">
          <label class="field-label" for="user-name">Your name <span class="optional">(optional)</span></label>
          <input
            id="user-name"
            class="text-input"
            type="text"
            placeholder="Leave blank to omit"
            bind:value={userName}
          />
        </section>

        <section class="field-group">
          <h3 class="section-heading">Proficiency levels</h3>

          {#each SUPPORTED_LANGUAGES as lang}
            <div class="lang-section">
              <h4 class="lang-label">{lang.label}</h4>

              {#if lang.code === 'ja'}
                <label class="select-label" for="level-ja">Grammar level</label>
                <select
                  id="level-ja"
                  class="level-select"
                  value={levelFor('ja')}
                  on:change={(e) => setLevel('ja', e.currentTarget.value)}
                >
                  {#each JLPT_LEVELS as lvl}
                    <option value={lvl}>{lvl}</option>
                  {/each}
                </select>

                <label class="select-label" for="level-ja-kanji">
                  Kanji reading level
                  <span class="tooltip" title="Kanji level is independent of grammar. 'Native' means educated Japanese adult level (beyond JLPT N1).">ⓘ</span>
                </label>
                <select
                  id="level-ja-kanji"
                  class="level-select"
                  value={levelFor('ja_kanji')}
                  on:change={(e) => setLevel('ja_kanji', e.currentTarget.value)}
                >
                  {#each JLPT_LEVELS as lvl}
                    <option value={lvl}>{lvl}</option>
                  {/each}
                </select>
              {:else}
                <label class="select-label" for="level-{lang.code}">Level</label>
                <select
                  id="level-{lang.code}"
                  class="level-select"
                  value={levelFor(lang.code)}
                  on:change={(e) => setLevel(lang.code, e.currentTarget.value)}
                >
                  {#each CEFR_LEVELS as lvl}
                    <option value={lvl}>{lvl}</option>
                  {/each}
                </select>
              {/if}
            </div>
          {/each}
        </section>

        <div class="actions">
          <button class="save-btn" on:click={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.25);
    z-index: 200;
    display: flex;
    justify-content: flex-end;
  }

  .backdrop {
    position: absolute;
    inset: 0;
    background: transparent;
    border: none;
    cursor: default;
    padding: 0;
  }

  .panel {
    position: relative;
    z-index: 1;
    width: 320px;
    height: 100%;
    background: #fff;
    display: flex;
    flex-direction: column;
    box-shadow: -4px 0 16px rgba(0, 0, 0, 0.12);
  }

  .panel-header {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    background: var(--kc-user-bubble, #f5f5f5);
    border-bottom: 1px solid color-mix(in srgb, var(--kc-primary, #ccc) 20%, transparent);
  }

  .panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--kc-primary, #333);
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 0.9rem;
    color: #888;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    transition: color 0.15s;
  }

  .close-btn:hover {
    color: var(--kc-primary, #333);
  }

  .panel-body {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
  }

  .status-text {
    color: #999;
    font-size: 0.9rem;
    font-style: italic;
    margin: 0;
  }

  .error-text {
    color: #c0392b;
    font-size: 0.85rem;
    margin: 0 0 12px;
  }

  .field-group {
    margin-bottom: 20px;
  }

  .field-label {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    color: #444;
    margin-bottom: 6px;
  }

  .optional {
    font-weight: 400;
    color: #999;
  }

  .text-input {
    width: 100%;
    box-sizing: border-box;
    padding: 7px 10px;
    font-size: 0.9rem;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    outline: none;
    transition: border-color 0.15s;
  }

  .text-input:focus {
    border-color: var(--kc-primary, #666);
  }

  .section-heading {
    font-size: 0.85rem;
    font-weight: 600;
    color: #444;
    margin: 0 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #eee;
  }

  .lang-section {
    margin-bottom: 16px;
  }

  .lang-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--kc-primary, #555);
    margin: 0 0 8px;
  }

  .select-label {
    display: block;
    font-size: 0.8rem;
    color: #666;
    margin-bottom: 4px;
  }

  .level-select {
    width: 100%;
    box-sizing: border-box;
    padding: 6px 8px;
    font-size: 0.85rem;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    background: #fff;
    margin-bottom: 10px;
    outline: none;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .level-select:focus {
    border-color: var(--kc-primary, #666);
  }

  .tooltip {
    font-size: 0.75rem;
    color: #999;
    cursor: help;
    margin-left: 4px;
  }

  .actions {
    padding-top: 8px;
    border-top: 1px solid #eee;
  }

  .save-btn {
    width: 100%;
    padding: 9px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #fff;
    background: var(--kc-primary, #444);
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .save-btn:hover:not(:disabled) {
    opacity: 0.88;
  }

  .save-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
