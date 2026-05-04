<script lang="ts">
  import { translateTurn } from '../lib/api/translate'

  export let assistantTurnId: string
  export let targetLanguage = 'English'

  type State = 'idle' | 'loading' | 'loaded' | 'error'

  let state: State = 'idle'
  let translation = ''
  let errorMessage = ''
  let open = false

  async function handleTranslate() {
    if (state === 'loading') return
    if (state === 'loaded') {
      open = !open
      return
    }
    state = 'loading'
    try {
      translation = await translateTurn(assistantTurnId, targetLanguage)
      state = 'loaded'
      open = true
    } catch (e) {
      errorMessage = e instanceof Error ? e.message : 'Translation failed'
      state = 'error'
    }
  }
</script>

<div class="card">
  <button
    class="toggle"
    class:loaded={state === 'loaded'}
    on:click={handleTranslate}
    disabled={state === 'loading'}
    aria-expanded={state === 'loaded' && open}
  >
    {#if state === 'loading'}
      <span class="spinner" aria-hidden="true" />
    {:else}
      <span class="icon">{state === 'loaded' && open ? '▾' : '▸'}</span>
    {/if}
    <span class="label">Translation</span>
  </button>

  {#if state === 'loaded' && open}
    <div class="body">
      <p>{translation}</p>
    </div>
  {/if}

  {#if state === 'error'}
    <p class="error">{errorMessage}</p>
  {/if}
</div>

<style>
  .card {
    margin-top: 4px;
    border: 1px solid color-mix(in srgb, var(--kc-secondary, #aaa) 30%, transparent);
    border-radius: 8px;
    background: #fff;
    max-width: 72%;
    align-self: flex-start;
    font-size: 0.83rem;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 7px 12px;
    background: none;
    border: none;
    cursor: pointer;
    color: var(--kc-secondary, #aaa);
    font-size: 0.83rem;
    font-weight: 600;
    text-align: left;
    border-radius: 8px;
  }

  .toggle:hover:not(:disabled) {
    background: var(--kc-bot-bubble, #f9f9f9);
  }

  .toggle:disabled {
    cursor: default;
  }

  .body {
    padding: 4px 12px 12px;
    border-top: 1px solid color-mix(in srgb, var(--kc-secondary, #aaa) 30%, transparent);
  }

  .body p {
    color: #333;
    line-height: 1.5;
    margin: 0;
  }

  .error {
    font-size: 0.75rem;
    color: #c0392b;
    margin: 0 12px 8px;
  }

  .spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid color-mix(in srgb, var(--kc-secondary, #aaa) 30%, transparent);
    border-top-color: var(--kc-secondary, #aaa);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
