<script lang="ts">
  import { onDestroy } from 'svelte'
  import { sessionStore } from '../lib/stores/session'
  import { generateNarration } from '../lib/api/narration'
  import { LANGUAGE_NATIVE_NAMES } from '../lib/constants'
  import AudioPlayer from './AudioPlayer.svelte'

  let text = ''
  let loading = false
  let error = ''
  let audioUrl: string | null = null
  let prevBlobUrl: string | null = null

  $: language = $sessionStore.language
  $: languageDisplay = LANGUAGE_NATIVE_NAMES[language] ?? language.toUpperCase()

  function revokePrev() {
    if (prevBlobUrl) {
      URL.revokeObjectURL(prevBlobUrl)
      prevBlobUrl = null
    }
  }

  async function handleGenerate() {
    if (!text.trim()) return
    loading = true
    error = ''

    try {
      const result = await generateNarration(text)
      revokePrev()
      // The URL from the server is a direct path, not a blob URL — no revocation needed
      // but we track it for consistency if we ever switch to blob URLs.
      audioUrl = result.audio_url
    } catch (err: unknown) {
      error = err instanceof Error ? err.message : 'Failed to generate audio'
      audioUrl = null
    } finally {
      loading = false
    }
  }

  onDestroy(() => {
    revokePrev()
  })
</script>

<div class="narration-panel">
  <div class="narration-inner">
    <h2 class="panel-title">Narrate in <span class="lang-label">{languageDisplay}</span></h2>

    <textarea
      class="narration-textarea"
      bind:value={text}
      placeholder="Paste or type text here..."
      rows={6}
      disabled={loading}
    />

    <button
      class="generate-btn"
      on:click={handleGenerate}
      disabled={loading || !text.trim()}
    >
      {#if loading}
        Generating…
      {:else}
        Generate Audio
      {/if}
    </button>

    {#if error}
      <p class="error-msg">{error}</p>
    {/if}

    {#if audioUrl}
      <div class="preview-section">
        <hr class="divider" />
        <div class="preview-header">
          <span class="preview-label">Preview</span>
          <a
            class="download-btn"
            href={audioUrl}
            download
            title="Download audio"
            aria-label="Download audio"
          >
            ↓ Download
          </a>
        </div>
        <AudioPlayer src={audioUrl} variant="assistant" />
      </div>
    {/if}
  </div>
</div>

<style>
  .narration-panel {
    flex: 1;
    display: flex;
    justify-content: center;
    overflow-y: auto;
    padding: 2rem 1rem;
  }

  .narration-inner {
    width: 100%;
    max-width: 680px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .panel-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
    color: #222;
  }

  .lang-label {
    color: var(--kc-primary, #555);
  }

  .narration-textarea {
    width: 100%;
    resize: vertical;
    padding: 0.75rem;
    border-radius: 6px;
    border: 1px solid #d0d0d0;
    background: #fff;
    color: #222;
    font-size: 0.95rem;
    font-family: inherit;
    line-height: 1.5;
    box-sizing: border-box;
  }

  .narration-textarea:focus {
    outline: none;
    border-color: var(--kc-primary, #555);
  }

  .narration-textarea:disabled {
    background: #f8f8f8;
    opacity: 0.7;
  }

  .generate-btn {
    align-self: flex-start;
    padding: 0.5rem 1.25rem;
    border: none;
    border-radius: 5px;
    background: var(--kc-primary, #555);
    color: #fff;
    font-size: 0.9rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .generate-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .error-msg {
    color: #e57373;
    font-size: 0.85rem;
    margin: 0;
  }

  .divider {
    border: none;
    border-top: 1px solid var(--kc-border, #333);
    margin: 0.5rem 0;
  }

  .preview-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .preview-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .preview-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .download-btn {
    font-size: 0.8rem;
    color: #555;
    text-decoration: none;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    border: 1px solid #d0d0d0;
    transition: background 0.15s;
  }

  .download-btn:hover {
    background: #f0f0f0;
  }
</style>
