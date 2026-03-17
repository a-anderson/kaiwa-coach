<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import AudioPlayer from './AudioPlayer.svelte'

  export let text: string
  export let audioUrl: string | null = null
  export let hasAudio: boolean = false
  export let regenPending: boolean = false
  export let autoplay: boolean = false

  const dispatch = createEventDispatcher<{ regen: void; shadow: void }>()
</script>

<div class="bubble assistant-bubble">
  <p class="text">{text}</p>

  {#if audioUrl}
    <AudioPlayer src={audioUrl} variant="assistant" {autoplay} />
  {:else if hasAudio}
    <p class="audio-unavailable">Audio not available.</p>
  {/if}

  <div class="actions">
    {#if audioUrl}
      <button
        class="action-btn"
        on:click={() => dispatch('shadow')}
        title="Shadow this turn"
        aria-label="Shadow this turn"
      >
        Shadow
      </button>
    {/if}
    <button
      class="action-btn"
      on:click={() => dispatch('regen')}
      disabled={regenPending}
      title="Regenerate audio"
      aria-label="Regenerate audio"
    >
      {regenPending ? '…' : '↺'}
    </button>
  </div>
</div>

<style>
  .bubble {
    max-width: 72%;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    background: var(--kc-bot-bubble, #f9f9f9);
    border: 1px solid color-mix(in srgb, var(--kc-secondary, #aaa) 30%, transparent);
    align-self: flex-start;
  }

  .text {
    font-size: 0.9rem;
    line-height: 1.55;
    color: #1a1a1a;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .audio-unavailable {
    margin-top: 6px;
    font-size: 0.75rem;
    color: #aaa;
    font-style: italic;
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 4px;
  }

  .action-btn {
    background: none;
    border: none;
    padding: 2px 6px;
    font-size: 0.78rem;
    color: #bbb;
    cursor: pointer;
    border-radius: 4px;
    transition: color 0.15s;
    line-height: 1;
  }

  .action-btn:not(:disabled):hover {
    color: var(--kc-primary, #555);
  }

  .action-btn:disabled {
    cursor: default;
    opacity: 0.5;
  }
</style>
