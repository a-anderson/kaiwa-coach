<script lang="ts">
  export let text: string
  export let audioUrl: string | null = null
  export let hasAudio: boolean = false
</script>

<div class="bubble assistant-bubble">
  <p class="text">{text}</p>

  {#if audioUrl}
    <!-- Native audio player; WaveSurfer waveform planned for a later phase. -->
    <audio class="player" src={audioUrl} controls preload="none" />
  {:else if hasAudio}
    <p class="audio-unavailable">Audio not available — regenerate to play.</p>
  {/if}
</div>

<style>
  .bubble {
    max-width: 72%;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    background: var(--kc-bot-bubble, #f9f9f9);
    border: 1px solid #e8e8e8;
    align-self: flex-start;
  }

  .text {
    font-size: 0.9rem;
    line-height: 1.55;
    color: #1a1a1a;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .player {
    display: block;
    margin-top: 8px;
    width: 100%;
    height: 32px;
    accent-color: var(--kc-primary, #333);
  }

  .audio-unavailable {
    margin-top: 6px;
    font-size: 0.75rem;
    color: #aaa;
    font-style: italic;
  }
</style>
