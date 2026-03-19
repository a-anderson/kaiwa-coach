<script lang="ts">
  import AudioPlayer from './AudioPlayer.svelte'

  export let text: string
  export let audioUrl: string | null = null
</script>

<div class="bubble user-bubble">
  {#if text}
    <p class="text">{text}</p>
  {/if}
  {#if audioUrl}
    <AudioPlayer src={audioUrl} />
  {/if}
</div>

<style>
  .bubble {
    max-width: 72%;
    padding: 10px 14px;
    border-radius: 16px 16px 4px 16px;
    background: var(--kc-user-bubble, #f0f0f0);
    border: 1px solid color-mix(in srgb, var(--kc-primary, #ccc) 20%, transparent);
    align-self: flex-end;
  }

  .text {
    font-size: 0.9rem;
    line-height: 1.55;
    color: #1a1a1a;
    white-space: pre-wrap;
    word-break: break-word;
  }
</style>
