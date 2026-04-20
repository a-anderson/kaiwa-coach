<script lang="ts">
  import { onDestroy } from 'svelte'

  export let text: string
  export let label: string = 'Copy'

  let copied = false
  let _timer: ReturnType<typeof setTimeout> | null = null

  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      copied = true
      if (_timer) clearTimeout(_timer)
      _timer = setTimeout(() => { copied = false }, 1500)
    }).catch(() => {
      if (import.meta.env.DEV) console.warn('[CopyButton] clipboard write failed')
    })
  }

  onDestroy(() => { if (_timer) clearTimeout(_timer) })
</script>

<button class="copy-btn" class:copied on:click={handleCopy} title={label} aria-label={label}>
  {#if copied}
    <!-- Checkmark confirmation -->
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M2 7l4 4 6-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  {:else}
    <!-- Two overlapping rounded squares -->
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <rect x="4" y="4" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.4"/>
      <rect x="1" y="1" width="9" height="9" rx="1.5" fill="var(--kc-bg, white)" stroke="currentColor" stroke-width="1.4"/>
    </svg>
  {/if}
</button>

<style>
  .copy-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    cursor: pointer;
    color: #ccc;
    padding: 2px 4px;
    border-radius: 3px;
    transition: color 0.15s;
    line-height: 1;
  }

  .copy-btn:hover {
    color: #999;
  }

  .copy-btn.copied {
    color: #6ab04c;
  }
</style>
