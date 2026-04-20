<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { fly } from 'svelte/transition'
  import { uiStore } from '../lib/stores/ui'

  let panelEl: HTMLElement
  let previousFocus: HTMLElement | null = null

  const FOCUSABLE =
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

  function close() {
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

  onMount(() => {
    previousFocus = document.activeElement as HTMLElement | null
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
      <p class="placeholder">User settings coming soon.</p>
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
    color: #333;
  }

  .panel-body {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
  }

  .placeholder {
    color: #999;
    font-size: 0.9rem;
    font-style: italic;
    margin: 0;
  }
</style>
