<script lang="ts">
  import { uiStore, type Tab } from '../lib/stores/ui'

  const tabs: { id: Tab; label: string }[] = [
    { id: 'chat', label: 'Chat' },
    { id: 'monologue', label: 'Monologue' },
    { id: 'narration', label: 'Narration' },
  ]

  let tablistEl: HTMLElement

  function setTab(id: Tab) {
    uiStore.update((s) => ({ ...s, activeTab: id }))
  }

  function handleKeydown(e: KeyboardEvent) {
    const idx = tabs.findIndex((t) => t.id === $uiStore.activeTab)
    let next = idx
    if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length
    else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = tabs.length - 1
    else return
    e.preventDefault()
    setTab(tabs[next].id)
    tablistEl.querySelectorAll<HTMLElement>('[role="tab"]')[next]?.focus()
  }
</script>

<div class="tab-bar" role="tablist" tabindex="-1" bind:this={tablistEl} on:keydown={handleKeydown}>
  {#each tabs as tab}
    <button
      id="tab-{tab.id}"
      class="tab-btn"
      class:active={$uiStore.activeTab === tab.id}
      role="tab"
      aria-selected={$uiStore.activeTab === tab.id}
      aria-controls="tabpanel-{tab.id}"
      tabindex={$uiStore.activeTab === tab.id ? 0 : -1}
      on:click={() => setTab(tab.id)}
    >
      {tab.label}
    </button>
  {/each}
</div>

<style>
  .tab-bar {
    flex-shrink: 0;
    display: flex;
    background: #fafafa;
    border-bottom: 1px solid #e0e0e0;
    padding: 0 8px;
  }

  .tab-btn {
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent;
    font-size: 0.85rem;
    color: #888;
    cursor: pointer;
    margin-bottom: -1px;
    transition: color 0.15s, border-color 0.15s;
  }

  .tab-btn:hover {
    color: #555;
  }

  .tab-btn.active {
    color: var(--kc-primary, #333);
    font-weight: 600;
    border-bottom-color: var(--kc-primary, #333);
  }
</style>
