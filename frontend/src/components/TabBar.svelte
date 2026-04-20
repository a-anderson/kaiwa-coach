<script lang="ts">
  import { uiStore } from '../lib/stores/ui'

  type Tab = 'chat' | 'monologue' | 'narration'

  const tabs: { id: Tab; label: string }[] = [
    { id: 'chat', label: 'Chat' },
    { id: 'monologue', label: 'Monologue' },
    { id: 'narration', label: 'Narration' },
  ]

  function setTab(id: Tab) {
    uiStore.update((s) => ({ ...s, activeTab: id }))
  }
</script>

<div class="tab-bar" role="tablist">
  {#each tabs as tab}
    <button
      class="tab-btn"
      class:active={$uiStore.activeTab === tab.id}
      role="tab"
      aria-selected={$uiStore.activeTab === tab.id}
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
