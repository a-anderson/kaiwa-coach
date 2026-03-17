<script lang="ts">
  import { sessionStore } from '../lib/stores/session'
  import { createConversation, getConversation } from '../lib/api/conversations'
  import LanguageSelector from './LanguageSelector.svelte'
  import ConversationList from './ConversationList.svelte'

  let listRef: ConversationList
  let creating = false
  let loadError: string | null = null

  async function handleSelect(id: string): Promise<void> {
    loadError = null
    try {
      const convo = await getConversation(id)
      sessionStore.update((s) => ({
        ...s,
        conversationId: id,
        language: convo.language,
        turns: convo.turns,
      }))
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to load conversation'
    }
  }

  async function handleNew(): Promise<void> {
    creating = true
    loadError = null
    try {
      const convo = await createConversation($sessionStore.language)
      await listRef.refresh()
      await handleSelect(convo.id)
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to create conversation'
    } finally {
      creating = false
    }
  }
</script>

<aside class="sidebar">
  <header class="sidebar-header">
    <span class="logo">KaiwaCoach</span>
    <LanguageSelector />
  </header>

  <div class="list-container">
    <ConversationList bind:this={listRef} on:select={(e) => handleSelect(e.detail)} />
  </div>

  <footer class="sidebar-footer">
    {#if loadError}
      <p class="load-error">{loadError}</p>
    {/if}
    <button class="new-btn" on:click={handleNew} disabled={creating}>
      {creating ? 'Creating…' : '+ New conversation'}
    </button>
  </footer>
</aside>

<style>
  .sidebar {
    width: 260px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #e0e0e0;
    background: #fafafa;
    overflow: hidden;
  }

  .sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 16px;
    border-bottom: 1px solid #e0e0e0;
    gap: 8px;
    flex-shrink: 0;
  }

  .logo {
    font-size: 1rem;
    font-weight: 700;
    color: var(--kc-primary, #333);
    white-space: nowrap;
  }

  .list-container {
    flex: 1;
    overflow-y: auto;
  }

  .sidebar-footer {
    flex-shrink: 0;
    padding: 12px 16px;
    border-top: 1px solid #e0e0e0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .load-error {
    font-size: 0.78rem;
    color: #c0392b;
    text-align: center;
  }

  .new-btn {
    width: 100%;
    padding: 8px;
    border: 1px dashed var(--kc-primary, #999);
    border-radius: 6px;
    background: transparent;
    color: var(--kc-primary, #666);
    font-size: 0.85rem;
    cursor: pointer;
    transition: background 0.15s;
  }

  .new-btn:not(:disabled):hover {
    background: var(--kc-user-bubble, #f5f5f5);
  }

  .new-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
