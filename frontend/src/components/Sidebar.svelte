<script lang="ts">
  import { sessionStore } from '../lib/stores/session'
  import { createConversation, getConversation, deleteConversation, deleteAllConversations, setSessionLanguage } from '../lib/api/conversations'
  import ConversationList from './ConversationList.svelte'
  import ConfirmDialog from './ConfirmDialog.svelte'

  let listRef: ConversationList

  export async function refresh(): Promise<void> {
    await listRef.refresh()
  }
  let creating = false
  let loadError: string | null = null
  let confirmDeleteAll = false

  async function handleSelect(id: string): Promise<void> {
    loadError = null
    try {
      const convo = await getConversation(id)
      await setSessionLanguage(convo.language)
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

  async function handleDeleteAll(): Promise<void> {
    confirmDeleteAll = false
    loadError = null
    try {
      await deleteAllConversations()
      sessionStore.update((s) => ({ ...s, conversationId: null, turns: [] }))
      await listRef.refresh()
    } catch (e) {
      loadError = e instanceof Error ? e.message : 'Failed to delete all conversations'
    }
  }

  async function handleNew(): Promise<void> {
    creating = true
    loadError = null
    try {
      const { conversationId: prevId, turns: prevTurns } = $sessionStore
      const convo = await createConversation($sessionStore.language)
      if (prevId && prevTurns.length === 0) {
        await deleteConversation(prevId)
      }
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
  <div class="list-container">
    <div class="new-btn-row">
      <button class="new-btn" on:click={handleNew} disabled={creating}>
        {creating ? 'Creating…' : '+ New conversation'}
      </button>
    </div>
    <ConversationList bind:this={listRef} on:select={(e) => handleSelect(e.detail)} />
  </div>

  <footer class="sidebar-footer">
    {#if loadError}
      <p class="load-error">{loadError}</p>
    {/if}
    <button class="delete-all-btn" on:click={() => { confirmDeleteAll = true }}>
      Delete all history
    </button>
  </footer>
</aside>

<ConfirmDialog
  open={confirmDeleteAll}
  message="Delete all conversations? This cannot be undone."
  confirmLabel="Delete all"
  on:confirm={handleDeleteAll}
  on:cancel={() => { confirmDeleteAll = false }}
/>

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

  .list-container {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .new-btn-row {
    flex-shrink: 0;
    padding: 10px 12px 4px;
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

  .sidebar-footer {
    flex-shrink: 0;
    padding: 8px 12px;
    border-top: 1px solid #e0e0e0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .load-error {
    font-size: 0.78rem;
    color: #c0392b;
    text-align: center;
  }

  .delete-all-btn {
    width: 100%;
    padding: 6px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #bbb;
    font-size: 0.78rem;
    cursor: pointer;
    transition: color 0.15s;
  }

  .delete-all-btn:hover {
    color: #c0392b;
  }
</style>
