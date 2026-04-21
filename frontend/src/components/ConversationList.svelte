<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { sessionStore } from '../lib/stores/session'
  import { listConversations, deleteConversation } from '../lib/api/conversations'
  import ConversationItem from './ConversationItem.svelte'
  import ConfirmDialog from './ConfirmDialog.svelte'
  import type { ConversationSummary } from '../lib/types/api'

  const dispatch = createEventDispatcher<{ select: string }>()

  // Default guarantees type is never undefined; the union excludes it at the type level.
  export let type: 'chat' | 'monologue' = 'chat'

  let conversations: ConversationSummary[] = []
  let loading = true
  let error: string | null = null
  let pendingDeleteId: string | null = null

  export async function refresh(): Promise<void> {
    await load()
  }

  async function load(): Promise<void> {
    loading = true
    error = null
    try {
      conversations = await listConversations(type)
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load conversations'
    } finally {
      loading = false
    }
  }

  // Comma expression makes `type` an explicit reactive dependency so load()
  // re-runs on every tab switch without needing to pass it as an argument.
  $: type, load()

  async function confirmDelete(): Promise<void> {
    if (!pendingDeleteId) return
    const id = pendingDeleteId
    pendingDeleteId = null
    try {
      await deleteConversation(id)
      if ($sessionStore.conversationId === id) {
        sessionStore.update((s) => ({ ...s, conversationId: null, turns: [] }))
      }
      await load()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to delete conversation'
    }
  }
</script>

{#if loading}
  <div class="state-msg">Loading…</div>
{:else if error}
  <div class="state-msg error">
    {error}
    <button class="retry-btn" on:click={load}>Retry</button>
  </div>
{:else if conversations.length === 0}
  <div class="state-msg empty">No conversations yet.</div>
{:else}
  <ul class="list">
    {#each conversations as convo (convo.id)}
      <ConversationItem
        conversation={convo}
        active={$sessionStore.conversationId === convo.id}
        on:select={(e) => dispatch('select', e.detail)}
        on:delete={(e) => (pendingDeleteId = e.detail)}
      />
    {/each}
  </ul>
{/if}

<ConfirmDialog
  open={pendingDeleteId !== null}
  message="Delete this conversation? This cannot be undone."
  confirmLabel="Delete"
  on:confirm={confirmDelete}
  on:cancel={() => (pendingDeleteId = null)}
/>

<style>
  .list {
    padding: 4px 0;
    margin: 0;
  }

  .state-msg {
    padding: 20px 16px;
    font-size: 0.83rem;
    color: #999;
    text-align: center;
  }

  .state-msg.error {
    color: #c0392b;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .retry-btn {
    font-size: 0.8rem;
    padding: 4px 10px;
    border: 1px solid #c0392b;
    border-radius: 4px;
    background: transparent;
    color: #c0392b;
    cursor: pointer;
  }

  .retry-btn:hover {
    background: #fdf2f2;
  }
</style>
