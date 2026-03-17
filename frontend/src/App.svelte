<script lang="ts">
  // Import themeStore for its side-effect: keeps data-language on <html> in sync.
  import { themeStore } from './lib/stores/theme'
  import { sessionStore } from './lib/stores/session'
  import { uiStore } from './lib/stores/ui'
  import Sidebar from './components/Sidebar.svelte'
  import ConversationHeader from './components/ConversationHeader.svelte'
  import ChatThread from './components/ChatThread.svelte'
  import ShadowingPanel from './components/ShadowingPanel.svelte'
  import InputArea from './components/InputArea.svelte'

  // Consume the store so the import is not tree-shaken.
  $: _theme = $themeStore

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && $uiStore.shadowingTurnId !== null) {
      uiStore.update((s) => ({ ...s, shadowingTurnId: null }))
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="app">
  <Sidebar />

  <main class="main-panel">
    {#if $sessionStore.conversationId === null}
      <div class="no-conversation">
        <p class="empty-title">No conversation selected</p>
        <p class="empty-hint">Choose one from the sidebar, or click <strong>+ New</strong> to start.</p>
      </div>
    {:else}
      <ConversationHeader />
      <ChatThread />
      {#if $uiStore.shadowingTurnId}
        <ShadowingPanel />
      {/if}
      <InputArea />
    {/if}
  </main>
</div>

<style>
  .app {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  .main-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
  }

  .no-conversation {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: #888;
    margin: 0;
  }

  .empty-hint {
    font-size: 0.85rem;
    color: #bbb;
    margin: 0;
  }
</style>
