<script lang="ts">
  // Import themeStore for its side-effect: keeps data-language on <html> in sync.
  import { themeStore } from './lib/stores/theme'
  import { sessionStore } from './lib/stores/session'
  import { uiStore } from './lib/stores/ui'
  import Sidebar from './components/Sidebar.svelte'
  import TabBar from './components/TabBar.svelte'
  import ConversationHeader from './components/ConversationHeader.svelte'
  import ChatThread from './components/ChatThread.svelte'
  import ShadowingPanel from './components/ShadowingPanel.svelte'
  import InputArea from './components/InputArea.svelte'
  import LanguageSelector from './components/LanguageSelector.svelte'
  import SettingsPanel from './components/SettingsPanel.svelte'
  import NarrationPanel from './components/NarrationPanel.svelte'
  import MonologuePanel from './components/MonologuePanel.svelte'

  let sidebarRef: Sidebar

  // Consume the store so the import is not tree-shaken.
  $: _theme = $themeStore

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      if ($uiStore.settingsOpen) {
        uiStore.update((s) => ({ ...s, settingsOpen: false }))
      } else if ($uiStore.shadowingTurnId !== null) {
        uiStore.update((s) => ({ ...s, shadowingTurnId: null }))
      }
    }
  }

  function toggleSettings() {
    uiStore.update((s) => ({ ...s, settingsOpen: !s.settingsOpen }))
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="app">
  <header class="top-bar">
    <span class="logo">KaiwaCoach</span>
    <div class="top-bar-right">
      <LanguageSelector on:newconversation={() => sidebarRef?.refresh()} />
      <button class="gear-btn" on:click={toggleSettings} aria-label="Open settings" title="Settings">
        ⚙
      </button>
    </div>
  </header>

  <TabBar />

  <div class="content">
    {#if $uiStore.activeTab !== 'narration'}
      <Sidebar bind:this={sidebarRef} />
    {/if}

    <main class="main-panel">
      {#if $uiStore.activeTab === 'chat'}
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
          <InputArea on:turncomplete={() => sidebarRef?.refresh()} />
        {/if}
      {:else if $uiStore.activeTab === 'monologue'}
        <MonologuePanel />
      {:else if $uiStore.activeTab === 'narration'}
        <NarrationPanel />
      {/if}
    </main>
  </div>

  {#if $uiStore.settingsOpen}
    <SettingsPanel />
  {/if}
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  .top-bar {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    height: 49px;
    border-bottom: 1px solid color-mix(in srgb, var(--kc-primary, #ccc) 20%, transparent);
    background: var(--kc-user-bubble, #f5f5f5);
  }

  .logo {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--kc-primary, #333);
    white-space: nowrap;
  }

  .top-bar-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .gear-btn {
    display: inline-flex;
    align-items: center;
    background: none;
    border: none;
    font-size: 2.8rem;
    line-height: 1;
    color: var(--kc-primary, #333);
    cursor: pointer;
    padding: 0 4px;
    border-radius: 4px;
    transform: translateY(-10%);
    transition: opacity 0.15s;
  }

  .gear-btn:hover {
    opacity: 0.7;
  }

  .content {
    flex: 1;
    display: flex;
    overflow: hidden;
    min-height: 0;
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
