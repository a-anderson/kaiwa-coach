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
      <button
        class="gear-btn"
        on:click={toggleSettings}
        aria-label={$uiStore.settingsOpen ? 'Close settings' : 'Open settings'}
        aria-pressed={$uiStore.settingsOpen}
        title="Settings"
      >
        <!-- Heroicons cog-6-tooth, MIT licence -->
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
        </svg>
      </button>
    </div>
  </header>

  <TabBar />

  <div class="content">
    {#if $uiStore.activeTab !== 'narration'}
      <Sidebar bind:this={sidebarRef} />
    {/if}

    <main class="main-panel">
      <div
        id="tabpanel-{$uiStore.activeTab}"
        role="tabpanel"
        aria-labelledby="tab-{$uiStore.activeTab}"
        class="tab-content"
      >
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
      </div>
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
    display: flex;
    align-items: center;
    background: none;
    border: none;
    color: var(--kc-primary, #333);
    cursor: pointer;
    padding: 0 4px;
    border-radius: 4px;
    transition: opacity 0.15s;
  }

  .gear-btn svg {
    width: 28px;
    height: 28px;
    display: block;
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

  .tab-content {
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
