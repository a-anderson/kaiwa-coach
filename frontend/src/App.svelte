<script lang="ts">
  // Import themeStore for its side-effect: keeps data-language on <html> in sync.
  import { themeStore } from './lib/stores/theme'
  import LanguageSelector from './components/LanguageSelector.svelte'

  // Consume the store so the import is not tree-shaken.
  $: _theme = $themeStore
</script>

<div class="app">
  <aside class="sidebar">
    <header class="sidebar-header">
      <span class="logo">KaiwaCoach</span>
      <LanguageSelector />
    </header>

    <nav class="conversation-list">
      <!-- Phase 4: conversation list with load/delete -->
    </nav>

    <footer class="sidebar-footer">
      <button class="new-conversation-btn" disabled>
        + New conversation
      </button>
    </footer>
  </aside>

  <main class="main-panel">
    <div class="chat-thread">
      <!-- Phase 4: ChatThread component -->
      <p class="placeholder">Select or create a conversation to begin.</p>
    </div>

    <div class="input-area">
      <!-- Phase 5: TextInput + AudioRecorder components -->
    </div>
  </main>
</div>

<style>
  .app {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Sidebar ── */

  .sidebar {
    width: 260px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #e0e0e0;
    background: #fafafa;
  }

  .sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 16px;
    border-bottom: 1px solid #e0e0e0;
    gap: 8px;
  }

  .logo {
    font-size: 1rem;
    font-weight: 700;
    color: var(--kc-primary, #333);
    white-space: nowrap;
  }

  .conversation-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid #e0e0e0;
  }

  .new-conversation-btn {
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

  .new-conversation-btn:not(:disabled):hover {
    background: var(--kc-user-bubble, #f5f5f5);
  }

  .new-conversation-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }

  /* ── Main panel ── */

  .main-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .chat-thread {
    flex: 1;
    overflow-y: auto;
    padding: 32px 24px;
  }

  .placeholder {
    color: #aaa;
    text-align: center;
    margin-top: 60px;
    font-size: 0.9rem;
  }

  .input-area {
    border-top: 1px solid #e0e0e0;
    padding: 16px 24px;
    min-height: 72px;
  }
</style>
