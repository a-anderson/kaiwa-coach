<script lang="ts">
  // Import themeStore for its side-effect: keeps data-language on <html> in sync.
  import { themeStore } from './lib/stores/theme'
  import { sessionStore } from './lib/stores/session'
  import Sidebar from './components/Sidebar.svelte'

  // Consume the store so the import is not tree-shaken.
  $: _theme = $themeStore
</script>

<div class="app">
  <Sidebar />

  <main class="main-panel">
    <div class="chat-thread">
      {#if $sessionStore.conversationId === null}
        <p class="placeholder">Select or create a conversation to begin.</p>
      {:else if $sessionStore.turns.length === 0}
        <p class="placeholder">No messages yet — start the conversation!</p>
      {:else}
        <!-- Phase 5: ChatThread component -->
        <p class="placeholder">
          {$sessionStore.turns.length} turn(s) loaded — chat UI coming in Phase 5.
        </p>
      {/if}
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
