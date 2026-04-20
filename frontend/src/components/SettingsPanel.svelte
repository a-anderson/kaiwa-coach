<script lang="ts">
  import { uiStore } from '../lib/stores/ui'

  function close() {
    uiStore.update((s) => ({ ...s, settingsOpen: false }))
  }
</script>

<div class="overlay" role="dialog" aria-label="Settings" aria-modal="true">
  <button class="backdrop" on:click={close} aria-label="Close settings" tabindex="-1" />
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Settings</span>
      <button class="close-btn" on:click={close} aria-label="Close settings">✕</button>
    </div>
    <div class="panel-body">
      <p class="placeholder">User settings coming soon.</p>
    </div>
  </div>
</div>

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.25);
    z-index: 200;
    display: flex;
    justify-content: flex-end;
  }

  .backdrop {
    position: absolute;
    inset: 0;
    background: transparent;
    border: none;
    cursor: default;
    padding: 0;
  }

  .panel {
    position: relative;
    z-index: 1;
    width: 320px;
    height: 100%;
    background: #fff;
    display: flex;
    flex-direction: column;
    box-shadow: -4px 0 16px rgba(0, 0, 0, 0.12);
  }

  .panel-header {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    border-bottom: 1px solid #e0e0e0;
  }

  .panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: #333;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 0.9rem;
    color: #888;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    transition: color 0.15s;
  }

  .close-btn:hover {
    color: #333;
  }

  .panel-body {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
  }

  .placeholder {
    color: #999;
    font-size: 0.9rem;
    font-style: italic;
    margin: 0;
  }
</style>
