<script lang="ts">
  import { createEventDispatcher } from 'svelte'

  export let open: boolean = false
  export let message: string = 'Are you sure?'
  export let confirmLabel: string = 'Delete'

  const dispatch = createEventDispatcher<{ confirm: void; cancel: void }>()
</script>

{#if open}
  <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
  <div class="overlay" on:click|self={() => dispatch('cancel')}>
    <div class="dialog" role="alertdialog" aria-modal="true">
      <p class="message">{message}</p>
      <div class="actions">
        <button class="btn-cancel" on:click={() => dispatch('cancel')}>Cancel</button>
        <button class="btn-confirm" on:click={() => dispatch('confirm')}>{confirmLabel}</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .dialog {
    background: #fff;
    border-radius: 8px;
    padding: 24px;
    min-width: 280px;
    max-width: 380px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18);
  }

  .message {
    font-size: 0.95rem;
    color: #333;
    margin-bottom: 20px;
    line-height: 1.5;
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  button {
    padding: 7px 16px;
    border-radius: 5px;
    font-size: 0.875rem;
    cursor: pointer;
    border: 1px solid transparent;
  }

  .btn-cancel {
    background: #f0f0f0;
    color: #444;
    border-color: #ddd;
  }

  .btn-cancel:hover {
    background: #e4e4e4;
  }

  .btn-confirm {
    background: #c0392b;
    color: #fff;
  }

  .btn-confirm:hover {
    background: #a93226;
  }
</style>
