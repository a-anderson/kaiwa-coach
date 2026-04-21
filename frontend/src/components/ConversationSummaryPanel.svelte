<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { slide } from 'svelte/transition'
  import type { ConversationSummaryResponse } from '../lib/api/conversations'

  export let data: ConversationSummaryResponse

  const dispatch = createEventDispatcher<{ close: void }>()
</script>

<div class="summary-panel" transition:slide={{ duration: 200 }}>
  <div class="summary-header">
    <span class="summary-title">Conversation Summary</span>
    <button class="close-btn" on:click={() => dispatch('close')} title="Close summary">×</button>
  </div>

  <div class="summary-body">
    {#if data.top_error_patterns.length > 0}
      <section class="summary-section">
        <h4>Top error patterns</h4>
        <ul>
          {#each data.top_error_patterns as pattern}
            <li>{pattern}</li>
          {/each}
        </ul>
      </section>
    {/if}

    {#if data.priority_areas.length > 0}
      <section class="summary-section">
        <h4>Priority areas</h4>
        <ul>
          {#each data.priority_areas as area}
            <li>{area}</li>
          {/each}
        </ul>
      </section>
    {/if}

    {#if data.overall_notes}
      <section class="summary-section">
        <h4>Overall notes</h4>
        <p>{data.overall_notes}</p>
      </section>
    {/if}
  </div>
</div>

<style>
  .summary-panel {
    background: var(--kc-bot-bubble, #f9f9f9);
    border-bottom: 1px solid color-mix(in srgb, var(--kc-secondary, #aaa) 30%, transparent);
    padding: 6px 16px;
    font-size: 0.85rem;
  }

  .summary-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
  }

  .summary-title {
    font-weight: 600;
    color: var(--kc-primary, #444);
    font-size: 0.88rem;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 1.1rem;
    color: #999;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
  }

  .close-btn:hover {
    color: #444;
  }

  .summary-body {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }

  .summary-section {
    flex: 1;
    min-width: 180px;
  }

  h4 {
    margin: 0 0 6px 0;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--kc-primary, #555);
    opacity: 0.75;
  }

  ul {
    margin: 0;
    padding-left: 16px;
  }

  li {
    margin-bottom: 3px;
    color: #333;
    line-height: 1.4;
  }

  p {
    margin: 0;
    color: #333;
    line-height: 1.4;
  }
</style>
