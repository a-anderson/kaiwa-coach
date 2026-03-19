<script lang="ts">
  import type { CorrectionData } from '../lib/types/api'

  export let correction: CorrectionData

  let open = false

  $: hasContent =
    correction.corrected ||
    correction.native ||
    correction.explanation
</script>

{#if hasContent}
  <div class="card">
    <button
      class="toggle"
      on:click={() => (open = !open)}
      aria-expanded={open}
    >
      <span class="icon">{open ? '▾' : '▸'}</span>
      <span class="label">Corrections</span>
    </button>

    {#if open}
      <div class="body">
        {#if correction.corrected}
          <section class="section">
            <h4>Corrected</h4>
            <p>{correction.corrected}</p>
          </section>
        {/if}

        {#if correction.native}
          <section class="section">
            <h4>Natural phrasing</h4>
            <p>{correction.native}</p>
          </section>
        {/if}

        {#if correction.explanation}
          <section class="section">
            <h4>Explanation</h4>
            <p>{correction.explanation}</p>
          </section>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .card {
    margin-top: 6px;
    border: 1px solid var(--kc-correction-border, #e0a0a0);
    border-radius: 8px;
    background: #fff;
    max-width: 72%;
    align-self: flex-end;
    font-size: 0.83rem;
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 7px 12px;
    background: none;
    border: none;
    cursor: pointer;
    color: var(--kc-primary, #333);
    font-size: 0.83rem;
    font-weight: 600;
    text-align: left;
    border-radius: 8px;
  }

  .toggle:hover {
    background: var(--kc-user-bubble, #f5f5f5);
  }

  .body {
    padding: 4px 12px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-top: 1px solid color-mix(in srgb, var(--kc-correction-border, #e0a0a0) 30%, transparent);
  }

  .section h4 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #999;
    margin-bottom: 3px;
  }

  .section p {
    color: #333;
    line-height: 1.5;
  }
</style>
