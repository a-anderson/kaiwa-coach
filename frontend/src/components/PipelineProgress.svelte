<script lang="ts">
  import { uiStore } from '../lib/stores/ui'

  // Canonical display order — ASR only appears for audio turns.
  const STAGE_ORDER = ['asr', 'llm', 'tts', 'corrections']
  const STAGE_LABELS: Record<string, string> = {
    asr: 'Transcribing',
    llm: 'Generating reply',
    corrections: 'Checking corrections',
    tts: 'Synthesising audio',
  }

  $: stages = STAGE_ORDER
    .filter((s) => s in $uiStore.stageStatuses)
    .map((s) => ({ name: s, status: $uiStore.stageStatuses[s], label: STAGE_LABELS[s] }))
</script>

{#if $uiStore.isSubmitting}
  <div class="progress" role="status" aria-live="polite">
    {#each stages as stage (stage.name)}
      <span class="stage" class:running={stage.status === 'running'} class:done={stage.status === 'complete'}>
        {#if stage.status === 'complete'}
          <span class="icon">✓</span>
        {:else}
          <span class="icon spinner" aria-hidden="true">◌</span>
        {/if}
        {stage.label}
      </span>
    {/each}
    {#if stages.length === 0}
      <span class="stage running">
        <span class="icon spinner" aria-hidden="true">◌</span>
        Starting…
      </span>
    {/if}
  </div>
{/if}

<style>
  .progress {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    padding: 6px 0 2px;
    font-size: 0.78rem;
  }

  .stage {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #bbb;
    transition: color 0.2s;
  }

  .stage.running {
    color: var(--kc-primary, #555);
    font-weight: 600;
  }

  .stage.done {
    color: #aaa;
    text-decoration: line-through;
  }

  .icon {
    font-size: 0.85em;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .spinner {
    display: inline-block;
    animation: spin 1s linear infinite;
  }
</style>
