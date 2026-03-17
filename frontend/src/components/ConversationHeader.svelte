<script lang="ts">
  import { sessionStore } from '../lib/stores/session'
  import { regenConversationAudio } from '../lib/api/regen'

  let regenPending = false
  let regenError: string | null = null
  let regenProgress = 0
  let regenTotal = 0

  async function handleRegenAll() {
    const convId = $sessionStore.conversationId
    if (!convId || regenPending) return

    regenPending = true
    regenError = null
    regenProgress = 0
    regenTotal = $sessionStore.turns.filter((t) => t.reply_text).length

    try {
      for await (const event of regenConversationAudio(convId)) {
        if (event.event === 'turn_done') {
          regenProgress += 1
          const { assistant_turn_id, audio_url } = event.data
          sessionStore.update((s) => ({
            ...s,
            turns: s.turns.map((t) =>
              t.assistant_turn_id === assistant_turn_id
                ? { ...t, assistant_audio_url: audio_url, has_assistant_audio: true }
                : t,
            ),
          }))
        } else if (event.event === 'error') {
          regenError = event.data.message
          break
        }
      }
    } catch (e) {
      regenError = e instanceof Error ? e.message : 'Regen failed'
    } finally {
      regenPending = false
    }
  }
</script>

{#if $sessionStore.conversationId}
  <header class="conv-header">
    <span class="title">
      {$sessionStore.turns.length} turn{$sessionStore.turns.length === 1 ? '' : 's'}
    </span>

    <div class="actions">
      {#if regenError}
        <span class="error">{regenError}</span>
      {/if}

      <button
        class="regen-all-btn"
        on:click={handleRegenAll}
        disabled={regenPending || $sessionStore.turns.length === 0}
        title="Regenerate audio for all turns"
      >
        {#if regenPending}
          ↺ {regenProgress}/{regenTotal}
        {:else}
          ↺ Regen all audio
        {/if}
      </button>
    </div>
  </header>
{/if}

<style>
  .conv-header {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 24px;
    border-bottom: 1px solid #ececec;
    background: #fafafa;
    font-size: 0.8rem;
    color: #888;
    gap: 8px;
  }

  .title {
    font-weight: 500;
  }

  .actions {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .error {
    color: #c0392b;
    font-size: 0.75rem;
  }

  .regen-all-btn {
    background: none;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #666;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    white-space: nowrap;
  }

  .regen-all-btn:not(:disabled):hover {
    border-color: var(--kc-primary, #555);
    color: var(--kc-primary, #555);
  }

  .regen-all-btn:disabled {
    opacity: 0.45;
    cursor: default;
  }
</style>
