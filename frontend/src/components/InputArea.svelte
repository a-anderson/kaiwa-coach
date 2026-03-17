<script lang="ts">
  import { sessionStore } from '../lib/stores/session'
  import { uiStore } from '../lib/stores/ui'
  import { submitTextTurn } from '../lib/api/turns'
  import { createConversation } from '../lib/api/conversations'
  import PipelineProgress from './PipelineProgress.svelte'
  import type { TurnRecord, CorrectionData, SSEStageEvent, SSECompleteEvent, SSEErrorEvent } from '../lib/types/api'

  let text = ''
  let correctionsEnabled = true
  let submitError: string | null = null

  /** Build the conversation_history string from loaded turns (see Appendix A.9). */
  function buildHistory(turns: TurnRecord[]): string {
    const lines: string[] = []
    for (const t of turns) {
      const userText = t.asr_text ?? t.user_text ?? ''
      if (userText) lines.push(`User: ${userText}`)
      if (t.reply_text) lines.push(`Assistant: ${t.reply_text}`)
    }
    return lines.join('\n')
  }

  async function handleSubmit(): Promise<void> {
    const trimmed = text.trim()
    if (!trimmed || $uiStore.isSubmitting) return

    submitError = null
    text = ''

    // Auto-create a conversation if none is active.
    let convId = $sessionStore.conversationId
    if (!convId) {
      try {
        const convo = await createConversation($sessionStore.language)
        convId = convo.id
        sessionStore.update((s) => ({ ...s, conversationId: convId! }))
      } catch (e) {
        submitError = e instanceof Error ? e.message : 'Failed to create conversation'
        text = trimmed
        return
      }
    }

    uiStore.update((s) => ({ ...s, isSubmitting: true, stageStatuses: {} }))

    const history = buildHistory($sessionStore.turns)
    let pendingCorrections: CorrectionData | null = null

    try {
      for await (const frame of submitTextTurn({
        conversationId: convId,
        text: trimmed,
        conversationHistory: history,
        correctionsEnabled,
        language: $sessionStore.language,
      })) {
        if (frame.event === 'stage') {
          const stage = frame.data as SSEStageEvent
          uiStore.update((s) => ({
            ...s,
            stageStatuses: { ...s.stageStatuses, [stage.stage]: stage.status },
          }))
          if (stage.stage === 'corrections' && stage.status === 'complete' && stage.data) {
            pendingCorrections = stage.data
          }
        } else if (frame.event === 'complete') {
          const result = frame.data as SSECompleteEvent
          const newTurn: TurnRecord = {
            user_turn_id: result.user_turn_id,
            assistant_turn_id: result.assistant_turn_id,
            user_text: trimmed,
            asr_text: null,
            reply_text: result.reply_text,
            correction: pendingCorrections,
            has_user_audio: false,
            has_assistant_audio: result.audio_url !== null,
            user_audio_url: null,
            assistant_audio_url: result.audio_url,
          }
          sessionStore.update((s) => ({ ...s, turns: [...s.turns, newTurn] }))
        } else if (frame.event === 'error') {
          submitError = (frame.data as SSEErrorEvent).message ?? 'Turn failed'
        }
      }
    } catch (e) {
      submitError = e instanceof Error ? e.message : 'Connection error'
    } finally {
      uiStore.update((s) => ({ ...s, isSubmitting: false, stageStatuses: {} }))
    }
  }

  function handleKeydown(e: KeyboardEvent): void {
    // Enter submits; Shift+Enter inserts newline.
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }
</script>

<div class="input-area">
  <PipelineProgress />

  {#if submitError}
    <p class="error">{submitError}</p>
  {/if}

  <div class="row">
    <textarea
      class="input"
      bind:value={text}
      on:keydown={handleKeydown}
      placeholder="Type a message… (Enter to send, Shift+Enter for new line)"
      rows="1"
      disabled={$uiStore.isSubmitting}
      aria-label="Message input"
    />

    <button
      class="send-btn"
      on:click={handleSubmit}
      disabled={$uiStore.isSubmitting || !text.trim()}
      aria-label="Send message"
    >
      {$uiStore.isSubmitting ? '…' : '↑'}
    </button>
  </div>

  <div class="options">
    <label class="toggle-label">
      <input type="checkbox" bind:checked={correctionsEnabled} disabled={$uiStore.isSubmitting} />
      Corrections
    </label>
  </div>
</div>

<style>
  .input-area {
    flex-shrink: 0;
    border-top: 1px solid #e0e0e0;
    padding: 10px 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .error {
    font-size: 0.8rem;
    color: #c0392b;
  }

  .row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
  }

  .input {
    flex: 1;
    resize: none;
    border: 1px solid #d0d0d0;
    border-radius: 10px;
    padding: 9px 13px;
    font-size: 0.9rem;
    font-family: inherit;
    line-height: 1.4;
    outline: none;
    field-sizing: content; /* auto-grow with content */
    max-height: 180px;
    overflow-y: auto;
  }

  .input:focus {
    border-color: var(--kc-primary, #555);
  }

  .input:disabled {
    background: #f8f8f8;
    color: #aaa;
  }

  .send-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: var(--kc-primary, #333);
    color: #fff;
    border: none;
    font-size: 1.1rem;
    cursor: pointer;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.15s;
  }

  .send-btn:disabled {
    opacity: 0.35;
    cursor: default;
  }

  .send-btn:not(:disabled):hover {
    opacity: 0.85;
  }

  .options {
    display: flex;
    gap: 12px;
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.78rem;
    color: #888;
    cursor: pointer;
    user-select: none;
  }

  .toggle-label input {
    accent-color: var(--kc-primary, #333);
    cursor: pointer;
  }
</style>
