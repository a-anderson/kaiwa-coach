<script lang="ts">
  import { sessionStore } from '../lib/stores/session'
  import { uiStore } from '../lib/stores/ui'
  import { submitTextTurn, submitAudioTurn } from '../lib/api/turns'
  import { createConversation } from '../lib/api/conversations'
  import PipelineProgress from './PipelineProgress.svelte'
  import AudioRecorder from './AudioRecorder.svelte'
  import type { TurnRecord, CorrectionData, SSEStageEvent, SSECompleteEvent, SSEErrorEvent } from '../lib/types/api'
  import { createEventDispatcher, onDestroy } from 'svelte'
  import { isSupportedLanguage } from '../lib/constants'

  const dispatch = createEventDispatcher<{ turncomplete: void }>()

  let text = ''
  let correctionsEnabled = true
  let submitError: string | null = null
  let showRecorder = false
  let lastBlobUrl: string | null = null

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

  /** Ensure a conversation exists, creating one if needed. Returns the id or null on error. */
  async function ensureConversation(): Promise<string | null> {
    let convId = $sessionStore.conversationId
    if (convId) return convId
    try {
      const convo = await createConversation($sessionStore.language)
      convId = convo.id
      sessionStore.update((s) => ({ ...s, conversationId: convId! }))
      return convId
    } catch (e) {
      submitError = e instanceof Error ? e.message : 'Failed to create conversation'
      return null
    }
  }

  /** Consume an SSE stream, progressively updating the pending turn then committing it. */
  async function drainStream(
    stream: AsyncGenerator<import('../lib/types/api').SSEEvent>,
  ): Promise<void> {
    try {
      for await (const frame of stream) {
        if (frame.event === 'stage') {
          const stage = frame.data as SSEStageEvent
          uiStore.update((s) => ({
            ...s,
            stageStatuses: { ...s.stageStatuses, [stage.stage]: stage.status },
          }))
          if (stage.status === 'complete') {
            if (stage.stage === 'asr' && stage.transcript) {
              sessionStore.update((s) => ({
                ...s,
                pendingTurn: s.pendingTurn ? { ...s.pendingTurn, asr_text: stage.transcript! } : null,
              }))
            } else if (stage.stage === 'llm' && stage.reply) {
              sessionStore.update((s) => ({
                ...s,
                pendingTurn: s.pendingTurn ? { ...s.pendingTurn, reply_text: stage.reply! } : null,
              }))
            } else if (stage.stage === 'corrections' && stage.data) {
              sessionStore.update((s) => ({
                ...s,
                pendingTurn: s.pendingTurn ? { ...s.pendingTurn, correction: stage.data! } : null,
              }))
            } else if (stage.stage === 'tts' && stage.audio_url) {
              sessionStore.update((s) => ({
                ...s,
                pendingTurn: s.pendingTurn
                  ? { ...s.pendingTurn, assistant_audio_url: stage.audio_url!, has_assistant_audio: true }
                  : null,
              }))
              uiStore.update((s) => ({ ...s, autoplayPending: true }))
            }
          }
        } else if (frame.event === 'complete') {
          const result = frame.data as SSECompleteEvent
          sessionStore.update((s) => {
            const p = s.pendingTurn
            const finalTurn: TurnRecord = {
              user_turn_id: result.user_turn_id,
              assistant_turn_id: result.assistant_turn_id,
              user_text: p?.user_text ?? null,
              asr_text: result.asr_text ?? p?.asr_text ?? null,
              reply_text: result.reply_text,
              correction: p?.correction ?? null,
              has_user_audio: p?.has_user_audio ?? false,
              has_assistant_audio: result.audio_url !== null,
              user_audio_url: p?.user_audio_url ?? null,
              assistant_audio_url: result.audio_url,
            }
            return { ...s, turns: [...s.turns, finalTurn], pendingTurn: null }
          })
          uiStore.update((s) => ({ ...s, autoplayPending: false }))
          dispatch('turncomplete')
        } else if (frame.event === 'error') {
          submitError = (frame.data as SSEErrorEvent).message ?? 'Turn failed'
        }
      }
    } catch (e) {
      submitError = e instanceof Error ? e.message : 'Connection error'
    }
  }

  async function handleSubmit(): Promise<void> {
    const trimmed = text.trim()
    if (!trimmed || $uiStore.isSubmitting) return

    if (!isSupportedLanguage($sessionStore.language)) {
      submitError = `Unsupported language: "${$sessionStore.language}". Please select a language from the menu.`
      return
    }

    submitError = null
    text = ''

    const convId = await ensureConversation()
    if (!convId) { text = trimmed; return }

    const history = buildHistory($sessionStore.turns)

    sessionStore.update((s) => ({
      ...s,
      pendingTurn: {
        user_turn_id: '',
        assistant_turn_id: null,
        user_text: trimmed,
        asr_text: null,
        reply_text: null,
        correction: null,
        has_user_audio: false,
        has_assistant_audio: false,
        user_audio_url: null,
        assistant_audio_url: null,
        status: 'pending',
      },
    }))
    uiStore.update((s) => ({ ...s, isSubmitting: true, stageStatuses: {} }))

    try {
      await drainStream(
        submitTextTurn({ conversationId: convId, text: trimmed, conversationHistory: history, correctionsEnabled, language: $sessionStore.language }),
      )
    } finally {
      sessionStore.update((s) => ({ ...s, pendingTurn: null }))
      uiStore.update((s) => ({ ...s, isSubmitting: false, stageStatuses: {}, autoplayPending: false }))
    }
  }

  async function handleAudioRecorded(e: CustomEvent<{ blob: Blob }>): Promise<void> {
    showRecorder = false
    if ($uiStore.isSubmitting) return

    if (!isSupportedLanguage($sessionStore.language)) {
      submitError = `Unsupported language: "${$sessionStore.language}". Please select a language from the menu.`
      return
    }

    submitError = null
    const convId = await ensureConversation()
    if (!convId) return

    const history = buildHistory($sessionStore.turns)
    // Revoke any previous blob URL before creating a new one.
    if (lastBlobUrl) {
      URL.revokeObjectURL(lastBlobUrl)
      lastBlobUrl = null
    }
    const localUrl = URL.createObjectURL(e.detail.blob)
    lastBlobUrl = localUrl

    sessionStore.update((s) => ({
      ...s,
      pendingTurn: {
        user_turn_id: '',
        assistant_turn_id: null,
        user_text: null,
        asr_text: null,
        reply_text: null,
        correction: null,
        has_user_audio: true,
        has_assistant_audio: false,
        user_audio_url: localUrl,
        assistant_audio_url: null,
        status: 'pending',
      },
    }))
    uiStore.update((s) => ({ ...s, isSubmitting: true, stageStatuses: {} }))

    try {
      await drainStream(
        submitAudioTurn({ conversationId: convId, audioBlob: e.detail.blob, conversationHistory: history, correctionsEnabled, language: $sessionStore.language }),
      )
    } finally {
      sessionStore.update((s) => ({ ...s, pendingTurn: null }))
      uiStore.update((s) => ({ ...s, isSubmitting: false, stageStatuses: {}, autoplayPending: false }))
    }
  }

  onDestroy(() => {
    if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl)
  })

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

  {#if showRecorder}
    <AudioRecorder
      autostart
      on:recorded={handleAudioRecorded}
      on:cancel={() => { showRecorder = false }}
    />
  {:else}
    <div class="row">
      <button
        class="mic-btn"
        on:click={() => { showRecorder = true }}
        disabled={$uiStore.isSubmitting}
        aria-label="Record audio"
      >
        🎙
      </button>

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
  {/if}

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

  .mic-btn,
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

  .mic-btn {
    background: transparent;
    border: 1px solid #d0d0d0;
    color: #555;
    font-size: 1rem;
  }

  .mic-btn:not(:disabled):hover {
    border-color: var(--kc-primary, #555);
    color: var(--kc-primary, #555);
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
