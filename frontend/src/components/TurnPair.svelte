<script lang="ts">
  import type { TurnRecord } from '../lib/types/api'
  import { sessionStore } from '../lib/stores/session'
  import { uiStore } from '../lib/stores/ui'
  import { regenTurnAudio } from '../lib/api/regen'
  import UserBubble from './UserBubble.svelte'
  import AssistantBubble from './AssistantBubble.svelte'
  import CorrectionCard from './CorrectionCard.svelte'

  export let turn: TurnRecord

  $: userText = turn.asr_text ?? turn.user_text ?? ''

  let regenPending = false
  let regenError: string | null = null

  async function handleRegen() {
    if (!turn.assistant_turn_id || regenPending) return
    regenPending = true
    regenError = null
    try {
      const audioUrl = await regenTurnAudio(turn.assistant_turn_id)
      sessionStore.update((s) => ({
        ...s,
        turns: s.turns.map((t) =>
          t.assistant_turn_id === turn.assistant_turn_id
            ? { ...t, assistant_audio_url: audioUrl, has_assistant_audio: true }
            : t,
        ),
      }))
    } catch (e) {
      regenError = e instanceof Error ? e.message : 'Regen failed'
    } finally {
      regenPending = false
    }
  }
</script>

<div class="pair">
  {#if userText || turn.user_audio_url}
    <UserBubble text={userText} audioUrl={turn.user_audio_url} />
  {/if}

  {#if turn.correction}
    <CorrectionCard correction={turn.correction} />
  {/if}

  {#if turn.status === 'pending' && !turn.reply_text}
    <div class="typing-bubble" aria-label="Assistant is typing">
      <span class="dot" />
      <span class="dot" />
      <span class="dot" />
    </div>
  {/if}

  {#if turn.reply_text}
    <AssistantBubble
      text={turn.reply_text}
      audioUrl={turn.assistant_audio_url}
      hasAudio={turn.has_assistant_audio}
      autoplay={turn.autoplay ?? false}
      {regenPending}
      on:regen={handleRegen}
      on:shadow={() => uiStore.update((s) => ({ ...s, shadowingTurnId: turn.assistant_turn_id }))}
    />
    {#if regenError}
      <p class="regen-error">{regenError}</p>
    {/if}
  {/if}
</div>

<style>
  .pair {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 20px;
  }

  .regen-error {
    font-size: 0.75rem;
    color: #c0392b;
    margin: 0;
  }

  .typing-bubble {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    background: var(--kc-bot-bubble, #f9f9f9);
    border: 1px solid color-mix(in srgb, var(--kc-secondary, #aaa) 30%, transparent);
    align-self: flex-start;
    width: fit-content;
  }

  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--kc-secondary, #aaa) 70%, transparent);
    animation: bounce 1.2s ease-in-out infinite;
  }

  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }
</style>
