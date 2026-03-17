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

  {#if turn.reply_text}
    <AssistantBubble
      text={turn.reply_text}
      audioUrl={turn.assistant_audio_url}
      hasAudio={turn.has_assistant_audio}
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
</style>
