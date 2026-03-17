<script lang="ts">
  import type { TurnRecord } from '../lib/types/api'
  import UserBubble from './UserBubble.svelte'
  import AssistantBubble from './AssistantBubble.svelte'
  import CorrectionCard from './CorrectionCard.svelte'

  export let turn: TurnRecord

  $: userText = turn.asr_text ?? turn.user_text ?? ''
</script>

<div class="pair">
  {#if userText}
    <UserBubble text={userText} />
  {/if}

  {#if turn.correction}
    <CorrectionCard correction={turn.correction} />
  {/if}

  {#if turn.reply_text}
    <AssistantBubble
      text={turn.reply_text}
      audioUrl={turn.assistant_audio_url}
      hasAudio={turn.has_assistant_audio}
    />
  {/if}
</div>

<style>
  .pair {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 20px;
  }
</style>
