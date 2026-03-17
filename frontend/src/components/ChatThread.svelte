<script lang="ts">
  import { afterUpdate } from 'svelte'
  import { sessionStore } from '../lib/stores/session'
  import TurnPair from './TurnPair.svelte'

  let container: HTMLElement

  // Scroll to bottom whenever turns change.
  afterUpdate(() => {
    if (container) container.scrollTop = container.scrollHeight
  })
</script>

<div class="thread" bind:this={container}>
  {#if $sessionStore.turns.length === 0}
    <p class="empty">No messages yet — start the conversation!</p>
  {:else}
    {#each $sessionStore.turns as turn (turn.user_turn_id)}
      <TurnPair {turn} />
    {/each}
  {/if}
</div>

<style>
  .thread {
    flex: 1;
    overflow-y: auto;
    padding: 24px 24px 8px;
    display: flex;
    flex-direction: column;
  }

  .empty {
    color: #aaa;
    text-align: center;
    margin-top: 60px;
    font-size: 0.9rem;
  }
</style>
