<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { ConversationSummary } from '../lib/types/api'

  export let conversation: ConversationSummary
  export let active: boolean = false

  const dispatch = createEventDispatcher<{ select: string; delete: string }>()

  const LANGUAGE_FLAGS: Record<string, string> = {
    ja: '🇯🇵',
    fr: '🇫🇷',
    en: '🇺🇸',
    es: '🇪🇸',
    it: '🇮🇹',
    pt: '🇵🇹',
  }

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return ''
    const d = new Date(dateStr)
    const now = new Date()
    const diffDays = Math.floor((now.getTime() - d.getTime()) / 86_400_000)
    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return d.toLocaleDateString(undefined, { weekday: 'long' })
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  }
</script>

<li class="item" class:active aria-current={active ? 'page' : undefined}>
  <button class="item-btn" on:click={() => dispatch('select', conversation.id)}>
    <span class="flag" aria-hidden="true">
      {LANGUAGE_FLAGS[conversation.language] ?? '🌐'}
    </span>
    <span class="text">
      <span class="title">{conversation.title ?? 'Untitled'}</span>
      {#if conversation.preview_text}
        <span class="preview">{conversation.preview_text}</span>
      {/if}
      {#if conversation.updated_at}
        <span class="date">{formatDate(conversation.updated_at)}</span>
      {/if}
    </span>
  </button>

  <button
    class="delete-btn"
    title="Delete conversation"
    aria-label="Delete conversation"
    on:click={() => dispatch('delete', conversation.id)}
  >
    ✕
  </button>
</li>

<style>
  .item {
    display: flex;
    align-items: stretch;
    list-style: none;
    transition: background 0.1s;
  }

  .item:hover,
  .item:focus-within {
    background: #f0f0f0;
  }

  .item.active {
    background: var(--kc-user-bubble, #f0f0f0);
    border-left: 3px solid var(--kc-primary, #333);
  }

  .item-btn {
    flex: 1;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 10px 8px 10px 12px;
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
    min-width: 0;
    outline: none;
  }

  .item-btn:focus-visible {
    outline: 2px solid var(--kc-primary, #333);
    outline-offset: -2px;
  }

  .flag {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 1px;
  }

  .text {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1a1a1a;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .preview {
    font-size: 0.78rem;
    color: #777;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .date {
    font-size: 0.72rem;
    color: #aaa;
  }

  .delete-btn {
    flex-shrink: 0;
    align-self: center;
    background: none;
    border: none;
    cursor: pointer;
    color: #bbb;
    font-size: 0.8rem;
    padding: 4px 8px;
    border-radius: 3px;
    line-height: 1;
    opacity: 0;
    transition: opacity 0.1s, color 0.1s;
  }

  .item:hover .delete-btn,
  .item:focus-within .delete-btn {
    opacity: 1;
  }

  .delete-btn:hover {
    color: #c0392b;
    background: #fdf2f2;
  }
</style>
