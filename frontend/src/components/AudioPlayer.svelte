<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import WaveSurfer from 'wavesurfer.js'

  export let src: string
  export let variant: 'user' | 'assistant' = 'user'
  export let autoplay: boolean = false

  let container: HTMLElement
  let ws: WaveSurfer | null = null
  let playing = false
  let ready = false

  $: accentVar = variant === 'assistant' ? 'var(--kc-secondary, #555)' : 'var(--kc-primary, #333)'

  onMount(() => {
    const cs = getComputedStyle(document.documentElement)
    const activeVar  = variant === 'assistant' ? '--kc-waveform-bot-active'   : '--kc-waveform-active'
    const inactiveVar = variant === 'assistant' ? '--kc-waveform-bot-inactive' : '--kc-waveform-inactive'

    ws = WaveSurfer.create({
      container,
      waveColor: cs.getPropertyValue(inactiveVar).trim() || '#ccc',
      progressColor: cs.getPropertyValue(activeVar).trim() || '#555',
      height: 36,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      interact: true,
    })

    ws.on('ready', () => { ready = true; if (autoplay) ws?.play() })
    ws.on('play', () => { playing = true })
    ws.on('pause', () => { playing = false })
    ws.on('finish', () => { playing = false })
    // ws assignment triggers the reactive statement below, which performs the initial load.
  })

  // Single code path for all loads: fires on initial mount (ws goes null→instance)
  // and on every subsequent src change.
  $: if (ws && src) {
    ready = false
    playing = false
    ws.load(src)
  }

  onDestroy(() => {
    ws?.destroy()
    ws = null
  })

  function togglePlayback() {
    ws?.playPause()
  }
</script>

<div class="player" style="--player-accent: {accentVar}">
  <button
    class="play-btn"
    on:click={togglePlayback}
    disabled={!ready}
    aria-label={playing ? 'Pause' : 'Play'}
  >
    {playing ? '⏸' : '▶'}
  </button>
  <div class="wave" bind:this={container} />
</div>

<style>
  .player {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }

  .play-btn {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: none;
    background: var(--player-accent, #333);
    color: #fff;
    font-size: 0.85rem;
    cursor: pointer;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.15s;
  }

  .play-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .play-btn:not(:disabled):hover {
    opacity: 0.8;
  }

  .wave {
    flex: 1;
    min-width: 0;
  }
</style>
