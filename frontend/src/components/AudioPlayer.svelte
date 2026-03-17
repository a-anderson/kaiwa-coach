<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import WaveSurfer from 'wavesurfer.js'

  export let src: string

  let container: HTMLElement
  let ws: WaveSurfer | null = null
  let playing = false
  let ready = false

  onMount(() => {
    ws = WaveSurfer.create({
      container,
      waveColor: getComputedStyle(document.documentElement)
        .getPropertyValue('--kc-waveform-inactive').trim() || '#ccc',
      progressColor: getComputedStyle(document.documentElement)
        .getPropertyValue('--kc-waveform-active').trim() || '#555',
      height: 36,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      interact: true,
      url: src,
    })

    ws.on('ready', () => { ready = true })
    ws.on('play', () => { playing = true })
    ws.on('pause', () => { playing = false })
    ws.on('finish', () => { playing = false })
  })

  onDestroy(() => {
    ws?.destroy()
    ws = null
  })

  function togglePlayback() {
    ws?.playPause()
  }
</script>

<div class="player">
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
    background: var(--kc-primary, #333);
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
