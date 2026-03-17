<script lang="ts">
  /**
   * AudioRecorder — live waveform via WaveSurfer Record plugin.
   *
   * Events:
   *   on:recorded  — fires with { blob: Blob } when recording stops
   *   on:cancel    — fires when user discards the recording
   */
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import WaveSurfer from 'wavesurfer.js'
  import RecordPlugin from 'wavesurfer.js/dist/plugins/record.js'

  const dispatch = createEventDispatcher<{ recorded: { blob: Blob }; cancel: void }>()

  let container: HTMLElement
  let ws: WaveSurfer | null = null
  let record: InstanceType<typeof RecordPlugin> | null = null
  let recording = false
  let blob: Blob | null = null

  onMount(() => {
    ws = WaveSurfer.create({
      container,
      waveColor: getComputedStyle(document.documentElement)
        .getPropertyValue('--kc-waveform-active').trim() || '#555',
      height: 36,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      interact: false,
    })

    record = ws.registerPlugin(RecordPlugin.create({ renderRecordedAudio: false }))

    record.on('record-end', (b: Blob) => {
      blob = b
      recording = false
    })
  })

  onDestroy(() => {
    record?.stopRecording()
    ws?.destroy()
    ws = null
    record = null
  })

  async function toggleRecording() {
    if (!record) return
    if (recording) {
      record.stopRecording()
    } else {
      blob = null
      await record.startRecording()
      recording = true
    }
  }

  function send() {
    if (blob) dispatch('recorded', { blob })
  }

  function cancel() {
    blob = null
    dispatch('cancel')
  }
</script>

<div class="recorder">
  <div class="wave" bind:this={container} />

  <div class="controls">
    {#if blob}
      <button class="action-btn send-btn" on:click={send} aria-label="Send recording">
        Send ↑
      </button>
      <button class="action-btn cancel-btn" on:click={cancel} aria-label="Discard recording">
        Discard
      </button>
    {:else}
      <button
        class="action-btn"
        class:recording
        on:click={toggleRecording}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
      >
        {recording ? '⏹ Stop' : '🎙 Record'}
      </button>
      {#if recording}
        <span class="rec-indicator" aria-hidden="true">●</span>
      {/if}
    {/if}
  </div>
</div>

<style>
  .recorder {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .wave {
    width: 100%;
    min-height: 36px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-btn {
    padding: 5px 12px;
    border-radius: 6px;
    border: none;
    font-size: 0.8rem;
    cursor: pointer;
    transition: opacity 0.15s;
    background: var(--kc-primary, #333);
    color: #fff;
  }

  .action-btn:hover {
    opacity: 0.85;
  }

  .action-btn.recording {
    background: #c0392b;
  }

  .cancel-btn {
    background: #aaa;
  }

  .send-btn {
    background: var(--kc-primary, #333);
  }

  .rec-indicator {
    color: #c0392b;
    font-size: 0.9rem;
    animation: blink 1s step-start infinite;
  }

  @keyframes blink {
    50% { opacity: 0; }
  }
</style>
