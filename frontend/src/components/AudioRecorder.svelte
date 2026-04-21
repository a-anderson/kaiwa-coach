<script lang="ts">
  /**
   * AudioRecorder — live waveform via WaveSurfer Record plugin.
   *
   * Events:
   *   on:recorded  — fires with { blob: Blob } when recording is ready to submit
   *   on:cancel    — fires when user cancels (chat: closes recorder; monologue: clears ready blob)
   */
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import WaveSurfer from 'wavesurfer.js'
  import RecordPlugin from 'wavesurfer.js/dist/plugins/record.js'
  import AudioPlayer from './AudioPlayer.svelte'

  const dispatch = createEventDispatcher<{ recorded: { blob: Blob }; cancel: void }>()

  export let autostart = false
  /** When false (monologue): hides Send button and auto-emits 'recorded' when recording stops. */
  export let showSendButton: boolean = true

  let container: HTMLElement
  let ws: WaveSurfer | null = null
  let record: InstanceType<typeof RecordPlugin> | null = null
  let recording = false
  let blob: Blob | null = null
  let previewUrl: string | null = null

  onMount(async () => {
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
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      previewUrl = URL.createObjectURL(b)
      if (!showSendButton) dispatch('recorded', { blob: b })
    })

    if (autostart) {
      await record.startRecording()
      recording = true
    }
  })

  onDestroy(() => {
    record?.stopRecording()
    ws?.destroy()
    ws = null
    record = null
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    previewUrl = null
  })

  async function toggleRecording() {
    if (!record) return
    if (recording) {
      record.stopRecording()
    } else {
      blob = null
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null }
      await record.startRecording()
      recording = true
    }
  }

  function send() {
    if (blob) {
      dispatch('recorded', { blob })
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null }
    }
  }

  async function reRecord() {
    blob = null
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null }
    // In monologue context, tell parent the ready blob is gone.
    if (!showSendButton) dispatch('cancel')
    if (record) {
      await record.startRecording()
      recording = true
    }
  }

  function cancel() {
    blob = null
    if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null }
    dispatch('cancel')
  }
</script>

<div class="recorder">
  <div class="wave" bind:this={container} class:hidden={!!blob} />

  {#if blob && previewUrl}
    <AudioPlayer src={previewUrl} variant="user" />
  {/if}

  <div class="controls">
    {#if blob}
      <button class="action-btn rerecord-btn" on:click={reRecord} aria-label="Re-record">
        ↺ Re-record
      </button>
      {#if showSendButton}
        <button class="action-btn cancel-btn" on:click={cancel} aria-label="Cancel recording">
          ✕ Cancel
        </button>
        <button class="action-btn send-btn" on:click={send} aria-label="Send recording">
          Send ↑
        </button>
      {/if}
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

  .wave.hidden {
    display: none;
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

  .rerecord-btn {
    background: #aaa;
  }

  .cancel-btn {
    background: transparent;
    color: #888;
    border: 1px solid #ccc;
  }

  .cancel-btn:hover {
    background: #f5f5f5;
    opacity: 1;
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
