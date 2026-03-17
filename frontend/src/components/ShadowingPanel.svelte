<script lang="ts">
  /**
   * ShadowingPanel — rendered only while shadowingTurnId is non-null (App.svelte
   * gates mounting with {#if $uiStore.shadowingTurnId}). Because the component is
   * freshly mounted each time the panel opens, onMount always fires with the DOM
   * ready and waveContainer bound.
   */
  import { onMount, onDestroy, tick } from 'svelte'
  import { uiStore } from '../lib/stores/ui'
  import { sessionStore } from '../lib/stores/session'
  import AudioPlayer from './AudioPlayer.svelte'
  import WaveSurfer from 'wavesurfer.js'
  import RecordPlugin from 'wavesurfer.js/dist/plugins/record.js'

  // ── Reference turn ────────────────────────────────────────────────────
  $: turn = $sessionStore.turns.find(
    (t) => t.assistant_turn_id === $uiStore.shadowingTurnId,
  ) ?? null

  function close() {
    uiStore.update((s) => ({ ...s, shadowingTurnId: null }))
  }

  // ── Attempt recording state ───────────────────────────────────────────
  type Phase = 'idle' | 'recording' | 'done'
  let phase: Phase = 'idle'
  let attemptUrl: string | null = null

  let waveContainer: HTMLElement
  let ws: WaveSurfer | null = null
  let record: InstanceType<typeof RecordPlugin> | null = null

  function mountRecorder() {
    if (!waveContainer) return
    ws = WaveSurfer.create({
      container: waveContainer,
      waveColor: getComputedStyle(document.documentElement)
        .getPropertyValue('--kc-waveform-active').trim() || '#555',
      height: 48,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      interact: false,
    })
    record = ws.registerPlugin(RecordPlugin.create({ renderRecordedAudio: false }))
    record.on('record-end', (blob: Blob) => {
      attemptUrl = URL.createObjectURL(blob)
      phase = 'done'
    })
  }

  onMount(() => {
    mountRecorder()
  })

  onDestroy(() => {
    record?.stopRecording()
    ws?.destroy()
    if (attemptUrl) URL.revokeObjectURL(attemptUrl)
  })

  // Incremented on Try Again to force the recorder subtree to remount via {#key}.
  let attemptKey = 0

  async function startRecording() {
    if (!record) return
    await record.startRecording()
    phase = 'recording'
  }

  function stopRecording() {
    record?.stopRecording()
    // phase transitions to 'done' via record-end event
  }

  async function tryAgain() {
    if (attemptUrl) {
      URL.revokeObjectURL(attemptUrl)
      attemptUrl = null
    }
    // Destroy current WaveSurfer instance before the DOM node is replaced.
    record?.stopRecording()
    ws?.destroy()
    ws = null
    record = null
    phase = 'idle'
    attemptKey += 1
    // Wait for {#key} to swap in the new waveContainer div, then remount.
    await tick()
    mountRecorder()
  }
</script>

<section class="panel" aria-label="Shadowing panel">
  <!-- Header -->
  <div class="header">
    <span class="label">
      Shadowing: <em>{turn?.reply_text ?? ''}</em>
    </span>
    <button class="close-btn" on:click={close} aria-label="Close shadowing panel">✕</button>
  </div>

  <!-- Body -->
  <div class="body">
    <!-- Reference -->
    <div class="column">
      <p class="col-title">Reference</p>
      {#if turn?.assistant_audio_url}
        <AudioPlayer src={turn.assistant_audio_url} />
      {:else}
        <p class="no-audio">No audio — regenerate first.</p>
      {/if}
    </div>

    <!-- Attempt -->
    <div class="column">
      <p class="col-title">Your attempt</p>

      {#if phase === 'done' && attemptUrl}
        <AudioPlayer src={attemptUrl} />
        <button class="try-again-btn" on:click={tryAgain}>Try Again</button>
      {:else}
        {#key attemptKey}
          <!-- waveContainer is bound fresh on every key change -->
          <div class="wave" bind:this={waveContainer} />
        {/key}

        {#if phase === 'idle'}
          <button class="record-btn" on:click={startRecording} aria-label="Start recording">
            🎙 Record
          </button>
        {:else if phase === 'recording'}
          <div class="recording-row">
            <span class="rec-dot" aria-hidden="true">●</span>
            <button class="stop-btn" on:click={stopRecording} aria-label="Stop recording">
              ⏹ Stop
            </button>
          </div>
        {/if}
      {/if}
    </div>
  </div>
</section>

<style>
  .panel {
    flex-shrink: 0;
    border-top: 2px solid var(--kc-primary, #555);
    background: #fff;
    display: flex;
    flex-direction: column;
    max-height: 260px;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: color-mix(in srgb, var(--kc-primary, #555) 8%, white);
    border-bottom: 1px solid #ececec;
    gap: 12px;
  }

  .label {
    font-size: 0.82rem;
    color: #444;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .label em {
    font-style: normal;
    font-weight: 500;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 0.9rem;
    color: #888;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    flex-shrink: 0;
    transition: color 0.15s;
  }

  .close-btn:hover {
    color: #333;
  }

  .body {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .column {
    flex: 1;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
    overflow: hidden;
  }

  .column:first-child {
    border-right: 1px solid #ececec;
  }

  .col-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0;
  }

  .no-audio {
    font-size: 0.8rem;
    color: #bbb;
    font-style: italic;
    margin: 0;
  }

  .wave {
    width: 100%;
    min-height: 48px;
  }

  .record-btn,
  .stop-btn,
  .try-again-btn {
    align-self: flex-start;
    padding: 5px 14px;
    border-radius: 6px;
    border: none;
    font-size: 0.8rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .record-btn {
    background: var(--kc-primary, #333);
    color: #fff;
  }

  .stop-btn {
    background: #c0392b;
    color: #fff;
  }

  .try-again-btn {
    background: #eee;
    color: #555;
  }

  .record-btn:hover,
  .stop-btn:hover,
  .try-again-btn:hover {
    opacity: 0.85;
  }

  .recording-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .rec-dot {
    color: #c0392b;
    font-size: 0.8rem;
    animation: blink 1s step-start infinite;
  }

  @keyframes blink {
    50% { opacity: 0; }
  }
</style>
