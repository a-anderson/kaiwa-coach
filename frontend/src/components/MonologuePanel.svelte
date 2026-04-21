<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import { sessionStore } from '../lib/stores/session'
  import { LANGUAGE_NATIVE_NAMES } from '../lib/constants'
  import { createMonologueConversation } from '../lib/api/conversations'
  import { submitMonologueText, submitMonologueAudio } from '../lib/api/monologue'
  import AudioRecorder from './AudioRecorder.svelte'
  import AudioPlayer from './AudioPlayer.svelte'
  import CopyButton from './CopyButton.svelte'
  import type { CorrectionData, TurnRecord, MonologueCorrections, MonologueSummary } from '../lib/types/api'

  const dispatch = createEventDispatcher<{ turncomplete: void }>()

  type InputMode = 'text' | 'mic' | 'file'

  let inputMode: InputMode = 'text'
  let textInput = ''
  let audioBlob: Blob | null = null
  let fileInput: HTMLInputElement
  let filePreviewUrl: string | null = null

  let submitting = false
  let stageStatuses: Record<string, 'running' | 'complete'> = {}
  let submitError: string | null = null

  let resultTranscript: string | null = null
  let resultCorrections: MonologueCorrections | null = null
  let resultSummary: MonologueSummary | null = null
  let hasResult = false

  $: language = $sessionStore.language
  $: languageDisplay = LANGUAGE_NATIVE_NAMES[language] ?? language.toUpperCase()

  // Track the last conversation ID loaded into the results display.
  // Prevents the store subscription from overwriting live SSE results when
  // handleSubmit pushes the completed turn into the store.
  let _lastHandledId: string | null = null
  let _unsub: (() => void) | null = null

  onMount(() => {
    _unsub = sessionStore.subscribe(($s) => {
      const id = $s.conversationId
      const type = $s.conversationType

      if (
        type === 'monologue' &&
        id !== null &&
        id !== _lastHandledId &&
        !submitting &&
        $s.turns.length > 0
      ) {
        // Sidebar selection or tab-switch restore — load from already-fetched turns.
        _lastHandledId = id
        const turn = $s.turns[0]
        resultTranscript = turn.user_text ?? turn.asr_text ?? null
        resultCorrections = turn.correction ?? null
        resultSummary = null  // summary is not persisted
        hasResult = true
      } else if (type === 'monologue' && id === null && !submitting) {
        // "New Monologue" was clicked — reset to empty form.
        _lastHandledId = null
        hasResult = false
        resultTranscript = null
        resultCorrections = null
        resultSummary = null
        submitError = null
        stageStatuses = {}
      }
    })
  })

  onDestroy(() => {
    _unsub?.()
    if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl)
  })

  function setMode(mode: InputMode) {
    if (mode === inputMode) return
    inputMode = mode
    audioBlob = null
    textInput = ''
    submitError = null
    if (filePreviewUrl) { URL.revokeObjectURL(filePreviewUrl); filePreviewUrl = null }
  }

  function handleRecorded(e: CustomEvent<{ blob: Blob }>) {
    audioBlob = e.detail.blob
  }

  function handleRecorderRerecord() {
    audioBlob = null
  }

  function handleFileChange() {
    const file = fileInput?.files?.[0]
    if (filePreviewUrl) { URL.revokeObjectURL(filePreviewUrl); filePreviewUrl = null }
    if (file) {
      audioBlob = file
      filePreviewUrl = URL.createObjectURL(file)
    } else {
      audioBlob = null
    }
  }

  async function handleSubmit() {
    if (submitting) return
    if (inputMode === 'text' && !textInput.trim()) return
    if ((inputMode === 'mic' || inputMode === 'file') && !audioBlob) return

    submitting = true
    submitError = null
    stageStatuses = {}
    hasResult = false
    resultTranscript = null
    resultCorrections = null
    resultSummary = null

    try {
      const { conversation_id } = await createMonologueConversation()

      // Set _lastHandledId before the store update that follows SSE completion,
      // so the subscription does not overwrite the live results we are about to show.
      _lastHandledId = conversation_id

      const generator =
        inputMode === 'text'
          ? submitMonologueText({ conversation_id, text: textInput.trim() })
          : submitMonologueAudio({ conversation_id, audio: audioBlob! })

      for await (const event of generator) {
        if (event.event === 'stage') {
          stageStatuses = { ...stageStatuses, [event.data.stage]: event.data.status }
          if (event.data.stage === 'asr' && event.data.status === 'complete') {
            resultTranscript = (event.data as { transcript?: string }).transcript ?? null
          }
        } else if (event.event === 'complete') {
          resultTranscript = resultTranscript ?? event.data.asr_text ?? event.data.input_text ?? null
          resultCorrections = event.data.corrections
          resultSummary = event.data.summary
          hasResult = true

          // Push a synthetic TurnRecord into the store so results survive a tab switch.
          const syntheticTurn: TurnRecord = {
            user_turn_id: event.data.user_turn_id,
            assistant_turn_id: null,
            user_text: event.data.input_text ?? null,
            asr_text: event.data.asr_text ?? null,
            reply_text: null,
            correction: event.data.corrections as unknown as CorrectionData,
            has_user_audio: inputMode !== 'text',
            has_assistant_audio: false,
            user_audio_url: null,
            assistant_audio_url: null,
          }
          sessionStore.update((s) => ({
            ...s,
            conversationId: conversation_id,
            conversationType: 'monologue',
            turns: [syntheticTurn],
          }))
          dispatch('turncomplete')
        } else if (event.event === 'error') {
          submitError = event.data.message || 'An error occurred'
        }
      }
    } catch (e) {
      submitError = e instanceof Error ? e.message : 'Submission failed'
    } finally {
      submitting = false
    }
  }

  // ── Copy helpers ─────────────────────────────────────────────────────────────

  function correctionsCopyText(c: MonologueCorrections): string {
    const parts: string[] = []
    if (c.errors.length > 0) {
      parts.push('Errors:\n' + c.errors.map((e) => `• ${e}`).join('\n'))
    }
    if (c.corrected) parts.push(`Corrected: ${c.corrected}`)
    if (c.native) parts.push(`Natural phrasing: ${c.native}`)
    if (c.explanation) parts.push(`Explanation: ${c.explanation}`)
    return parts.join('\n\n')
  }

  function summaryCopyText(s: MonologueSummary): string {
    const parts: string[] = []
    if (s.improvement_areas.length > 0) {
      parts.push(
        'Areas to focus on:\n' +
        s.improvement_areas.map((a, i) => `${i + 1}. ${a}`).join('\n'),
      )
    }
    if (s.overall_assessment) parts.push(`Overall: ${s.overall_assessment}`)
    return parts.join('\n\n')
  }

  // ── Progress ─────────────────────────────────────────────────────────────────

  const STAGE_LABELS: Record<string, string> = {
    asr: 'Transcribing',
    corrections: 'Checking corrections',
    summary: 'Generating summary',
  }
  const STAGE_ORDER = ['asr', 'corrections', 'summary']

  $: progressStages = STAGE_ORDER
    .filter((s) => s in stageStatuses)
    .map((s) => ({ name: s, status: stageStatuses[s], label: STAGE_LABELS[s] }))
</script>

<div class="monologue-panel">
  <div class="monologue-inner">

    <h2 class="panel-title">Analyse in <span class="lang-label">{languageDisplay}</span></h2>

    <!-- Mode selector -->
    <div class="mode-tabs" role="tablist">
      <button
        role="tab"
        aria-selected={inputMode === 'text'}
        class="mode-tab"
        class:active={inputMode === 'text'}
        on:click={() => setMode('text')}
        disabled={submitting}
      >Text</button>
      <button
        role="tab"
        aria-selected={inputMode === 'mic'}
        class="mode-tab"
        class:active={inputMode === 'mic'}
        on:click={() => setMode('mic')}
        disabled={submitting}
      >Mic</button>
      <button
        role="tab"
        aria-selected={inputMode === 'file'}
        class="mode-tab"
        class:active={inputMode === 'file'}
        on:click={() => setMode('file')}
        disabled={submitting}
      >Upload</button>
    </div>

    <!-- Input area -->
    {#if inputMode === 'text'}
      <div class="textarea-wrap">
        <textarea
          class="text-input"
          bind:value={textInput}
          placeholder="Type or paste your {languageDisplay} text here…"
          disabled={submitting}
          rows={6}
        ></textarea>
        <button
          class="clear-btn"
          on:click={() => { textInput = '' }}
          disabled={!textInput || submitting}
          aria-label="Clear text"
          title="Clear text"
        >✕ Clear</button>
      </div>
    {:else if inputMode === 'mic'}
      <div class="audio-area">
        <AudioRecorder
          showSendButton={false}
          on:recorded={handleRecorded}
          on:rerecord={handleRecorderRerecord}
        />
      </div>
    {:else if inputMode === 'file'}
      <div class="audio-area">
        <input
          type="file"
          accept="audio/*"
          bind:this={fileInput}
          on:change={handleFileChange}
          disabled={submitting}
          class="file-input"
        />
        {#if filePreviewUrl}
          <AudioPlayer src={filePreviewUrl} variant="user" />
        {/if}
      </div>
    {/if}

    <button
      class="analyse-btn"
      on:click={handleSubmit}
      disabled={submitting
        || (inputMode === 'text' && !textInput.trim())
        || ((inputMode === 'mic' || inputMode === 'file') && !audioBlob)}
    >
      {submitting ? 'Analysing…' : 'Analyse'}
    </button>

    <!-- Pipeline progress -->
    {#if submitting}
      <div class="progress" role="status" aria-live="polite">
        {#each progressStages as stage (stage.name)}
          <span class="stage" class:running={stage.status === 'running'} class:done={stage.status === 'complete'}>
            {#if stage.status === 'complete'}
              <span class="icon">✓</span>
            {:else}
              <span class="icon spinner" aria-hidden="true">◌</span>
            {/if}
            {stage.label}
          </span>
        {/each}
        {#if progressStages.length === 0}
          <span class="stage running">
            <span class="icon spinner" aria-hidden="true">◌</span>
            Starting…
          </span>
        {/if}
      </div>
    {/if}

    {#if submitError}
      <p class="error-msg">{submitError}</p>
    {/if}

    <!-- Results -->
    {#if hasResult}
      <hr class="divider" />

      {#if resultTranscript}
        <section class="result-section">
          <h3>Your text</h3>
          <p class="result-text">{resultTranscript}</p>
          <div class="section-footer">
            <CopyButton text={resultTranscript} label="Copy text" />
          </div>
        </section>
      {/if}

      {#if resultCorrections}
        {@const hasErrors = resultCorrections.errors.length > 0}
        <section class="result-section">
          <h3>Corrections</h3>
          {#if hasErrors}
            <ul class="errors-list">
              {#each resultCorrections.errors as err}
                <li>{err}</li>
              {/each}
            </ul>
          {:else}
            <p class="no-errors">No errors detected.</p>
          {/if}
          {#if resultCorrections.corrected}
            <div class="sub-section">
              <h4>Corrected</h4>
              <p class="result-text">{resultCorrections.corrected}</p>
              <div class="section-footer">
                <CopyButton text={resultCorrections.corrected} label="Copy corrected text" />
              </div>
            </div>
          {/if}
          {#if resultCorrections.native}
            <div class="sub-section">
              <h4>Natural phrasing</h4>
              <p class="result-text">{resultCorrections.native}</p>
              <div class="section-footer">
                <CopyButton text={resultCorrections.native} label="Copy natural phrasing" />
              </div>
            </div>
          {/if}
          {#if resultCorrections.explanation}
            <div class="sub-section">
              <h4>Explanation</h4>
              <p class="result-text">{resultCorrections.explanation}</p>
              <div class="section-footer">
                <CopyButton text={resultCorrections.explanation} label="Copy explanation" />
              </div>
            </div>
          {/if}
          <div class="section-footer section-footer--section">
            <CopyButton text={correctionsCopyText(resultCorrections)} label="Copy all corrections" />
          </div>
        </section>
      {/if}

      {#if resultSummary}
        <section class="result-section summary-section">
          <h3>Summary</h3>
          {#if resultSummary.improvement_areas.length > 0}
            <div class="sub-section">
              <h4>Areas to focus on</h4>
              <ol class="areas-list">
                {#each resultSummary.improvement_areas as area}
                  <li>{area}</li>
                {/each}
              </ol>
              <div class="section-footer">
                <CopyButton
                  text={resultSummary.improvement_areas.map((a, i) => `${i + 1}. ${a}`).join('\n')}
                  label="Copy areas to focus on"
                />
              </div>
            </div>
          {/if}
          {#if resultSummary.overall_assessment}
            <div class="sub-section">
              <h4>Overall</h4>
              <p class="result-text">{resultSummary.overall_assessment}</p>
              <div class="section-footer">
                <CopyButton text={resultSummary.overall_assessment} label="Copy overall assessment" />
              </div>
            </div>
          {/if}
          <div class="section-footer section-footer--section">
            <CopyButton text={summaryCopyText(resultSummary)} label="Copy full summary" />
          </div>
        </section>
      {:else if !submitting}
        <section class="result-section summary-section">
          <h3>Summary</h3>
          <p class="summary-unavailable">Summary is not saved — run a new analysis to see it.</p>
        </section>
      {/if}

    {/if}

  </div>
</div>

<style>
  .monologue-panel {
    flex: 1;
    display: flex;
    justify-content: center;
    overflow-y: auto;
    padding: 2rem 1rem;
  }

  .monologue-inner {
    width: 100%;
    max-width: 680px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .panel-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
    color: #222;
  }

  .lang-label {
    color: var(--kc-primary, #555);
  }

  /* ── Mode tabs ── */

  .mode-tabs {
    display: flex;
    gap: 6px;
  }

  .mode-tab {
    padding: 5px 14px;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    background: #f7f7f7;
    font-size: 0.83rem;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .mode-tab.active {
    background: var(--kc-primary, #555);
    color: #fff;
    border-color: var(--kc-primary, #555);
  }

  .mode-tab:disabled {
    opacity: 0.5;
    cursor: default;
  }

  /* ── Input ── */

  .textarea-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .text-input {
    width: 100%;
    resize: vertical;
    padding: 0.75rem;
    border-radius: 6px;
    border: 1px solid #d0d0d0;
    background: #fff;
    color: #222;
    font-size: 0.95rem;
    font-family: inherit;
    line-height: 1.5;
    box-sizing: border-box;
  }

  .text-input:focus {
    outline: none;
    border-color: var(--kc-primary, #555);
  }

  .text-input:disabled {
    background: #f8f8f8;
    opacity: 0.7;
  }

  .clear-btn {
    align-self: flex-end;
    background: none;
    border: none;
    cursor: pointer;
    color: #bbb;
    font-size: 0.75rem;
    padding: 2px 4px;
    border-radius: 3px;
    transition: color 0.15s, opacity 0.15s;
    line-height: 1;
  }

  .clear-btn:not(:disabled):hover {
    color: #888;
  }

  .clear-btn:disabled {
    opacity: 0.35;
    cursor: default;
  }

  .audio-area {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 0.5rem 0;
  }

  .file-input {
    font-size: 0.85rem;
  }

  /* ── Analyse button ── */

  .analyse-btn {
    align-self: flex-start;
    padding: 0.5rem 1.25rem;
    border: none;
    border-radius: 5px;
    background: var(--kc-primary, #555);
    color: #fff;
    font-size: 0.9rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .analyse-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* ── Progress ── */

  .progress {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    padding: 6px 0 2px;
    font-size: 0.78rem;
  }

  .stage {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #bbb;
    transition: color 0.2s;
  }

  .stage.running {
    color: var(--kc-primary, #555);
    font-weight: 600;
  }

  .stage.done {
    color: #aaa;
    text-decoration: line-through;
  }

  .icon {
    font-size: 0.85em;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .spinner {
    display: inline-block;
    animation: spin 1s linear infinite;
  }

  .error-msg {
    color: #e57373;
    font-size: 0.85rem;
    margin: 0;
  }

  /* ── Results ── */

  .divider {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 0;
  }

  .result-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .result-section h3 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #999;
    margin: 0;
  }

  .summary-section h3 {
    color: var(--kc-primary, #555);
  }

  .sub-section {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .sub-section h4 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #bbb;
    margin: 0;
  }

  .result-text {
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0;
    color: #333;
  }

  .errors-list,
  .areas-list {
    margin: 0;
    padding-left: 18px;
    font-size: 0.88rem;
    color: #444;
    line-height: 1.6;
  }

  .no-errors {
    font-size: 0.85rem;
    color: #aaa;
    margin: 0;
    font-style: italic;
  }

  .summary-unavailable {
    font-size: 0.85rem;
    color: #aaa;
    margin: 0;
    font-style: italic;
  }

  /* ── Copy buttons ── */

  .section-footer {
    display: flex;
    justify-content: flex-end;
    margin-top: 1px;
  }

  /* Bottom-of-section copy button gets a faint top border to separate it from sub-sections */
  .section-footer--section {
    margin-top: 4px;
    padding-top: 4px;
    border-top: 1px solid #f0f0f0;
  }
</style>
