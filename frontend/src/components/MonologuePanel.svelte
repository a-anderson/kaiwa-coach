<script lang="ts">
  import { onDestroy } from 'svelte'
  import { uiStore } from '../lib/stores/ui'
  import { sessionStore } from '../lib/stores/session'
  import { createMonologueConversation, getConversation } from '../lib/api/conversations'
  import { submitMonologueText, submitMonologueAudio } from '../lib/api/monologue'
  import AudioRecorder from './AudioRecorder.svelte'
  import type { MonologueCorrections, MonologueSummary } from '../lib/types/api'

  type InputMode = 'text' | 'mic' | 'file'

  let inputMode: InputMode = 'text'
  let textInput = ''
  let audioBlob: Blob | null = null
  let fileInput: HTMLInputElement

  let submitting = false
  let stageStatuses: Record<string, 'running' | 'complete'> = {}
  let submitError: string | null = null

  // Results from the most recently completed submission
  let resultTranscript: string | null = null
  let resultCorrections: MonologueCorrections | null = null
  let resultSummary: MonologueSummary | null = null
  let hasResult = false

  // Selected past session (read-only view)
  let selectedSessionId: string | null = null
  let loadingSession = false
  let sessionLoadError: string | null = null

  // Reset selected session when switching away from monologue tab
  $: if ($uiStore.activeTab !== 'monologue') {
    selectedSessionId = null
  }

  function setMode(mode: InputMode) {
    if (mode === inputMode) return
    inputMode = mode
    audioBlob = null
    textInput = ''
    submitError = null
  }

  function handleRecorded(e: CustomEvent<{ blob: Blob }>) {
    audioBlob = e.detail.blob
  }

  function handleFileChange() {
    const file = fileInput?.files?.[0]
    if (file) {
      audioBlob = file
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
          resultTranscript = resultTranscript ?? event.data.asr_text ?? null
          resultCorrections = event.data.corrections
          resultSummary = event.data.summary
          hasResult = true
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

  async function loadPastSession(id: string) {
    selectedSessionId = id
    loadingSession = true
    sessionLoadError = null
    hasResult = false
    resultTranscript = null
    resultCorrections = null
    resultSummary = null

    try {
      const convo = await getConversation(id)
      const turn = convo.turns[0]
      if (turn) {
        resultTranscript = turn.asr_text ?? null
        if (turn.correction) {
          resultCorrections = turn.correction
        }
        // Summary is not stored in TurnRecord; show corrections only for past sessions.
        hasResult = true
      }
    } catch (e) {
      sessionLoadError = e instanceof Error ? e.message : 'Failed to load session'
    } finally {
      loadingSession = false
    }
  }

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
  {#if selectedSessionId && !loadingSession && hasResult}
    <!-- Read-only past session view -->
    <div class="results-view">
      <button class="back-btn" on:click={() => { selectedSessionId = null; hasResult = false }}>
        ← Back
      </button>
      {#if resultTranscript}
        <section class="result-section">
          <h3>Transcript</h3>
          <p class="result-text">{resultTranscript}</p>
        </section>
      {/if}
      {#if resultCorrections}
        {#if resultCorrections.errors.length > 0}
          <section class="result-section">
            <h3>Errors</h3>
            <ul class="errors-list">
              {#each resultCorrections.errors as err}
                <li>{err}</li>
              {/each}
            </ul>
          </section>
        {/if}
        {#if resultCorrections.corrected}
          <section class="result-section">
            <h3>Corrected</h3>
            <p class="result-text">{resultCorrections.corrected}</p>
          </section>
        {/if}
        {#if resultCorrections.native}
          <section class="result-section">
            <h3>Native</h3>
            <p class="result-text">{resultCorrections.native}</p>
          </section>
        {/if}
        {#if resultCorrections.explanation}
          <section class="result-section">
            <h3>Explanation</h3>
            <p class="result-text">{resultCorrections.explanation}</p>
          </section>
        {/if}
      {/if}
    </div>
  {:else}
    <!-- Submission form -->
    <div class="form-area">
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
        <div class="mode-spacer"></div>
        <button
          class="analyse-btn"
          on:click={handleSubmit}
          disabled={submitting
            || (inputMode === 'text' && !textInput.trim())
            || ((inputMode === 'mic' || inputMode === 'file') && !audioBlob)}
        >
          {submitting ? 'Analysing…' : 'Analyse'}
        </button>
      </div>

      <div class="input-area">
        {#if inputMode === 'text'}
          <textarea
            class="text-input"
            bind:value={textInput}
            placeholder="Type or paste your {$sessionStore.language} text here…"
            disabled={submitting}
            rows="5"
          ></textarea>
        {:else if inputMode === 'mic'}
          <div class="recorder-wrapper">
            <AudioRecorder on:recorded={handleRecorded} />
            {#if audioBlob}
              <p class="audio-ready">Recording ready — click Analyse to submit.</p>
            {/if}
          </div>
        {:else if inputMode === 'file'}
          <div class="file-wrapper">
            <input
              type="file"
              accept="audio/*"
              bind:this={fileInput}
              on:change={handleFileChange}
              disabled={submitting}
              class="file-input"
            />
            {#if audioBlob}
              <p class="audio-ready">File selected — click Analyse to submit.</p>
            {/if}
          </div>
        {/if}
      </div>

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
    </div>

    <!-- Results -->
    {#if hasResult}
      <div class="results-area">
        {#if resultTranscript}
          <section class="result-section">
            <h3>Transcript</h3>
            <p class="result-text">{resultTranscript}</p>
          </section>
        {/if}

        {#if resultCorrections}
          <section class="result-section">
            <h3>Corrections</h3>
            {#if resultCorrections.errors.length > 0}
              <div class="corrections-block">
                <p class="label">Errors:</p>
                <ul class="errors-list">
                  {#each resultCorrections.errors as err}
                    <li>{err}</li>
                  {/each}
                </ul>
              </div>
            {:else}
              <p class="no-errors">No errors detected.</p>
            {/if}
            {#if resultCorrections.corrected}
              <div class="corrections-block">
                <p class="label">Corrected:</p>
                <p class="result-text">{resultCorrections.corrected}</p>
              </div>
            {/if}
            {#if resultCorrections.native}
              <div class="corrections-block">
                <p class="label">Native:</p>
                <p class="result-text">{resultCorrections.native}</p>
              </div>
            {/if}
            {#if resultCorrections.explanation}
              <div class="corrections-block">
                <p class="label">Explanation:</p>
                <p class="result-text">{resultCorrections.explanation}</p>
              </div>
            {/if}
          </section>
        {/if}

        {#if resultSummary}
          <section class="result-section summary-section">
            <h3>Summary</h3>
            {#if resultSummary.improvement_areas.length > 0}
              <div class="corrections-block">
                <p class="label">Areas to focus on:</p>
                <ol class="areas-list">
                  {#each resultSummary.improvement_areas as area, i}
                    <li>{area}</li>
                  {/each}
                </ol>
              </div>
            {/if}
            {#if resultSummary.overall_assessment}
              <div class="corrections-block">
                <p class="label">Overall:</p>
                <p class="result-text">{resultSummary.overall_assessment}</p>
              </div>
            {/if}
          </section>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<style>
  .monologue-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 20px 24px;
    gap: 16px;
  }

  .form-area {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 760px;
  }

  .mode-tabs {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .mode-tab {
    padding: 5px 14px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: #f7f7f7;
    font-size: 0.82rem;
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

  .mode-spacer {
    flex: 1;
  }

  .analyse-btn {
    padding: 6px 18px;
    background: var(--kc-primary, #555);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .analyse-btn:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .input-area {
    display: flex;
    flex-direction: column;
  }

  .text-input {
    width: 100%;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.9rem;
    font-family: inherit;
    resize: vertical;
    min-height: 100px;
    box-sizing: border-box;
  }

  .text-input:focus {
    outline: none;
    border-color: var(--kc-primary, #555);
  }

  .recorder-wrapper,
  .file-wrapper {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px 0;
  }

  .file-input {
    font-size: 0.85rem;
  }

  .audio-ready {
    font-size: 0.82rem;
    color: #666;
    margin: 0;
  }

  /* ── Progress ── */

  .progress {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    padding: 4px 0;
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
    font-size: 0.82rem;
    color: #c0392b;
    margin: 0;
  }

  /* ── Results ── */

  .results-area {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 760px;
    border-top: 1px solid #eee;
    padding-top: 16px;
  }

  .results-view {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 760px;
  }

  .back-btn {
    align-self: flex-start;
    background: none;
    border: none;
    color: var(--kc-primary, #555);
    font-size: 0.85rem;
    cursor: pointer;
    padding: 0;
    margin-bottom: 4px;
  }

  .back-btn:hover {
    text-decoration: underline;
  }

  .result-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .result-section h3 {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #888;
    margin: 0;
  }

  .summary-section h3 {
    color: var(--kc-primary, #555);
  }

  .corrections-block {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #999;
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
</style>
