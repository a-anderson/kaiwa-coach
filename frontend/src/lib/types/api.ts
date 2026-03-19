/** TypeScript interfaces mirroring the KaiwaCoach API schemas. */

// ── Conversations ──────────────────────────────────────────────────────────

export interface ConversationSummary {
  id: string
  title: string | null
  language: string
  updated_at: string | null
  preview_text: string | null
}

export interface CorrectionData {
  errors: string[]
  corrected: string
  native: string
  explanation: string
}

export interface TurnRecord {
  user_turn_id: string
  assistant_turn_id: string | null
  user_text: string | null
  asr_text: string | null
  reply_text: string | null
  correction: CorrectionData | null
  has_user_audio: boolean
  has_assistant_audio: boolean
  user_audio_url: string | null
  assistant_audio_url: string | null
  /** Set to 'pending' for in-flight turns only; absent on committed turns. */
  status?: 'pending'
}

export interface ConversationDetail {
  id: string
  title: string | null
  language: string
  created_at: string | null
  updated_at: string | null
  turns: TurnRecord[]
}

// ── Turn requests ──────────────────────────────────────────────────────────

export interface TurnTextRequest {
  conversation_id?: string | null
  language?: string
  text: string
  conversation_history?: string
  corrections_enabled?: boolean
}

// ── SSE events ─────────────────────────────────────────────────────────────

export interface SSEStageEvent {
  stage: string
  status: 'running' | 'complete'
  // stage-specific optional fields
  reply?: string
  transcript?: string
  audio_url?: string | null
  data?: CorrectionData
}

export interface SSECompleteEvent {
  conversation_id: string
  user_turn_id: string
  assistant_turn_id: string
  reply_text: string
  audio_url: string | null
  asr_text?: string  // audio turns only
}

export interface SSEErrorEvent {
  message: string
  request_id?: string
}

export type SSEEvent =
  | { event: 'stage'; data: SSEStageEvent }
  | { event: 'complete'; data: SSECompleteEvent }
  | { event: 'error'; data: SSEErrorEvent }

// ── Settings ───────────────────────────────────────────────────────────────

export interface SettingsResponse {
  language: string
}
