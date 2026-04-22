/** TypeScript interfaces mirroring the KaiwaCoach API schemas. */

// ── Conversations ──────────────────────────────────────────────────────────

export interface ConversationSummary {
  id: string
  title: string | null
  language: string
  updated_at: string | null
  preview_text: string | null
  conversation_type?: string
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
  /** Set client-side only in the SSE complete handler; never returned by the API. True when the turn should autoplay on mount. */
  autoplay?: boolean
}

export interface ConversationDetail {
  id: string
  title: string | null
  language: string
  created_at: string | null
  updated_at: string | null
  turns: TurnRecord[]
  conversation_type?: string
}

// ── Monologue ──────────────────────────────────────────────────────────────

export interface MonologueCorrections {
  errors: string[]
  corrected: string
  native: string
  explanation: string
}

export interface MonologueSummary {
  improvement_areas: string[]
  overall_assessment: string
}

export interface MonologueSSECompleteEvent {
  conversation_id: string
  user_turn_id: string
  input_text: string
  asr_text: string | null
  corrections: MonologueCorrections
  summary: MonologueSummary
}

export type MonologueSSEEvent =
  | { event: 'stage'; data: { stage: string; status: 'running' | 'complete'; [key: string]: unknown } }
  | { event: 'complete'; data: MonologueSSECompleteEvent }
  | { event: 'error'; data: { message: string; request_id?: string } }

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
