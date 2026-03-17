import { fetchSSE } from './client'
import type { SSEEvent } from '../types/api'

export interface TextTurnParams {
  conversationId: string
  text: string
  conversationHistory: string
  correctionsEnabled: boolean
  language: string
}

export async function* submitTextTurn(
  params: TextTurnParams,
): AsyncGenerator<SSEEvent> {
  const gen = fetchSSE('/api/turns/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_id: params.conversationId,
      text: params.text,
      conversation_history: params.conversationHistory,
      corrections_enabled: params.correctionsEnabled,
      language: params.language,
    }),
  })
  for await (const frame of gen) {
    // Trust our own backend to send well-typed events.
    yield frame as unknown as SSEEvent
  }
}

export interface AudioTurnParams {
  conversationId: string
  audioBlob: Blob
  conversationHistory: string
  correctionsEnabled: boolean
  language: string
}

export async function* submitAudioTurn(
  params: AudioTurnParams,
): AsyncGenerator<SSEEvent> {
  const form = new FormData()
  form.append('conversation_id', params.conversationId)
  form.append('conversation_history', params.conversationHistory)
  form.append('corrections_enabled', String(params.correctionsEnabled))
  form.append('language', params.language)
  form.append('audio', params.audioBlob, 'recording.webm')

  const gen = fetchSSE('/api/turns/audio', { method: 'POST', body: form })
  for await (const frame of gen) {
    yield frame as unknown as SSEEvent
  }
}
