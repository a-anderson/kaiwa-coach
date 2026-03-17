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
