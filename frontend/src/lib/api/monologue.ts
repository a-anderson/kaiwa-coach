import { fetchSSE } from './client'
import type { MonologueSSEEvent } from '../types/api'

export async function* submitMonologueText(params: {
  conversation_id: string
  text: string
}): AsyncGenerator<MonologueSSEEvent> {
  const gen = fetchSSE('/api/turns/monologue/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  for await (const frame of gen) {
    yield frame as unknown as MonologueSSEEvent
  }
}

export async function* submitMonologueAudio(params: {
  conversation_id: string
  audio: Blob
}): AsyncGenerator<MonologueSSEEvent> {
  const form = new FormData()
  form.append('conversation_id', params.conversation_id)
  form.append('audio', params.audio, 'recording.webm')

  const gen = fetchSSE('/api/turns/monologue/audio', { method: 'POST', body: form })
  for await (const frame of gen) {
    yield frame as unknown as MonologueSSEEvent
  }
}
