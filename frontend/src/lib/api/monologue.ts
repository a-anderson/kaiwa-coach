import type { MonologueSSEEvent } from '../types/api'
import { checkOk } from './client'

export async function* submitMonologueText(params: {
  conversation_id: string
  text: string
}): AsyncGenerator<MonologueSSEEvent> {
  const res = await checkOk(
    await fetch('/api/turns/monologue/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  )
  yield* parseSSEStream(res)
}

export async function* submitMonologueAudio(params: {
  conversation_id: string
  audio: Blob
}): AsyncGenerator<MonologueSSEEvent> {
  const form = new FormData()
  form.append('conversation_id', params.conversation_id)
  form.append('audio', params.audio, 'recording.webm')

  const res = await checkOk(await fetch('/api/turns/monologue/audio', { method: 'POST', body: form }))
  yield* parseSSEStream(res)
}

async function* parseSSEStream(res: Response): AsyncGenerator<MonologueSSEEvent> {
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    let eventType = ''
    let dataLine = ''
    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLine = line.slice(5).trim()
      } else if (line === '' && eventType && dataLine) {
        try {
          const data = JSON.parse(dataLine)
          yield { event: eventType, data } as MonologueSSEEvent
        } catch {
          if (import.meta.env.DEV) {
            console.warn('Failed to parse monologue SSE data:', dataLine)
          }
        }
        eventType = ''
        dataLine = ''
      }
    }
  }
}
