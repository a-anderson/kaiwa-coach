import { fetchSSE } from './client'

export async function regenTurnAudio(assistantTurnId: string): Promise<string | null> {
  const res = await fetch(`/api/turns/${assistantTurnId}/regen-audio`, { method: 'POST' })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  const json = await res.json()
  return json.audio_url ?? null
}

export interface RegenTurnEvent {
  event: 'turn_done'
  data: { assistant_turn_id: string; audio_url: string | null }
}

export interface RegenCompleteEvent {
  event: 'complete'
  data: Record<string, never>
}

export interface RegenErrorEvent {
  event: 'error'
  data: { message: string }
}

export type RegenEvent = RegenTurnEvent | RegenCompleteEvent | RegenErrorEvent

export async function* regenConversationAudio(
  conversationId: string,
): AsyncGenerator<RegenEvent> {
  const gen = fetchSSE(`/api/conversations/${conversationId}/regen-audio`, { method: 'POST' })
  for await (const frame of gen) {
    yield frame as unknown as RegenEvent
  }
}
