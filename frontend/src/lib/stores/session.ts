import { writable } from 'svelte/store'
import type { TurnRecord } from '../types/api'

export interface SessionState {
  language: string
  conversationId: string | null
  turns: TurnRecord[]
}

export const sessionStore = writable<SessionState>({
  language: 'ja',
  conversationId: null,
  turns: [],
})
