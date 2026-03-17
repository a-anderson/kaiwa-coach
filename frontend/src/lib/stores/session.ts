import { writable } from 'svelte/store'
import type { TurnRecord } from '../types/api'

export interface SessionState {
  language: string
  conversationId: string | null
  turns: TurnRecord[]
  // Live in-progress turn — populated on submit, cleared when complete fires.
  pendingTurn: TurnRecord | null
}

export const sessionStore = writable<SessionState>({
  language: 'ja',
  conversationId: null,
  turns: [],
  pendingTurn: null,
})
