import { writable } from 'svelte/store'

export interface SessionState {
  language: string
  conversationId: string | null
}

export const sessionStore = writable<SessionState>({
  language: 'ja',
  conversationId: null,
})
