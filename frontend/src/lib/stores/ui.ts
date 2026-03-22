import { writable } from 'svelte/store'

export interface UIState {
  sidebarOpen: boolean
  isSubmitting: boolean
  // Maps stage name → current status; reset to {} at the start of each turn.
  stageStatuses: Record<string, 'running' | 'complete'>
  // Non-null while the shadowing panel is open.
  shadowingTurnId: string | null
  // True while the pending turn's TTS audio is ready to play but the turn has
  // not yet committed. Cleared on SSE complete or on any error/finally path.
  autoplayPending: boolean
}

export const uiStore = writable<UIState>({
  sidebarOpen: true,
  isSubmitting: false,
  stageStatuses: {},
  shadowingTurnId: null,
  autoplayPending: false,
})
