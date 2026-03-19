import { writable } from 'svelte/store'

export interface UIState {
  sidebarOpen: boolean
  isSubmitting: boolean
  // Maps stage name → current status; reset to {} at the start of each turn.
  stageStatuses: Record<string, 'running' | 'complete'>
  // Non-null while the shadowing panel is open.
  shadowingTurnId: string | null
  // Set to the assistant_turn_id of the most recently completed turn so its
  // AudioPlayer autoplays. Null for turns loaded from history.
  autoplayTurnId: string | null
}

export const uiStore = writable<UIState>({
  sidebarOpen: true,
  isSubmitting: false,
  stageStatuses: {},
  shadowingTurnId: null,
  autoplayTurnId: null,
})
