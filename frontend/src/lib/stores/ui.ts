import { writable } from 'svelte/store'

export type Tab = 'chat' | 'monologue' | 'narration'

export interface UIState {
  sidebarOpen: boolean
  isSubmitting: boolean
  // Maps stage name → current status; reset to {} at the start of each turn.
  stageStatuses: Record<string, 'running' | 'complete'>
  // Non-null while the shadowing panel is open.
  shadowingTurnId: string | null
  activeTab: Tab
  settingsOpen: boolean
}

export const uiStore = writable<UIState>({
  sidebarOpen: true,
  isSubmitting: false,
  stageStatuses: {},
  shadowingTurnId: null,
  activeTab: 'chat',
  settingsOpen: false,
})
