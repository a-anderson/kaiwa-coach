import { writable } from 'svelte/store'

export interface UIState {
  sidebarOpen: boolean
  isSubmitting: boolean
}

export const uiStore = writable<UIState>({
  sidebarOpen: true,
  isSubmitting: false,
})
