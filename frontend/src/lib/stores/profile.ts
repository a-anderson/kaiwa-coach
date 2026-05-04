import { writable } from 'svelte/store'
import { DEFAULT_TRANSLATION_LANGUAGE } from '../constants'

export interface ProfileState {
  translationLanguage: string
}

export const profileStore = writable<ProfileState>({
  translationLanguage: DEFAULT_TRANSLATION_LANGUAGE,
})
