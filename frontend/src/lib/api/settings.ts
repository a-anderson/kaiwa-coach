import { checkOk } from './client'

export interface UserProfile {
  user_name: string | null
  language_proficiency: Record<string, string>
  translation_language: string
}

export async function getProfile(): Promise<UserProfile> {
  const res = await checkOk(await fetch('/api/settings/profile'))
  return res.json()
}

export async function setProfile(profile: Partial<UserProfile>): Promise<void> {
  await checkOk(
    await fetch('/api/settings/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile),
    }),
  )
}
