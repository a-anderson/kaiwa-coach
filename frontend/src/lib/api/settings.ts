export interface UserProfile {
  user_name: string | null
  language_proficiency: Record<string, string>
}

async function checkOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  return res
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
