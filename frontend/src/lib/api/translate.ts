import { checkOk } from './client'

export async function translateTurn(
  assistantTurnId: string,
  targetLanguage = 'English',
): Promise<string> {
  const res = await checkOk(
    await fetch(`/api/turns/${assistantTurnId}/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_language: targetLanguage }),
    }),
  )
  const json = await res.json()
  return json.translation as string
}
