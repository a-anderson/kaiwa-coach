import { checkOk } from './client'

export async function generateNarration(text: string): Promise<{ audio_url: string }> {
  const res = await checkOk(
    await fetch('/api/narrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }),
  )
  return res.json()
}
