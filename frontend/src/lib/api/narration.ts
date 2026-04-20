async function checkOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  return res
}

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
