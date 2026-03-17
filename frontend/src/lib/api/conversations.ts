import type { ConversationSummary, ConversationDetail } from '../types/api'

async function checkOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  return res
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await checkOk(await fetch('/api/conversations'))
  return res.json()
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const res = await checkOk(await fetch(`/api/conversations/${id}`))
  return res.json()
}

export async function createConversation(
  language?: string,
  title?: string,
): Promise<ConversationSummary> {
  const res = await checkOk(
    await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language, title }),
    }),
  )
  return res.json()
}

export async function deleteConversation(id: string): Promise<void> {
  await checkOk(await fetch(`/api/conversations/${id}`, { method: 'DELETE' }))
}

export async function deleteAllConversations(): Promise<void> {
  await checkOk(await fetch('/api/conversations', { method: 'DELETE' }))
}
