import type { ConversationSummary, ConversationDetail } from '../types/api'
import { checkOk } from './client'

export async function listConversations(
  type?: 'chat' | 'monologue',
): Promise<ConversationSummary[]> {
  const url = type ? `/api/conversations?type=${type}` : '/api/conversations'
  const res = await checkOk(await fetch(url))
  return res.json()
}

export async function createMonologueConversation(): Promise<{ conversation_id: string }> {
  const res = await checkOk(
    await fetch('/api/conversations/monologue', { method: 'POST' }),
  )
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

export async function setSessionLanguage(language: string): Promise<void> {
  await checkOk(
    await fetch('/api/session/language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language }),
    }),
  )
}
