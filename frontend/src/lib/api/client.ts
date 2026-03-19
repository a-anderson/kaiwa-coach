/**
 * Fetch-based SSE reader that supports POST requests.
 *
 * The browser's native EventSource only handles GET. For POST SSE (turn
 * submission) we need to read the response body as a stream and parse
 * SSE frames manually.
 */

export interface SSEFrame {
  event: string
  data: unknown
}

export async function* fetchSSE(
  url: string,
  options: RequestInit,
): AsyncGenerator<SSEFrame> {
  const res = await fetch(url, options)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
  }
  if (!res.body) throw new Error('No response body')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE frames are separated by a blank line.
      // sse-starlette emits CRLF (\r\n\r\n); handle both to be safe.
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        if (!frame.trim()) continue
        let event = 'message'
        let data = ''
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) data = line.slice(5).trim()
        }
        if (data) {
          try {
            yield { event, data: JSON.parse(data) }
          } catch {
            // Skip malformed frames rather than crashing the stream.
            if (import.meta.env.DEV) {
              console.warn('[SSE] Skipped malformed frame:', { event, data })
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
