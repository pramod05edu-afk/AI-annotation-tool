/**
 * Interaction log for the effort study (manual vs. AI-assisted annotation).
 *
 * Every relevant action (image opened, box drawn, suggestion deleted, image done, ...) is
 * recorded with a browser timestamp, the image, the mode and a session id. Events are
 * buffered and sent in batches, so logging never slows down annotating. Phase 7 computes
 * from these logs: time per image, number of interactions, and how much AI output was kept.
 */

export type Mode = 'manual' | 'assisted'

export type LogEventName =
  | 'image_opened'
  | 'image_closed'
  | 'image_done'
  | 'image_reopened'
  | 'annotation_created'
  | 'annotation_edited'
  | 'annotation_deleted'
  | 'class_changed'
  | 'undo'
  | 'suggest'
  | 'magic_click'
  | 'magic_accepted'
  | 'smart_click'
  | 'mode_changed'

interface LogEvent {
  image_id: number
  session_id: string
  mode: Mode
  event: LogEventName
  payload: Record<string, unknown>
  ts: string
}

const SESSION_KEY = 'annotator.sessionId'
const FLUSH_AFTER_MS = 2000
const FLUSH_AT_EVENTS = 50
const BATCH_SIZE = 100 // keeps each request small (keepalive requests are limited to 64 kB)

/** One id per browser tab: groups the events of one annotation session. */
function getSessionId(): string {
  try {
    let id = sessionStorage.getItem(SESSION_KEY)
    if (!id) {
      id = crypto.randomUUID()
      sessionStorage.setItem(SESSION_KEY, id)
    }
    return id
  } catch {
    return fallbackSessionId
  }
}
const fallbackSessionId = crypto.randomUUID()

const queue: LogEvent[] = []
let timer: ReturnType<typeof setTimeout> | undefined

export function logEvent(
  imageId: number,
  mode: Mode,
  event: LogEventName,
  payload: Record<string, unknown> = {},
) {
  queue.push({
    image_id: imageId,
    session_id: getSessionId(),
    mode,
    event,
    payload,
    ts: new Date().toISOString(),
  })
  if (queue.length >= FLUSH_AT_EVENTS) flush()
  else timer ??= setTimeout(flush, FLUSH_AFTER_MS)
}

/** Send everything buffered. `keepalive`: the request survives the page being closed. */
export function flush(keepalive = false) {
  clearTimeout(timer)
  timer = undefined
  while (queue.length > 0) {
    const events = queue.splice(0, BATCH_SIZE)
    fetch('/api/logs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events }),
      keepalive,
    }).catch(() => {
      // Logging must never disturb annotating: a lost batch is accepted (rare, local network).
    })
  }
}

// Tab hidden or closed: send what is left right away.
window.addEventListener('pagehide', () => flush(true))
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') flush(true)
})
