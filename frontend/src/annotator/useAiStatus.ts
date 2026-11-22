import { useEffect, useState } from 'react'

import { type AiStatus, api, unwrap } from '../api/client'

const POLL_MS = 3000

/** Which AI models are ready. Polls while the backend is still loading them. */
export function useAiStatus(): AiStatus | undefined {
  const [status, setStatus] = useState<AiStatus>()
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async () => {
      try {
        const next = unwrap(await api.GET('/api/ai/status'))
        if (cancelled) return
        setStatus(next)
        if (next.detector.state === 'loading' || next.segmenter.state === 'loading') {
          timer = setTimeout(() => void poll(), POLL_MS)
        }
      } catch {
        if (!cancelled) timer = setTimeout(() => void poll(), POLL_MS)
      }
    }
    void poll()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])
  return status
}
