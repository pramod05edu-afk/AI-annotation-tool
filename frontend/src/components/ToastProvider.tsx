import { CircleCheck, Info, Sparkles, TriangleAlert, X } from 'lucide-react'
import { type ReactNode, useCallback, useRef, useState } from 'react'

import { type Toast, ToastContext, type ToastKind } from './toast'

const VISIBLE_MS = 4000
const ICONS = { info: Info, success: CircleCheck, error: TriangleAlert, ai: Sparkles }

/** Notifications in the bottom-right corner; each disappears after a few seconds. */
export default function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(1)

  const dismiss = useCallback(
    (id: number) => setToasts((all) => all.filter((t) => t.id !== id)),
    [],
  )

  const show = useCallback(
    (kind: ToastKind, title: string, detail?: string) => {
      const id = nextId.current++
      setToasts((all) => [...all.slice(-3), { id, kind, title, detail }]) // at most 4 at once
      setTimeout(() => dismiss(id), kind === 'error' ? VISIBLE_MS * 2 : VISIBLE_MS)
    },
    [dismiss],
  )

  return (
    <ToastContext.Provider value={show}>
      {children}
      <div className="toasts" aria-live="polite">
        {toasts.map((t) => {
          const Icon = ICONS[t.kind]
          return (
            <div key={t.id} className={`toast ${t.kind}`} role="status">
              <span className="toast-icon">
                <Icon size={16} />
              </span>
              <div className="toast-text">
                <b>{t.title}</b>
                {t.detail && <span>{t.detail}</span>}
              </div>
              <button
                type="button"
                className="icon"
                aria-label="Close"
                onClick={() => dismiss(t.id)}
              >
                <X size={14} />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}
