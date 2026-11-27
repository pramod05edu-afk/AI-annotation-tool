import { createContext, useContext } from 'react'

export type ToastKind = 'info' | 'success' | 'error' | 'ai'

export interface Toast {
  id: number
  kind: ToastKind
  title: string
  detail?: string
}

export type ShowToast = (kind: ToastKind, title: string, detail?: string) => void

export const ToastContext = createContext<ShowToast>(() => {})

/** Show a short notification in the corner: `toast('success', 'Saved', '3 images')`. */
export const useToast = () => useContext(ToastContext)
