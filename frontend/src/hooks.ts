import { useEffect, useState } from 'react'

import { errorMessage } from './api/client'

interface AsyncState<T> {
  load?: () => Promise<T> // which loader produced this result
  data?: T
  error?: string
}

/**
 * Run `load` when it changes and keep the result. Wrap `load` in useCallback with the values
 * it depends on (e.g. the project id); a new function means "load again".
 * `reload()` fetches again but keeps showing the old data until the new data arrives.
 */
export function useAsync<T>(load: () => Promise<T>) {
  const [state, setState] = useState<AsyncState<T>>({})
  const [version, setVersion] = useState(0)

  useEffect(() => {
    let cancelled = false // ignore answers that arrive after the user moved on
    load().then(
      (data) => !cancelled && setState({ load, data }),
      (error: unknown) => !cancelled && setState({ load, error: errorMessage(error) }),
    )
    return () => {
      cancelled = true
    }
  }, [load, version])

  const current = state.load === load
  return {
    data: current ? state.data : undefined,
    error: current ? state.error : undefined,
    loading: !current,
    reload: () => setVersion((v) => v + 1),
  }
}

/** Load an image file into an HTMLImageElement (Konva draws from these). */
export function useHtmlImage(url: string) {
  const [loaded, setLoaded] = useState<{ url: string; image?: HTMLImageElement; failed?: boolean }>(
    { url: '' },
  )
  useEffect(() => {
    const image = new window.Image()
    image.onload = () => setLoaded({ url, image })
    image.onerror = () => setLoaded({ url, failed: true })
    image.src = url
    return () => {
      image.onload = null
      image.onerror = null
    }
  }, [url])
  return loaded.url === url ? loaded : { url }
}

/** Track an element's size (the canvas fills whatever space the layout gives it). */
export function useElementSize<E extends HTMLElement>() {
  const [element, setElement] = useState<E | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    if (!element) return
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [element])
  return { ref: setElement, ...size }
}
