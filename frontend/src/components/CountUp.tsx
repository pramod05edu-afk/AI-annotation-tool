import { useEffect, useState } from 'react'

const DURATION_MS = 900

// Accessibility: people who turned off animations in their OS see the final number at once.
const reduceMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** A number that counts up from 0 when it appears (ease-out), e.g. on the dashboard cards. */
export default function CountUp({ value, decimals = 0, suffix = '' }: CountUpProps) {
  const [shown, setShown] = useState(0)

  useEffect(() => {
    if (reduceMotion()) return
    let frame = 0
    const start = performance.now()
    const step = (now: number) => {
      const t = Math.min((now - start) / DURATION_MS, 1)
      setShown(value * (1 - (1 - t) ** 3)) // ease-out cubic
      if (t < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [value])

  return (
    <>
      {(reduceMotion() ? value : shown).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}
      {suffix}
    </>
  )
}

interface CountUpProps {
  value: number
  decimals?: number
  suffix?: string
}
