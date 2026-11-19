import { useEffect, useState } from 'react'

/** Shape of the backend's GET /api/health response. */
interface Health {
  status: 'ok' | 'degraded'
  database: boolean
  storage: boolean
}

type LoadState =
  { kind: 'loading' } | { kind: 'loaded'; health: Health } | { kind: 'error'; message: string }

function StatusRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <li className={ok ? 'ok' : 'bad'}>
      <span>{ok ? '✓' : '✗'}</span> {label}
    </li>
  )
}

export default function App() {
  const [state, setState] = useState<LoadState>({ kind: 'loading' })

  useEffect(() => {
    fetch('/api/health')
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        setState({ kind: 'loaded', health: (await response.json()) as Health })
      })
      .catch((error: unknown) => {
        setState({ kind: 'error', message: error instanceof Error ? error.message : 'unknown' })
      })
  }, [])

  return (
    <main>
      <h1>AI Annotation Tool</h1>
      <p className="subtitle">AI-assisted labeling for object detection datasets.</p>

      <section className="card">
        <h2>System status</h2>
        {state.kind === 'loading' && <p>Checking the backend…</p>}
        {state.kind === 'error' && <p className="bad">Backend not reachable ({state.message}).</p>}
        {state.kind === 'loaded' && (
          <ul className="status">
            <StatusRow label="API" ok />
            <StatusRow label="Database (PostgreSQL)" ok={state.health.database} />
            <StatusRow label="Image storage (S3 / SeaweedFS)" ok={state.health.storage} />
          </ul>
        )}
      </section>
    </main>
  )
}
