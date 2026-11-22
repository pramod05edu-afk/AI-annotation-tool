import { lazy, Suspense, useCallback } from 'react'
import { Link, Route, Routes } from 'react-router'

import { api, unwrap } from './api/client'
import { useAsync } from './hooks'
import ProjectPage from './pages/ProjectPage'
import ProjectsPage from './pages/ProjectsPage'

// The annotator includes the canvas library (Konva, ~300 kB): load it only when it is opened.
const AnnotatorPage = lazy(() => import('./pages/AnnotatorPage'))

/** Small dot in the header: is the backend (database + storage) healthy? */
function HealthDot() {
  const load = useCallback(async () => unwrap(await api.GET('/api/health')), [])
  const { data, error } = useAsync(load)
  const ok = data?.status === 'ok'
  const title = error
    ? `Backend not reachable (${error})`
    : data
      ? `Database ${data.database ? 'ok' : 'down'} · storage ${data.storage ? 'ok' : 'down'}`
      : 'Checking backend…'
  return (
    <span className={`health ${data ? (ok ? 'ok' : 'bad') : error ? 'bad' : ''}`} title={title}>
      ● {data ? (ok ? 'Backend ok' : 'Backend degraded') : error ? 'Backend offline' : '…'}
    </span>
  )
}

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          AI Annotation Tool
        </Link>
        <HealthDot />
      </header>
      <Routes>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<ProjectPage />} />
        <Route
          path="/projects/:projectId/annotate/:imageId"
          element={
            <Suspense fallback={<p className="muted page">Loading annotator…</p>}>
              <AnnotatorPage />
            </Suspense>
          }
        />
        <Route
          path="*"
          element={
            <main className="page">
              <h1>Page not found</h1>
              <Link to="/">Back to projects</Link>
            </main>
          }
        />
      </Routes>
    </div>
  )
}
