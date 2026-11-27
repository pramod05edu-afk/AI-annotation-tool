import { Search } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { Link, NavLink, Route, Routes } from 'react-router'

import { api, unwrap } from './api/client'
import CommandPalette from './components/CommandPalette'
import { useAsync } from './hooks'
import ProjectPage from './pages/ProjectPage'
import ProjectsPage from './pages/ProjectsPage'

// The annotator includes the canvas library (Konva, ~300 kB): load it only when it is opened.
const AnnotatorPage = lazy(() => import('./pages/AnnotatorPage'))

/** Is the backend (database + storage) healthy, and are the AI models ready? */
function HealthStatus() {
  const loadHealth = useCallback(async () => unwrap(await api.GET('/api/health')), [])
  const loadAi = useCallback(async () => unwrap(await api.GET('/api/ai/status')), [])
  const health = useAsync(loadHealth)
  const ai = useAsync(loadAi)
  const ok = health.data?.status === 'ok'
  const aiReady = ai.data?.detector.state === 'ready'
  const label = health.error
    ? 'Backend offline'
    : !health.data
      ? 'Connecting…'
      : !ok
        ? 'Backend degraded'
        : aiReady
          ? 'Backend ok · AI ready'
          : 'Backend ok · AI loading'
  const title = health.data
    ? `Database ${health.data.database ? 'ok' : 'down'} · storage ${health.data.storage ? 'ok' : 'down'}`
    : (health.error ?? 'Checking backend…')
  return (
    <span
      className={`health ${health.error || (health.data && !ok) ? 'bad' : ok ? 'ok' : ''}`}
      title={title}
    >
      <span className="health-dot" />
      {label}
    </span>
  )
}

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false)

  // Ctrl+K (Cmd+K on a Mac) opens the command palette from anywhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((open) => !open)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark">◆</span>
          Annotate<span className="brand-suffix">.ai</span>
        </Link>
        <nav className="nav-links">
          <NavLink to="/" end>
            Projects
          </NavLink>
        </nav>
        <button type="button" className="search-pill" onClick={() => setPaletteOpen(true)}>
          <Search size={15} />
          Search projects and actions…
          <kbd>Ctrl K</kbd>
        </button>
        <HealthStatus />
      </header>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
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
