import { type FormEvent, useCallback, useState } from 'react'
import { Link, useNavigate } from 'react-router'

import { api, unwrap } from '../api/client'
import ProgressBar from '../components/ProgressBar'
import { useAsync } from '../hooks'

/** Parse "helmet, head" (commas or new lines) into a clean, de-duplicated class list. */
const parseClasses = (text: string) => [
  ...new Set(
    text
      .split(/[,\n]/)
      .map((c) => c.trim())
      .filter(Boolean),
  ),
]

function CreateProjectForm() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [classes, setClasses] = useState('')
  const [error, setError] = useState<string>()
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(undefined)
    try {
      const project = unwrap(
        await api.POST('/api/projects', { body: { name, classes: parseClasses(classes) } }),
      )
      void navigate(`/projects/${project.id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <form className="card form" onSubmit={(e) => void submit(e)}>
      <h2>New project</h2>
      <label>
        Name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Construction site helmets"
          required
        />
      </label>
      <label>
        Classes <span className="muted">(comma-separated; order = keyboard shortcuts 1–9)</span>
        <input
          value={classes}
          onChange={(e) => setClasses(e.target.value)}
          placeholder="helmet, head"
          required
        />
      </label>
      {error && <p className="error">{error}</p>}
      <button
        type="submit"
        className="primary"
        disabled={busy || parseClasses(classes).length === 0}
      >
        {busy ? 'Creating…' : 'Create project'}
      </button>
    </form>
  )
}

export default function ProjectsPage() {
  const load = useCallback(async () => unwrap(await api.GET('/api/projects')), [])
  const { data: projects, error, loading } = useAsync(load)

  return (
    <main className="page">
      <h1>Projects</h1>
      <p className="muted">
        Upload images, label them with boxes or polygons — with AI suggestions — and export a COCO
        or YOLO dataset.
      </p>

      {error && <p className="error">Could not load projects: {error}</p>}
      {loading && <p className="muted">Loading…</p>}
      {projects && projects.length === 0 && <p>No projects yet — create the first one below.</p>}

      <div className="project-list">
        {projects?.map((p) => (
          <Link key={p.id} to={`/projects/${p.id}`} className="card project-card">
            <h2>{p.name}</h2>
            <p className="chips">
              {p.classes.map((c) => (
                <span key={c} className="chip">
                  {c}
                </span>
              ))}
            </p>
            <ProgressBar project={p} />
          </Link>
        ))}
      </div>

      <CreateProjectForm />
    </main>
  )
}
