import { Bot, Cpu, Images, Lock, Plus, Tags } from 'lucide-react'
import { type FormEvent, type ReactNode, useCallback, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'

import { api, imageFileUrl, type Project, unwrap } from '../api/client'
import CountUp from '../components/CountUp'
import Modal from '../components/Modal'
import ProgressRing from '../components/ProgressRing'
import { classColor } from '../annotator/shapes'
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
    <form className="form" onSubmit={(e) => void submit(e)}>
      <label>
        Name
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Street scenes, Product shelf, Pets"
          required
        />
      </label>
      <label>
        Classes
        <input
          value={classes}
          onChange={(e) => setClasses(e.target.value)}
          placeholder="e.g. car, bicycle, pedestrian"
          required
        />
        <span className="muted small">
          Your own labels, comma-separated; their order sets the keyboard shortcuts 1–9. The AI
          looks for them by name, so plain words work best (&quot;traffic light&quot;, not
          &quot;TL_01&quot;).
        </span>
      </label>
      {parseClasses(classes).length > 0 && (
        <p className="chips">
          {parseClasses(classes).map((c, i, all) => (
            <span key={c} className="chip">
              <i style={{ background: classColor(all, c) }} />
              {i < 9 && <kbd>{i + 1}</kbd>} {c}
            </span>
          ))}
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <button
        type="submit"
        className="primary"
        disabled={busy || !name.trim() || parseClasses(classes).length === 0}
      >
        {busy ? 'Creating…' : 'Create project'}
      </button>
    </form>
  )
}

function StatCard(props: { icon: ReactNode; label: string; value: ReactNode; note: string }) {
  return (
    <div className="stat">
      <div className="stat-label">
        {props.icon}
        {props.label}
      </div>
      <div className="stat-value">{props.value}</div>
      <div className="stat-note">{props.note}</div>
    </div>
  )
}

function Stats() {
  const loadStats = useCallback(async () => unwrap(await api.GET('/api/stats')), [])
  const loadModels = useCallback(
    async () => unwrap(await api.GET('/api/models', { params: { query: { task: 'detect' } } })),
    [],
  )
  const { data: stats } = useAsync(loadStats)
  const { data: models } = useAsync(loadModels)
  if (!stats) return <div className="stats skeleton-row" aria-hidden />

  const bySource = stats.labels_by_source
  const aiLabels = (bySource.ai ?? 0) + (bySource.ai_edited ?? 0) + (bySource.ai_confirmed ?? 0)
  const aiShare = stats.labels_total ? (100 * aiLabels) / stats.labels_total : 0
  const detector = models?.find((m) => m.is_active)
  const map = detector?.metrics.map50_95 as number | undefined

  return (
    <div className="stats">
      <StatCard
        icon={<Images size={15} />}
        label="Images labelled"
        value={<CountUp value={stats.images_done} />}
        note={`of ${stats.images_total.toLocaleString('en-US')} in ${stats.projects} projects`}
      />
      <StatCard
        icon={<Tags size={15} />}
        label="Labels"
        value={<CountUp value={stats.labels_total} />}
        note={`${bySource.manual ?? 0} drawn by hand`}
      />
      <StatCard
        icon={<Bot size={15} />}
        label="Proposed by the AI"
        value={<CountUp value={aiShare} suffix=" %" />}
        note={`${bySource.ai_edited ?? 0} corrected by a human`}
      />
      <StatCard
        icon={<Cpu size={15} />}
        label="Fine-tuned detector"
        value={map != null ? <CountUp value={map} decimals={3} /> : '–'}
        note={detector ? `${detector.name} · test mAP50-95` : 'no detector loaded'}
      />
    </div>
  )
}

function ProjectCard({ project }: { project: Project }) {
  const done = project.status_counts?.done ?? 0
  const total = project.image_count
  const previews = project.preview_image_ids ?? []
  return (
    <Link to={`/projects/${project.id}`} className="project-card">
      <div className={`mosaic n${previews.length}`}>
        {previews.map((id) => (
          <img key={id} src={imageFileUrl(id)} alt="" loading="lazy" />
        ))}
        {previews.length === 0 && <span className="mosaic-empty">No images yet</span>}
      </div>
      <div className="project-card-body">
        <div className="project-card-title">
          <b>{project.name}</b>
          {project.annotation_mode && (
            <span className={`mode-pill ${project.annotation_mode}`}>
              <Lock size={11} /> {project.annotation_mode === 'manual' ? 'Manual' : 'Assisted'}
            </span>
          )}
        </div>
        <p className="chips">
          {project.classes.map((c) => (
            <span key={c} className="chip">
              <i style={{ background: classColor(project.classes, c) }} />
              {c}
            </span>
          ))}
        </p>
        <div className="project-card-meta">
          <span>
            {done} / {total} images done
          </span>
          <ProgressRing value={total ? done / total : 0} />
        </div>
      </div>
    </Link>
  )
}

export default function ProjectsPage() {
  const load = useCallback(async () => unwrap(await api.GET('/api/projects')), [])
  const { data: projects, error } = useAsync(load)
  const [params, setParams] = useSearchParams()
  const creating = params.get('new') === '1'
  const setCreating = (open: boolean) => setParams(open ? { new: '1' } : {})

  return (
    <main className="page wide">
      <div className="page-title">
        <div>
          <h1>Projects</h1>
          <p className="muted">
            Label any images with your own labels while the AI does most of the work. Review,
            correct, export.
          </p>
        </div>
        <button type="button" className="primary" onClick={() => setCreating(true)}>
          <Plus size={16} /> New project
        </button>
      </div>

      <Stats />

      <h2 className="section-title">Your projects</h2>
      {error && <p className="error">Could not load projects: {error}</p>}
      <div className="project-grid">
        {!projects &&
          !error &&
          [0, 1, 2].map((i) => <div key={i} className="project-card skeleton" aria-hidden />)}
        {projects?.map((p) => (
          <ProjectCard key={p.id} project={p} />
        ))}
        {projects && (
          <button type="button" className="project-card new-card" onClick={() => setCreating(true)}>
            <span className="new-card-plus">
              <Plus size={22} />
            </span>
            <b>New project</b>
            <span className="muted small">Define classes, then drop images</span>
          </button>
        )}
      </div>

      {creating && (
        <Modal title="New project" onClose={() => setCreating(false)}>
          <CreateProjectForm />
        </Modal>
      )}
    </main>
  )
}
