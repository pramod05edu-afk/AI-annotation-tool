import { Lock, Plus, Upload } from 'lucide-react'
import { type FormEvent, type ReactNode, useCallback, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'

import { api, imageFileUrl, type Project, unwrap } from '../api/client'
import CountUp from '../components/CountUp'
import Modal from '../components/Modal'
import { classColor } from '../annotator/shapes'
import { useAsync } from '../hooks'

/** Parse "car, bicycle" (commas or new lines) into a clean, de-duplicated class list. */
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
        Labels
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
              {c}
              {i < 9 && <kbd>{i + 1}</kbd>}
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

function Stat(props: { label: string; value: ReactNode; note: string; ai?: boolean }) {
  return (
    <div className={`stat ${props.ai ? 'ai' : ''}`}>
      <span className="stat-label">{props.label}</span>
      <span className="stat-value">{props.value}</span>
      <span className="stat-note">{props.note}</span>
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
  const aiLabels = (bySource.ai ?? 0) + (bySource.ai_edited ?? 0)
  const aiShare = stats.labels_total ? (100 * aiLabels) / stats.labels_total : 0
  const detector = models?.find((m) => m.is_active)
  const map = detector?.metrics.map50_95 as number | undefined

  return (
    <div className="stats">
      <Stat
        label="Images labelled"
        value={
          <>
            <CountUp value={stats.images_done} />
            <sup>/ {stats.images_total.toLocaleString('en-US')}</sup>
          </>
        }
        note={`across ${stats.projects} project${stats.projects === 1 ? '' : 's'}`}
      />
      <Stat
        label="Labels"
        value={<CountUp value={stats.labels_total} />}
        note={`${bySource.manual ?? 0} drawn by hand`}
      />
      <Stat
        ai
        label="Proposed by the AI"
        value={
          <>
            <CountUp value={aiShare} />
            <sup>%</sup>
          </>
        }
        note={`${bySource.ai_edited ?? 0} corrected by a human`}
      />
      <Stat
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
  const shown = project.classes.slice(0, 4)
  return (
    <Link to={`/projects/${project.id}`} className="project-card">
      <div className={`mosaic n${previews.length}`}>
        {project.annotation_mode && (
          <span className={`mode-pill ${project.annotation_mode}`}>
            <Lock size={10} /> {project.annotation_mode}
          </span>
        )}
        {previews.map((id) => (
          <img key={id} src={imageFileUrl(id)} alt="" loading="lazy" />
        ))}
        {previews.length === 0 && <span className="mosaic-empty">no images yet</span>}
      </div>
      <div className="project-card-body">
        <span className="project-card-title">{project.name}</span>
        <p className="chips">
          {shown.map((c) => (
            <span key={c} className="chip">
              <i style={{ background: classColor(project.classes, c) }} />
              {c}
            </span>
          ))}
          {project.classes.length > shown.length && (
            <span className="chip">+{project.classes.length - shown.length}</span>
          )}
        </p>
        <div className="project-card-meta">
          <span className="progress thin">
            <span
              className="progress-done"
              style={{ width: `${total ? (100 * done) / total : 0}%` }}
            />
          </span>
          {done} / {total}
        </div>
      </div>
    </Link>
  )
}

type Filter = 'all' | 'assisted' | 'manual' | 'active' | 'done'

const FILTERS: { key: Filter; label: string; test: (p: Project) => boolean }[] = [
  { key: 'all', label: 'All', test: () => true },
  { key: 'assisted', label: 'Assisted', test: (p) => p.annotation_mode === 'assisted' },
  { key: 'manual', label: 'Manual', test: (p) => p.annotation_mode === 'manual' },
  {
    key: 'active',
    label: 'In progress',
    test: (p) => (p.status_counts?.done ?? 0) < p.image_count || p.image_count === 0,
  },
  {
    key: 'done',
    label: 'Done',
    test: (p) => p.image_count > 0 && (p.status_counts?.done ?? 0) === p.image_count,
  },
]

export default function ProjectsPage() {
  const load = useCallback(async () => unwrap(await api.GET('/api/projects')), [])
  const { data: projects, error } = useAsync(load)
  const [params, setParams] = useSearchParams()
  const creating = params.get('new') === '1'
  const setCreating = (open: boolean) => setParams(open ? { new: '1' } : {})
  const [filter, setFilter] = useState<Filter>('all')
  const active = FILTERS.find((f) => f.key === filter) ?? FILTERS[0]
  const visible = projects?.filter((p) => active?.test(p) ?? true)

  return (
    <main className="page wide">
      <section className="hero">
        <div className="intro">
          <div>
            <h1>
              Label anything.
              <br />
              <em>Let the model do the first pass.</em>
            </h1>
            <p>
              Bring any images and your own labels. A general detector proposes boxes for any label
              name, a fine-tuned one takes over where it knows better. You review, correct and
              export COCO or YOLO.
            </p>
          </div>
          <div className="actions">
            <button type="button" className="primary" onClick={() => setCreating(true)}>
              <Plus size={16} /> New project
            </button>
            {projects?.[0] && (
              <Link className="button" to={`/projects/${projects[0].id}`}>
                <Upload size={15} /> Continue labelling
              </Link>
            )}
          </div>
        </div>
        <Stats />
      </section>

      <div className="filters" role="group" aria-label="Filter projects">
        <span className="label">Projects</span>
        {FILTERS.map((f) => (
          <button
            key={f.key}
            type="button"
            className={`pill ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
            aria-pressed={filter === f.key}
          >
            {f.label} <i>{projects ? projects.filter(f.test).length : '–'}</i>
          </button>
        ))}
      </div>

      {error && <p className="error">Could not load projects: {error}</p>}
      <div className="project-grid">
        {!projects &&
          !error &&
          [0, 1, 2].map((i) => <div key={i} className="project-card skeleton" aria-hidden />)}
        {visible?.map((p) => (
          <ProjectCard key={p.id} project={p} />
        ))}
        {visible && visible.length === 0 && filter !== 'all' && (
          <p className="empty-state">No projects in this view.</p>
        )}
        {projects && filter === 'all' && (
          <button type="button" className="project-card new-card" onClick={() => setCreating(true)}>
            <span className="new-card-plus">
              <Plus size={20} />
            </span>
            <b>New project</b>
            <span>Any images, any labels. The AI looks for your labels by name.</span>
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
