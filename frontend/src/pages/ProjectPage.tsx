import { ChevronLeft, Download, Pencil, Play, Trash2, Upload } from 'lucide-react'
import { type DragEvent, useCallback, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import {
  api,
  type ImageInfo,
  type ImageStatus,
  imageFileUrl,
  type UploadResult,
  unwrap,
  uploadImages,
} from '../api/client'
import type { Mode } from '../annotator/interactionLog'
import { classColor } from '../annotator/shapes'
import EditLabels from '../components/EditLabels'
import Modal from '../components/Modal'
import ModeSwitch from '../components/ModeSwitch'
import ProgressBar from '../components/ProgressBar'
import { useToast } from '../components/toast'
import { useAsync, usePersistentState } from '../hooks'

const STATUS_LABEL: Record<ImageStatus, string> = {
  todo: 'To do',
  in_progress: 'In progress',
  done: 'Done',
}
// TIFF and BMP are converted to PNG by the server (browsers cannot display them).
const ACCEPTED = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/tiff',
  'image/bmp',
  '.tif',
  '.tiff',
]

function UploadZone({ projectId, onUploaded }: { projectId: number; onUploaded: () => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number }>()
  const [result, setResult] = useState<UploadResult>()
  const [error, setError] = useState<string>()
  const toast = useToast()

  async function upload(fileList: FileList | null) {
    const files = [...(fileList ?? [])]
    if (files.length === 0) return
    setError(undefined)
    setResult(undefined)
    setProgress({ done: 0, total: files.length })
    try {
      const outcome = await uploadImages(projectId, files, (done, total) =>
        setProgress({ done, total }),
      )
      setResult(outcome)
      toast(
        outcome.rejected.length ? 'info' : 'success',
        `${outcome.uploaded.length} image${outcome.uploaded.length === 1 ? '' : 's'} uploaded`,
        outcome.rejected.length ? `${outcome.rejected.length} rejected (see below)` : undefined,
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setProgress(undefined)
      onUploaded() // refresh the grid, also after a partial failure
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault()
    setDragOver(false)
    void upload(event.dataTransfer.files)
  }

  return (
    <section
      className={`card dropzone ${dragOver ? 'over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPTED.join(',')}
        hidden
        onChange={(e) => {
          void upload(e.target.files)
          e.target.value = '' // allow choosing the same files again
        }}
      />
      {progress ? (
        <p>
          Uploading {progress.done} / {progress.total}…
          <progress value={progress.done} max={progress.total} />
        </p>
      ) : (
        <p className="dropzone-text">
          <span className="dropzone-icon">
            <Upload size={20} />
          </span>
          Drop images here or{' '}
          <button type="button" className="link" onClick={() => input.current?.click()}>
            choose files
          </button>{' '}
          <span className="muted">(PNG, JPEG, WEBP, TIFF, BMP)</span>
        </p>
      )}
      {result && (
        <p className="small">
          ✓ {result.uploaded.length} uploaded
          {result.rejected.length > 0 && (
            <span className="error">
              {' '}
              · {result.rejected.length} rejected:{' '}
              {result.rejected.map((r) => `${r.filename} (${r.reason})`).join(', ')}
            </span>
          )}
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  )
}

/** Download the dataset: the browser saves the zip the API streams back. */
function ExportPanel({ projectId, doneCount }: { projectId: number; doneCount: number }) {
  const [format, setFormat] = useState<'coco' | 'yolo'>('coco')
  const [scope, setScope] = useState<'all' | 'done'>('done')
  const nothingDone = scope === 'done' && doneCount === 0
  return (
    <section className="card export">
      <h2>Export dataset</h2>
      <label>
        Format{' '}
        <select value={format} onChange={(e) => setFormat(e.target.value as 'coco' | 'yolo')}>
          <option value="coco">COCO (JSON, boxes + polygons)</option>
          <option value="yolo">YOLO (txt labels + data.yaml)</option>
        </select>
      </label>
      <label>
        Images{' '}
        <select value={scope} onChange={(e) => setScope(e.target.value as 'all' | 'done')}>
          <option value="done">only done ({doneCount})</option>
          <option value="all">all</option>
        </select>
      </label>
      {nothingDone ? (
        <span className="muted small">Mark images as done first.</span>
      ) : (
        <a
          className="button primary"
          href={`/api/projects/${projectId}/export?format=${format}&scope=${scope}`}
          download
        >
          <Download size={16} /> Download zip
        </a>
      )}
    </section>
  )
}

function Thumbnail({ projectId, image }: { projectId: number; image: ImageInfo }) {
  return (
    <Link
      to={`/projects/${projectId}/annotate/${image.id}`}
      className="thumb"
      title={image.filename}
    >
      <img src={imageFileUrl(image.id)} alt={image.filename} loading="lazy" />
      <span className={`badge ${image.status}`}>{STATUS_LABEL[image.status]}</span>
      {image.annotation_count > 0 && <span className="count">{image.annotation_count}</span>}
    </Link>
  )
}

export default function ProjectPage() {
  const projectId = Number(useParams().projectId)
  const navigate = useNavigate()
  const [filter, setFilter] = useState<ImageStatus | 'all'>('all')
  const [editingLabels, setEditingLabels] = useState(false)
  const toast = useToast()
  // Same remembered setting as in the annotator: choose the mode BEFORE opening an image.
  const [mode, setMode] = usePersistentState<Mode>('annotator.mode', 'assisted')

  const loadProject = useCallback(
    async () =>
      unwrap(
        await api.GET('/api/projects/{project_id}', {
          params: { path: { project_id: projectId } },
        }),
      ),
    [projectId],
  )
  const loadImages = useCallback(
    async () =>
      unwrap(
        await api.GET('/api/projects/{project_id}/images', {
          params: { path: { project_id: projectId }, query: { limit: 1000 } },
        }),
      ),
    [projectId],
  )
  const project = useAsync(loadProject)
  const images = useAsync(loadImages)
  const refresh = () => {
    project.reload()
    images.reload()
  }

  async function deleteProject() {
    if (!project.data) return
    const ok = window.confirm(
      `Delete "${project.data.name}" with all ${project.data.image_count} images and annotations? This cannot be undone.`,
    )
    if (!ok) return
    unwrap(
      await api.DELETE('/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
      }),
    )
    void navigate('/')
  }

  if (project.error) {
    return (
      <main className="page">
        <p className="error">{project.error}</p>
        <Link to="/">Back to projects</Link>
      </main>
    )
  }
  if (!project.data) return <main className="page muted">Loading…</main>

  const all = images.data ?? []
  const shown = filter === 'all' ? all : all.filter((i) => i.status === filter)
  const firstOpen = all.find((i) => i.status !== 'done') ?? all[0]

  return (
    <main className="page wide">
      <Link to="/" className="back-link">
        <ChevronLeft size={16} /> Projects
      </Link>
      <div className="page-header">
        <div>
          <h1>{project.data.name}</h1>
          <p className="chips">
            {project.data.classes.map((c, i) => (
              <span key={c} className="chip">
                <i style={{ background: classColor(project.data?.classes ?? [], c) }} />
                <kbd>{i + 1}</kbd> {c}
              </span>
            ))}
            <button type="button" className="chip-button" onClick={() => setEditingLabels(true)}>
              <Pencil size={12} /> Edit labels
            </button>
          </p>
        </div>
        <div className="actions">
          {firstOpen && (
            <Link className="button primary" to={`/projects/${projectId}/annotate/${firstOpen.id}`}>
              <Play size={15} />
              {firstOpen.status === 'done' ? 'Review annotations' : 'Start annotating'}
            </Link>
          )}
          <button type="button" className="danger" onClick={() => void deleteProject()}>
            <Trash2 size={15} /> Delete
          </button>
        </div>
      </div>

      <div className="mode-row">
        <span className="muted small">Annotation mode:</span>
        <ModeSwitch
          mode={project.data.annotation_mode ?? mode}
          locked={project.data.annotation_mode != null}
          onChange={setMode}
        />
        <span className="muted small">
          {project.data.annotation_mode != null
            ? 'fixed for this project (e.g. for a controlled study)'
            : 'choose before you start annotating'}
        </span>
      </div>
      <ProgressBar project={project.data} />
      <UploadZone projectId={projectId} onUploaded={refresh} />
      <ExportPanel projectId={projectId} doneCount={project.data.status_counts?.done ?? 0} />

      <div className="tabs" role="tablist">
        {(['all', 'todo', 'in_progress', 'done'] as const).map((s) => (
          <button
            key={s}
            type="button"
            role="tab"
            aria-selected={filter === s}
            className={`pill ${filter === s ? 'active' : ''}`}
            onClick={() => setFilter(s)}
          >
            {s === 'all' ? 'All' : STATUS_LABEL[s]}{' '}
            <i>{s === 'all' ? all.length : (project.data?.status_counts?.[s] ?? 0)}</i>
          </button>
        ))}
      </div>

      {images.error && <p className="error">{images.error}</p>}
      <div className="grid">
        {shown.map((image) => (
          <Thumbnail key={image.id} projectId={projectId} image={image} />
        ))}
      </div>
      {images.data && shown.length === 0 && <p className="muted">No images here.</p>}

      {editingLabels && (
        <Modal title="Edit labels" onClose={() => setEditingLabels(false)}>
          <EditLabels
            project={project.data}
            onSaved={(saved) => {
              setEditingLabels(false)
              project.reload()
              toast('success', 'Labels saved', saved.classes.join(', '))
            }}
          />
        </Modal>
      )}
    </main>
  )
}
