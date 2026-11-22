/**
 * Annotate one image: draw / edit / delete shapes, switch images, autosave, mark done.
 *
 * Structure (each level only renders once its data is loaded):
 *   AnnotatorPage   project + image list, tool, class, mode, AI status (kept across images)
 *   └─ ImageLoader  this image's file, status and saved annotations   (remounted per image)
 *      └─ Editor    the editable shapes, undo history, saving, keyboard shortcuts
 */
import { useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import {
  type AiStatus,
  api,
  errorMessage,
  type ImageInfo,
  type ImageStatus,
  imageFileUrl,
  type Project,
  unwrap,
} from '../api/client'
import AnnotationCanvas from '../annotator/AnnotationCanvas'
import {
  classColor,
  fromApi,
  type MagicPoint,
  markEdited,
  newBox,
  newPolygon,
  type Point,
  type Shape,
  toApi,
  type Tool,
  type View,
} from '../annotator/shapes'
import { useAiStatus } from '../annotator/useAiStatus'
import { useAsync, useHtmlImage, usePersistentState } from '../hooks'

const AUTOSAVE_DELAY_MS = 800
const MAX_UNDO = 100

export type Mode = 'manual' | 'assisted'

const TOOLS: { tool: Tool; key: string; label: string; assistedOnly?: boolean }[] = [
  { tool: 'select', key: 'V', label: 'Select' },
  { tool: 'box', key: 'B', label: 'Box' },
  { tool: 'polygon', key: 'P', label: 'Polygon' },
  { tool: 'magic', key: 'M', label: '🪄 Magic', assistedOnly: true },
]

const SHORTCUTS: [string, string][] = [
  ['B / P / V', 'box / polygon / select tool'],
  ['1 – 9', 'choose class (also changes the selected shape)'],
  ['Click first point / Enter', 'close polygon'],
  ['Del', 'delete selected shape'],
  ['Ctrl + Z', 'undo'],
  ['Esc', 'cancel drawing / deselect'],
  ['Wheel · Space + drag', 'zoom · pan (F = fit)'],
  ['A / D', 'previous / next image'],
  ['Enter', 'mark done and go to next image'],
]

const AI_SHORTCUTS: [string, string][] = [
  ['S', 'suggest boxes (AI)'],
  ['M', 'magic tool: click = object, Shift / right click = not object'],
  ['Enter', 'accept the magic outline'],
]

/** Plain numbers from the API -> a fixed 4-tuple box. */
const toBox = ([x1 = 0, y1 = 0, x2 = 0, y2 = 0]: number[]): [number, number, number, number] => [
  x1,
  y1,
  x2,
  y2,
]

/** Why an AI button is disabled, for its tooltip. */
function modelHint(ai: AiStatus | undefined, which: 'detector' | 'segmenter') {
  const slot = ai?.[which]
  if (!slot) return 'Checking the AI models…'
  if (slot.state === 'loading') return `The ${which} is loading…`
  if (slot.state === 'failed')
    return `The ${which} is unavailable: ${slot.error ?? 'unknown error'}`
  return `The ${which} is ${slot.state}`
}

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)

interface EditorProps {
  project: Project
  image: ImageInfo
  htmlImage: HTMLImageElement
  initialShapes: Shape[]
  position: { index: number; total: number; prevId?: number; nextId?: number }
  tool: Tool
  setTool: (tool: Tool) => void
  activeClass: string
  setActiveClass: (name: string) => void
  mode: Mode
  setMode: (mode: Mode) => void
  confidence: number
  setConfidence: (value: number) => void
  ai: AiStatus | undefined
}

function Editor({ project, image, htmlImage, initialShapes, position, ...p }: EditorProps) {
  const navigate = useNavigate()
  const [shapes, setShapes] = useState(initialShapes)
  const [history, setHistory] = useState<Shape[][]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Point[]>([])
  const [view, setView] = useState<View | null>(null)
  const [panning, setPanning] = useState(false)
  const [status, setStatus] = useState<ImageStatus>(image.status)

  // AI assistance
  const assisted = p.mode === 'assisted'
  const detectorReady = p.ai?.detector.state === 'ready'
  const segmenterReady = p.ai?.segmenter.state === 'ready'
  const [aiBusy, setAiBusy] = useState(false)
  const [aiMessage, setAiMessage] = useState<string>()
  const [magicPoints, setMagicPoints] = useState<MagicPoint[]>([])
  const [magicPreview, setMagicPreview] = useState<{
    points: Point[]
    score: number
    modelVersionId: number | null
  } | null>(null)
  const magicRequest = useRef(0) // only the newest magic request may update the preview

  // Saving: every change bumps `version`; a save records which version it stored.
  const [version, setVersion] = useState(0)
  const [savedVersion, setSavedVersion] = useState(0)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string>()
  const latest = useRef({ shapes, version, savedVersion })
  const saveQueue = useRef<Promise<void>>(Promise.resolve())
  useEffect(() => {
    latest.current = { shapes, version, savedVersion }
  })

  const annotationsPath = { params: { path: { image_id: image.id } } }
  const classes = project.classes

  // ---------- editing (every change goes through these) ----------

  const bump = () => setVersion((v) => v + 1)
  // `latest.current.shapes` (not `shapes`): also correct when called after an `await`.
  const remember = () => setHistory((h) => [...h.slice(-(MAX_UNDO - 1)), latest.current.shapes])

  function commit(next: Shape[]) {
    remember()
    setShapes(next)
    latest.current = { ...latest.current, shapes: next }
    bump()
  }

  function create(shape: Shape) {
    commit([...latest.current.shapes, shape])
    setSelectedId(shape.id)
  }

  function edit(shape: Shape) {
    setShapes((all) =>
      all.map((s) => (s.id === shape.id ? { ...shape, source: markEdited(shape.source) } : s)),
    )
    bump()
  }

  function remove(id: string) {
    commit(shapes.filter((s) => s.id !== id))
    if (selectedId === id) setSelectedId(null)
  }

  function changeClass(id: string, className: string) {
    commit(shapes.map((s) => (s.id === id ? { ...s, className, source: markEdited(s.source) } : s)))
  }

  function undo() {
    const previous = history.at(-1)
    if (!previous) return
    setHistory(history.slice(0, -1))
    setShapes(previous)
    setSelectedId(null)
    bump()
  }

  function chooseTool(tool: Tool) {
    p.setTool(tool)
    setDraft([])
    clearMagic()
    if (tool !== 'select') setSelectedId(null)
  }

  function chooseMode(mode: Mode) {
    p.setMode(mode)
    setAiMessage(undefined)
    if (mode === 'manual' && p.tool === 'magic') chooseTool('box')
  }

  function chooseClass(className: string) {
    p.setActiveClass(className)
    if (selectedId) changeClass(selectedId, className)
  }

  function closePolygon() {
    if (draft.length < 3) return false
    create(newPolygon(draft, p.activeClass))
    setDraft([])
    return true
  }

  // ---------- AI assistance ----------

  /** Ask the detector for boxes; they replace earlier, untouched AI boxes. */
  async function suggest() {
    if (!assisted || !detectorReady || aiBusy) return
    setAiBusy(true)
    setAiMessage('Detecting…')
    try {
      const result = unwrap(
        await api.POST('/api/images/{image_id}/suggest', {
          ...annotationsPath,
          body: { conf: p.confidence },
        }),
      )
      const suggestions: Shape[] = result.suggestions.map((s) => ({
        ...newBox(toBox(s.bbox), s.class_name),
        source: 'ai',
        score: s.score,
        modelVersionId: result.model_version_id,
      }))
      // Keep everything the human made or changed; replace only untouched AI boxes.
      const kept = latest.current.shapes.filter((s) => !(s.source === 'ai' && s.kind === 'box'))
      commit([...kept, ...suggestions])
      const ignored = Object.entries(result.ignored)
        .map(([name, n]) => `${n} × ${name}`)
        .join(', ')
      setAiMessage(
        `${suggestions.length} suggestion${suggestions.length === 1 ? '' : 's'} · ` +
          `${Math.round(result.latency_ms)} ms${ignored ? ` · ignored: ${ignored}` : ''}`,
      )
    } catch (e) {
      setAiMessage(`⚠ ${errorMessage(e)}`)
    } finally {
      setAiBusy(false)
    }
  }

  function clearMagic() {
    magicRequest.current += 1 // results still in flight are ignored
    setMagicPoints([])
    setMagicPreview(null)
  }

  async function magicClick(point: Point, label: 0 | 1) {
    if (!segmenterReady) {
      setAiMessage('⚠ The segmentation model is not ready yet')
      return
    }
    const points = [...magicPoints, { point, label }]
    setMagicPoints(points)
    if (!points.some((m) => m.label === 1)) return // SAM needs at least one "object" click
    const request = ++magicRequest.current
    setAiMessage('Segmenting…')
    try {
      const result = unwrap(
        await api.POST('/api/images/{image_id}/segment', {
          ...annotationsPath,
          body: { points: points.map((m) => m.point), labels: points.map((m) => m.label) },
        }),
      )
      if (request !== magicRequest.current) return
      setMagicPreview({
        points: result.points.map(([x = 0, y = 0]): Point => [x, y]),
        score: result.score,
        modelVersionId: result.model_version_id,
      })
      setAiMessage(
        `Outline ready (${Math.round(result.latency_ms)} ms) · Enter = accept, ` +
          'click to add, Shift + click to remove areas, Esc = cancel',
      )
    } catch (e) {
      if (request === magicRequest.current) setAiMessage(`⚠ ${errorMessage(e)}`)
    }
  }

  function acceptMagic() {
    if (!magicPreview) return false
    create({
      ...newPolygon(magicPreview.points, p.activeClass),
      source: 'ai',
      score: magicPreview.score,
      modelVersionId: magicPreview.modelVersionId,
    })
    clearMagic()
    setAiMessage(undefined)
    return true
  }

  // When an image opens in assisted mode: pre-label it once (only if it has no annotations
  // yet) and let SAM encode it in the background, so the first magic click is fast.
  const assistedOnce = useRef({ prepared: false, suggested: false })
  const autoAssist = useEffectEvent(() => {
    if (!assisted) return
    const once = assistedOnce.current
    if (segmenterReady && !once.prepared) {
      once.prepared = true
      api.POST('/api/images/{image_id}/segment/prepare', annotationsPath).catch(() => {})
    }
    if (detectorReady && !once.suggested && initialShapes.length === 0 && image.status === 'todo') {
      once.suggested = true
      void suggest()
    }
  })
  useEffect(() => {
    const timer = setTimeout(autoAssist, 0) // after the first paint, outside the effect itself
    return () => clearTimeout(timer)
  }, [detectorReady, segmenterReady, assisted])

  // ---------- saving ----------

  /** Save the newest shapes; saves run one after another, never in parallel. */
  function save(): Promise<void> {
    const { shapes: current, version: savingVersion } = latest.current
    saveQueue.current = saveQueue.current.then(async () => {
      setSaving(true)
      try {
        unwrap(
          await api.PUT('/api/images/{image_id}/annotations', {
            ...annotationsPath,
            body: current.map(toApi),
          }),
        )
        setSavedVersion((v) => Math.max(v, savingVersion))
        setSaveError(undefined)
        if (current.length > 0) setStatus((s) => (s === 'todo' ? 'in_progress' : s))
      } catch (e) {
        setSaveError(e instanceof Error ? e.message : String(e))
      } finally {
        setSaving(false)
      }
    })
    return saveQueue.current
  }

  /** Make sure everything is stored (before leaving this image). */
  async function flush() {
    if (latest.current.version !== latest.current.savedVersion) await save()
    else await saveQueue.current
  }

  // Autosave shortly after the last change.
  const autosave = useEffectEvent(() => void save())
  useEffect(() => {
    if (version === savedVersion) return
    const timer = setTimeout(autosave, AUTOSAVE_DELAY_MS)
    return () => clearTimeout(timer)
  }, [version, savedVersion])

  // Safety net: tab closed or page left with unsaved changes -> send them anyway.
  // `keepalive` lets the request finish even after the page is gone.
  useEffect(() => {
    const sendUnsaved = () => {
      const { shapes: current, version: v, savedVersion: saved } = latest.current
      if (v === saved) return
      void fetch(`/api/images/${image.id}/annotations`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(current.map(toApi)),
        keepalive: true,
      })
    }
    window.addEventListener('pagehide', sendUnsaved)
    return () => {
      window.removeEventListener('pagehide', sendUnsaved)
      sendUnsaved() // component unmounts (navigation inside the app)
    }
  }, [image.id])

  // ---------- navigation and status ----------

  async function goTo(path: string) {
    await flush()
    void navigate(path)
  }
  const imagePath = (id: number) => `/projects/${project.id}/annotate/${id}`
  const goPrev = () => position.prevId && void goTo(imagePath(position.prevId))
  const goNext = () => position.nextId && void goTo(imagePath(position.nextId))

  async function setImageStatus(next: ImageStatus) {
    await flush()
    unwrap(
      await api.PATCH('/api/images/{image_id}', { ...annotationsPath, body: { status: next } }),
    )
    setStatus(next)
  }

  async function markDoneAndNext() {
    await setImageStatus('done')
    void navigate(position.nextId ? imagePath(position.nextId) : `/projects/${project.id}`)
  }

  // ---------- keyboard shortcuts ----------

  const onKeyDown = useEffectEvent((e: KeyboardEvent) => {
    if (isTyping(e.target)) return
    if (e.key === ' ') {
      e.preventDefault() // don't scroll the page
      setPanning(true)
      return
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault()
      undo()
      return
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return
    if (e.key === 'Enter' && p.tool === 'magic' && magicPreview) {
      e.preventDefault()
      acceptMagic()
      return
    }

    const classIndex = Number(e.key) - 1
    if (classIndex >= 0 && classIndex < Math.min(classes.length, 9)) {
      chooseClass(classes[classIndex] ?? p.activeClass)
      return
    }
    switch (e.key.toLowerCase()) {
      case 'v':
        return chooseTool('select')
      case 'b':
        return chooseTool('box')
      case 'p':
        return chooseTool('polygon')
      case 'm':
        if (assisted) chooseTool('magic')
        return
      case 's':
        void suggest()
        return
      case 'f':
        return setView(null)
      case 'delete':
      case 'backspace':
        if (selectedId) remove(selectedId)
        return
      case 'escape':
        if (magicPoints.length > 0) {
          clearMagic()
          setAiMessage(undefined)
        } else if (draft.length > 0) setDraft([])
        else setSelectedId(null)
        return
      case 'enter':
        e.preventDefault()
        if (!closePolygon()) void markDoneAndNext()
        return
      case 'a':
      case 'arrowleft':
        return goPrev()
      case 'd':
      case 'arrowright':
        return goNext()
    }
  })
  const onKeyUp = useEffectEvent((e: KeyboardEvent) => {
    if (e.key === ' ') setPanning(false)
  })
  useEffect(() => {
    const down = (e: KeyboardEvent) => onKeyDown(e)
    const up = (e: KeyboardEvent) => onKeyUp(e)
    const blur = () => setPanning(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [])

  // ---------- layout ----------

  const dirty = version !== savedVersion
  const saveLabel = saveError
    ? `⚠ Not saved: ${saveError}`
    : saving
      ? 'Saving…'
      : dirty
        ? 'Unsaved changes'
        : 'All changes saved'

  return (
    <div className="annotator">
      <div className="toolbar">
        <button type="button" onClick={() => void goTo(`/projects/${project.id}`)}>
          ← {project.name}
        </button>
        <span className="filename" title={image.filename}>
          {image.filename}{' '}
          <span className="muted">
            ({position.index + 1} / {position.total})
          </span>
        </span>
        <button type="button" disabled={!position.prevId} onClick={goPrev} title="Previous (A)">
          ‹
        </button>
        <button type="button" disabled={!position.nextId} onClick={goNext} title="Next (D)">
          ›
        </button>

        <span className="separator" />
        <div className="segmented" role="group" aria-label="Annotation mode">
          {(['manual', 'assisted'] as const).map((m) => (
            <button
              key={m}
              type="button"
              className={p.mode === m ? 'active' : ''}
              onClick={() => chooseMode(m)}
              title={m === 'manual' ? 'Label by hand only' : 'AI suggestions and magic click'}
            >
              {m === 'manual' ? 'Manual' : 'Assisted'}
            </button>
          ))}
        </div>

        <span className="separator" />
        {TOOLS.filter((t) => assisted || !t.assistedOnly).map(({ tool, key, label }) => (
          <button
            key={tool}
            type="button"
            className={p.tool === tool ? 'active' : ''}
            onClick={() => chooseTool(tool)}
            title={`${label} (${key})`}
          >
            {label} <kbd>{key}</kbd>
          </button>
        ))}

        <span className="separator" />
        {classes.map((c, i) => (
          <button
            key={c}
            type="button"
            className={`class-button ${p.activeClass === c ? 'active' : ''}`}
            style={{ borderColor: classColor(classes, c) }}
            onClick={() => chooseClass(c)}
            title={i < 9 ? `Class ${c} (${i + 1})` : c}
          >
            <span className="dot" style={{ background: classColor(classes, c) }} />
            {c} {i < 9 && <kbd>{i + 1}</kbd>}
          </button>
        ))}

        {assisted && (
          <>
            <span className="separator" />
            <button
              type="button"
              className="ai-button"
              onClick={() => void suggest()}
              disabled={!detectorReady || aiBusy}
              title={
                detectorReady ? 'Suggest boxes with the detector (S)' : modelHint(p.ai, 'detector')
              }
            >
              ✨ Suggest <kbd>S</kbd>
            </button>
            <label
              className="confidence"
              title="Only suggestions the model is at least this sure about"
            >
              min. confidence
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={p.confidence}
                onChange={(e) => p.setConfidence(Number(e.target.value))}
              />
              <output>{p.confidence.toFixed(2)}</output>
            </label>
          </>
        )}

        <span className="separator" />
        <button type="button" onClick={undo} disabled={history.length === 0} title="Undo (Ctrl+Z)">
          Undo
        </button>
        <button type="button" onClick={() => setView(null)} title="Fit image (F)">
          Fit
        </button>

        <span className="spacer" />
        {assisted && aiMessage && (
          <span className={`ai-message ${aiMessage.startsWith('⚠') ? 'error' : ''}`}>
            {aiMessage}
          </span>
        )}
        <span className={`save-state ${saveError ? 'error' : ''}`}>{saveLabel}</span>
        <span className={`badge ${status}`}>{status.replace('_', ' ')}</span>
        {status === 'done' ? (
          <button type="button" onClick={() => void setImageStatus('in_progress')}>
            Reopen
          </button>
        ) : (
          <button type="button" className="primary" onClick={() => void markDoneAndNext()}>
            Done ✓ <kbd>Enter</kbd>
          </button>
        )}
      </div>

      <div className="workspace">
        <AnnotationCanvas
          image={htmlImage}
          shapes={shapes}
          classes={classes}
          tool={p.tool}
          activeClass={p.activeClass}
          selectedId={selectedId}
          hoveredId={hoveredId}
          panning={panning}
          view={view}
          draft={draft}
          onViewChange={setView}
          onDraftChange={setDraft}
          onSelect={setSelectedId}
          onHover={setHoveredId}
          onCreate={create}
          onEditStart={remember}
          onEdit={edit}
          magicPoints={magicPoints}
          magicPreview={magicPreview?.points ?? null}
          onMagicClick={(point, label) => void magicClick(point, label)}
        />

        <aside className="sidebar">
          <h3>Annotations ({shapes.length})</h3>
          {shapes.length === 0 && (
            <p className="muted small">Draw a box (B) or a polygon (P) on the image.</p>
          )}
          <ul className="annotation-list">
            {shapes.map((s) => (
              <li
                key={s.id}
                className={`${s.id === selectedId ? 'selected' : ''} ${s.id === hoveredId ? 'hovered' : ''}`}
                onMouseEnter={() => setHoveredId(s.id)}
                onMouseLeave={() => setHoveredId(null)}
                onClick={() => {
                  chooseTool('select')
                  setSelectedId(s.id)
                }}
              >
                <span className="dot" style={{ background: classColor(classes, s.className) }} />
                <select
                  value={s.className}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => changeClass(s.id, e.target.value)}
                  aria-label="Class"
                >
                  {classes.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
                <span className="muted small">
                  {s.kind}
                  {s.source !== 'manual' && ` · ${s.source.replace('_', ' ')}`}
                  {s.source === 'ai' && s.score != null && ` ${s.score.toFixed(2)}`}
                </span>
                <button
                  type="button"
                  className="icon"
                  title="Delete"
                  onClick={(e) => {
                    e.stopPropagation()
                    remove(s.id)
                  }}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>

          <h3>Shortcuts</h3>
          <dl className="shortcuts">
            {[...SHORTCUTS, ...(assisted ? AI_SHORTCUTS : [])].map(([keys, action]) => (
              <div key={keys + action}>
                <dt>{keys}</dt>
                <dd>{action}</dd>
              </div>
            ))}
          </dl>
        </aside>
      </div>
    </div>
  )
}

/** Loads everything for one image, then shows the editor. */
function ImageLoader(
  props: Omit<EditorProps, 'image' | 'htmlImage' | 'initialShapes'> & { imageId: number },
) {
  const { imageId, ...rest } = props
  const load = useCallback(async () => {
    const path = { params: { path: { image_id: imageId } } }
    const [image, annotations] = await Promise.all([
      api.GET('/api/images/{image_id}', path).then(unwrap),
      api.GET('/api/images/{image_id}/annotations', path).then(unwrap),
    ])
    return { image, shapes: annotations.map(fromApi) }
  }, [imageId])
  const { data, error } = useAsync(load)
  const file = useHtmlImage(imageFileUrl(imageId))

  if (error) return <p className="error page">{error}</p>
  if (file.failed) return <p className="error page">Could not load the image file.</p>
  if (!data || !file.image) return <p className="muted page">Loading image…</p>
  return <Editor {...rest} image={data.image} htmlImage={file.image} initialShapes={data.shapes} />
}

export default function AnnotatorPage() {
  const params = useParams()
  const projectId = Number(params.projectId)
  const imageId = Number(params.imageId)
  const [tool, setTool] = useState<Tool>('box')
  const [activeClass, setActiveClass] = useState<string>()
  // Remembered in this browser; the effort study switches the mode per session.
  const [mode, setMode] = usePersistentState<Mode>('annotator.mode', 'assisted')
  const [confidence, setConfidence] = usePersistentState('annotator.confidence', 0.25)
  const ai = useAiStatus()

  const load = useCallback(async () => {
    const path = { params: { path: { project_id: projectId } } }
    const [project, images] = await Promise.all([
      api.GET('/api/projects/{project_id}', path).then(unwrap),
      api
        .GET('/api/projects/{project_id}/images', {
          params: { ...path.params, query: { limit: 1000 } },
        })
        .then(unwrap),
    ])
    return { project, imageIds: images.map((i) => i.id) }
  }, [projectId])
  const { data, error } = useAsync(load)

  if (error) return <p className="error page">{error}</p>
  if (!data) return <p className="muted page">Loading…</p>

  const { project, imageIds } = data
  const index = imageIds.indexOf(imageId)
  if (index === -1) {
    return (
      <main className="page">
        <p className="error">This image is not part of the project.</p>
        <Link to={`/projects/${projectId}`}>Back to the project</Link>
      </main>
    )
  }
  return (
    <ImageLoader
      key={imageId} // new image -> fresh editor state
      imageId={imageId}
      project={project}
      position={{
        index,
        total: imageIds.length,
        prevId: imageIds[index - 1],
        nextId: imageIds[index + 1],
      }}
      tool={tool}
      setTool={setTool}
      activeClass={
        activeClass && project.classes.includes(activeClass)
          ? activeClass
          : (project.classes[0] ?? '')
      }
      setActiveClass={setActiveClass}
      mode={mode}
      setMode={setMode}
      confidence={confidence}
      setConfidence={setConfidence}
      ai={ai}
    />
  )
}
