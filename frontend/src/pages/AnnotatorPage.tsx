/**
 * Annotate one image: draw / edit / delete shapes, switch images, autosave, mark done.
 *
 * Structure (each level only renders once its data is loaded):
 *   AnnotatorPage   project + image list, tool, class, mode, AI status (kept across images)
 *   └─ ImageLoader  this image's file, status and saved annotations   (remounted per image)
 *      └─ Editor    the editable shapes, undo history, saving, keyboard shortcuts
 */
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Keyboard,
  LassoSelect,
  LoaderCircle,
  type LucideIcon,
  Maximize,
  MousePointer2,
  Pentagon,
  RotateCcw,
  ScanSearch,
  Square,
  Trash2,
  Undo2,
} from 'lucide-react'
import { type CSSProperties, useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'
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
import { logEvent, type LogEventName, type Mode } from '../annotator/interactionLog'
import {
  bounds,
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
import Modal from '../components/Modal'
import ModeSwitch from '../components/ModeSwitch'
import { useAsync, useHtmlImage, usePersistentState } from '../hooks'

const AUTOSAVE_DELAY_MS = 800
const MAX_UNDO = 100

const TOOLS: {
  tool: Tool
  key: string
  label: string
  icon: LucideIcon
  assistedOnly?: boolean
}[] = [
  { tool: 'select', key: 'V', label: 'Select and edit', icon: MousePointer2 },
  { tool: 'box', key: 'B', label: 'Draw a box', icon: Square },
  { tool: 'polygon', key: 'P', label: 'Draw a polygon', icon: Pentagon },
  {
    tool: 'smart',
    key: 'C',
    label: 'Smart click: click an object, get its box and class',
    icon: Crosshair,
    assistedOnly: true,
  },
  {
    tool: 'magic',
    key: 'M',
    label: 'Magic: click an object, get its outline (SAM)',
    icon: LassoSelect,
    assistedOnly: true,
  },
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
  ['?', 'show these shortcuts'],
]

const AI_SHORTCUTS: [string, string][] = [
  ['S', 'suggest boxes (AI)'],
  ['C', 'smart click: click an object -> box with its class'],
  ['E', 'find similar: select a box, the AI adds more objects like it'],
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

const MODEL_NAMES = {
  detector: 'fine-tuned detector',
  general: 'general AI model',
  segmenter: 'segmentation model',
} as const

/** Why an AI button is disabled, for its tooltip. */
function modelHint(ai: AiStatus | undefined, which: keyof typeof MODEL_NAMES) {
  const slot = ai?.[which]
  const name = MODEL_NAMES[which]
  if (!slot) return 'Checking the AI models…'
  if (slot.state === 'loading') return `The ${name} is loading…`
  if (slot.state === 'failed') return `The ${name} is unavailable: ${slot.error ?? 'unknown error'}`
  return `The ${name} is ${slot.state}`
}

/** "yoloe-26s-seg" -> "YOLOE (general)", "yolo11n_helmet" -> "yolo11n_helmet (fine-tuned)". */
const modelLabel = (name: string | null | undefined) =>
  !name ? 'AI' : name.startsWith('yoloe') ? 'YOLOE (general)' : `${name} (fine-tuned)`

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
  modeLocked: boolean // the project prescribes the mode (effort study)
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
  const [showShortcuts, setShowShortcuts] = useState(false)

  // AI assistance
  const assisted = p.mode === 'assisted'
  const generalReady = p.ai?.general.state === 'ready'
  // Suggest / smart click: the fine-tuned specialist for the labels it was trained on, the
  // general model for all others (the backend decides per project); either one is enough.
  const detectorReady = p.ai?.detector.state === 'ready' || generalReady
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

  /** Record an action for the effort study (time and number of interactions per image). */
  const log = (event: LogEventName, payload: Record<string, unknown> = {}) =>
    logEvent(image.id, p.mode, event, payload)

  function create(shape: Shape) {
    commit([...latest.current.shapes, shape])
    setSelectedId(shape.id)
    log('annotation_created', { kind: shape.kind, source: shape.source, tool: p.tool })
  }

  /** A drag / resize / vertex move begins: one undo step and one "edit" interaction. */
  function startEdit(shape: Shape) {
    remember()
    log('annotation_edited', { kind: shape.kind, source: shape.source })
  }

  function edit(shape: Shape) {
    setShapes((all) =>
      all.map((s) => (s.id === shape.id ? { ...shape, source: markEdited(shape.source) } : s)),
    )
    bump()
  }

  function remove(id: string) {
    const shape = shapes.find((s) => s.id === id)
    if (shape) log('annotation_deleted', { kind: shape.kind, source: shape.source })
    commit(shapes.filter((s) => s.id !== id))
    if (selectedId === id) setSelectedId(null)
  }

  function changeClass(id: string, className: string) {
    const shape = shapes.find((s) => s.id === id)
    if (!shape || shape.className === className) return
    log('class_changed', { from: shape.className, to: className, source: shape.source })
    commit(shapes.map((s) => (s.id === id ? { ...s, className, source: markEdited(s.source) } : s)))
  }

  function undo() {
    const previous = history.at(-1)
    if (!previous) return
    log('undo')
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
    if (mode === p.mode) return
    logEvent(image.id, mode, 'mode_changed', { from: p.mode, to: mode })
    p.setMode(mode)
    setAiMessage(undefined)
    if (mode === 'manual' && (p.tool === 'magic' || p.tool === 'smart')) chooseTool('box')
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
  async function suggest(auto = false) {
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
        modelVersionId: s.model_version_id ?? result.model_version_id,
      }))
      // Keep everything the human made or changed; replace only untouched AI boxes.
      const before = latest.current.shapes
      const kept = before.filter((s) => !(s.source === 'ai' && s.kind === 'box'))
      commit([...kept, ...suggestions])
      log('suggest', {
        auto,
        count: suggestions.length,
        replaced: before.length - kept.length, // untouched AI boxes from an earlier run
        conf: p.confidence,
        latency_ms: result.latency_ms,
      })
      const ignored = Object.entries(result.ignored)
        .map(([name, n]) => `${n} × ${name}`)
        .join(', ')
      const used = (result.models ?? []).map(modelLabel).join(' + ')
      setAiMessage(
        `${suggestions.length} suggestion${suggestions.length === 1 ? '' : 's'} · ` +
          `${Math.round(result.latency_ms)} ms${used ? ` · ${used}` : ''}` +
          `${ignored ? ` · ignored: ${ignored}` : ''}`,
      )
    } catch (e) {
      setAiMessage(`⚠ ${errorMessage(e)}`)
    } finally {
      setAiBusy(false)
    }
  }

  /**
   * Smart click: the user clicks an object, the detector returns its best box there (also weak
   * candidates the auto-suggest hid), skipping objects that are labelled already.
   */
  async function smartClick(point: Point) {
    if (!detectorReady) {
      setAiMessage(`⚠ ${modelHint(p.ai, 'general')}`)
      return
    }
    setAiMessage('Looking for an object…')
    try {
      const result = unwrap(
        await api.POST('/api/images/{image_id}/click-box', {
          ...annotationsPath,
          body: { point, exclude: latest.current.shapes.map(bounds) },
        }),
      )
      const found = result.suggestion
      log('smart_click', { found: Boolean(found), latency_ms: result.latency_ms })
      if (!found) {
        setAiMessage('No object found here · draw it with the box tool (B)')
        return
      }
      create({
        ...newBox(toBox(found.bbox), found.class_name),
        source: 'ai',
        score: found.score,
        modelVersionId: found.model_version_id ?? result.model_version_id,
      })
      setAiMessage(
        `${found.class_name} added (${found.score.toFixed(2)}) · ${Math.round(result.latency_ms)} ms` +
          ` · ${modelLabel(found.model)} · wrong class? press its number key`,
      )
    } catch (e) {
      setAiMessage(`⚠ ${errorMessage(e)}`)
    }
  }

  /**
   * Find similar: the selected box is an EXAMPLE, the general model adds more objects that look
   * like it (visual prompt). Works for labels no model knows by name (a product, a defect).
   */
  async function findSimilar() {
    if (!assisted || aiBusy) return
    const example = latest.current.shapes.find((s) => s.id === selectedId)
    if (!example || example.kind !== 'box') {
      setAiMessage('⚠ Select a box first (V, then click it): it is the example to search for')
      return
    }
    if (!generalReady) {
      setAiMessage(`⚠ ${modelHint(p.ai, 'general')}`)
      return
    }
    setAiBusy(true)
    setAiMessage('Looking for similar objects…')
    try {
      const result = unwrap(
        await api.POST('/api/images/{image_id}/find-similar', {
          ...annotationsPath,
          body: {
            examples: [example.box],
            class_name: example.className,
            conf: p.confidence,
            exclude: latest.current.shapes.map(bounds),
          },
        }),
      )
      const found: Shape[] = result.suggestions.map((s) => ({
        ...newBox(toBox(s.bbox), s.class_name),
        source: 'ai',
        score: s.score,
        modelVersionId: s.model_version_id ?? result.model_version_id,
      }))
      if (found.length > 0) commit([...latest.current.shapes, ...found])
      log('find_similar', { found: found.length, latency_ms: result.latency_ms })
      setAiMessage(
        found.length > 0
          ? `${found.length} more “${example.className}” found · ` +
              `${Math.round(result.latency_ms)} ms · YOLOE (general)`
          : 'No similar objects found · try a lower min. confidence',
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
      log('magic_click', { label, clicks: points.length, latency_ms: result.latency_ms })
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
    log('magic_accepted', { clicks: magicPoints.length })
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
      void suggest(true)
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

  // Effort study: when was the image opened / left (the time in between = effort).
  const logOpenClose = useEffectEvent((event: 'image_opened' | 'image_closed') =>
    log(event, { annotations: latest.current.shapes.length }),
  )
  useEffect(() => {
    logOpenClose('image_opened')
    return () => logOpenClose('image_closed')
  }, [])

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
    const current = latest.current.shapes
    log(next === 'done' ? 'image_done' : 'image_reopened', {
      annotations: current.length,
      // How much of the final result the AI produced, unchanged or corrected (effort study).
      by_source: Object.fromEntries(
        (['manual', 'ai', 'ai_edited'] as const).map((src) => [
          src,
          current.filter((s) => s.source === src).length,
        ]),
      ),
    })
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
      case 'c':
        if (assisted) chooseTool('smart')
        return
      case 's':
        void suggest()
        return
      case 'e':
        void findSimilar()
        return
      case 'f':
        return setView(null)
      case '?':
        return setShowShortcuts((open) => !open)
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

  // AI results float over the canvas for a few seconds; instructions stay until used.
  const hideAiMessage = useEffectEvent(() => setAiMessage(undefined))
  useEffect(() => {
    if (!aiMessage || aiMessage.startsWith('Outline ready')) return
    const timer = setTimeout(hideAiMessage, 4500)
    return () => clearTimeout(timer)
  }, [aiMessage])

  const dirty = version !== savedVersion
  const saveLabel = saveError
    ? `⚠ Not saved: ${saveError}`
    : saving
      ? 'Saving…'
      : dirty
        ? 'Unsaved changes'
        : 'All changes saved'

  const confidenceClass = (score: number) => (score >= 0.7 ? 'high' : score >= 0.5 ? 'mid' : 'low')
  const aiError = aiMessage?.startsWith('⚠')

  return (
    <div className="annotator">
      <nav className="tool-rail" aria-label="Tools">
        {TOOLS.filter((t) => assisted || !t.assistedOnly).map(
          ({ tool, key, label, icon: Icon, assistedOnly }) => (
            <button
              key={tool}
              type="button"
              className={`rail-button ${assistedOnly ? 'ai' : ''} ${p.tool === tool ? 'active' : ''}`}
              onClick={() => chooseTool(tool)}
              data-tip={`${label} (${key})`}
              aria-label={label}
            >
              <Icon size={19} />
              <small>{key}</small>
            </button>
          ),
        )}
        <span className="rail-separator" />
        <button
          type="button"
          className="rail-button"
          onClick={undo}
          disabled={history.length === 0}
          data-tip="Undo (Ctrl+Z)"
          aria-label="Undo"
        >
          <Undo2 size={19} />
        </button>
        <button
          type="button"
          className="rail-button"
          onClick={() => setView(null)}
          data-tip="Fit image (F)"
          aria-label="Fit image"
        >
          <Maximize size={19} />
        </button>
        <button
          type="button"
          className="rail-button rail-bottom"
          onClick={() => setShowShortcuts(true)}
          data-tip="Keyboard shortcuts (?)"
          aria-label="Keyboard shortcuts"
        >
          <Keyboard size={19} />
          <small>?</small>
        </button>
      </nav>

      <div className="annotator-main">
        <div className="toolbar">
          <button
            type="button"
            className="ghost crumb"
            onClick={() => void goTo(`/projects/${project.id}`)}
            title="Back to the project"
          >
            <ChevronLeft size={16} />
            {project.name}
          </button>
          <span className="filename" title={image.filename}>
            {image.filename}
          </span>
          <div className="pager">
            <button
              type="button"
              className="icon"
              disabled={!position.prevId}
              onClick={goPrev}
              title="Previous (A)"
            >
              <ChevronLeft size={16} />
            </button>
            <span>
              {position.index + 1} / {position.total}
            </span>
            <button
              type="button"
              className="icon"
              disabled={!position.nextId}
              onClick={goNext}
              title="Next (D)"
            >
              <ChevronRight size={16} />
            </button>
          </div>

          <span className="separator" />
          <ModeSwitch mode={p.mode} locked={p.modeLocked} onChange={chooseMode} />

          {assisted && (
            <>
              <button
                type="button"
                className="signal suggest"
                onClick={() => void suggest()}
                disabled={!detectorReady || aiBusy}
                title={
                  detectorReady
                    ? 'Suggest boxes for this project’s labels (S)'
                    : modelHint(p.ai, 'general')
                }
              >
                {aiBusy ? (
                  <LoaderCircle size={15} className="spin" />
                ) : (
                  <span className="ai-tag">AI</span>
                )}
                Suggest <kbd>S</kbd>
              </button>
              <button
                type="button"
                className="find-similar"
                onClick={() => void findSimilar()}
                disabled={!generalReady || aiBusy}
                title={
                  generalReady
                    ? 'Select a box, then find more objects like it (E)'
                    : modelHint(p.ai, 'general')
                }
              >
                <ScanSearch size={15} />
                Find similar <kbd>E</kbd>
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

          <span className="spacer" />
          <span
            className={`save-state ${saveError ? 'error' : dirty || saving ? 'pending' : 'ok'}`}
          >
            {!saveError && !dirty && !saving && <Check size={14} />}
            {saveLabel}
          </span>
        </div>

        <div className="classbar">
          <div className="class-picker" role="group" aria-label="Class for new shapes">
            {classes.map((c, i) => (
              <button
                key={c}
                type="button"
                className={p.activeClass === c ? 'active' : ''}
                style={{ '--class-color': classColor(classes, c) } as CSSProperties}
                onClick={() => chooseClass(c)}
                title={i < 9 ? `Class ${c} (${i + 1})` : c}
              >
                <span className="dot" />
                {c}
                {i < 9 && <kbd>{i + 1}</kbd>}
              </button>
            ))}
          </div>
        </div>

        <div className="workspace">
          <div className="canvas-area">
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
              onEditStart={startEdit}
              onEdit={edit}
              magicPoints={magicPoints}
              magicPreview={magicPreview?.points ?? null}
              onMagicClick={(point, label) => void magicClick(point, label)}
              onSmartClick={(point) => void smartClick(point)}
            />
            <div className="canvas-hint">wheel zoom · space + drag pan · F fit</div>
            {assisted && aiMessage && (
              <div className={`ai-toast ${aiError ? 'error' : ''}`} role="status">
                <span className="ai-toast-icon">
                  <span className="ai-tag">AI</span>
                </span>
                {aiMessage.replace(/^⚠ /, '')}
              </div>
            )}
          </div>

          <aside className="sidebar">
            <div className="sidebar-head">
              <div className="sidebar-title">
                <h3>
                  Labels <span>{shapes.length}</span>
                </h3>
                <span className={`badge ${status}`}>{status.replace('_', ' ')}</span>
              </div>
              <div className="progress thin" title="Position in the project">
                <div
                  className="progress-done"
                  style={{ width: `${(100 * (position.index + 1)) / position.total}%` }}
                />
              </div>
            </div>

            {shapes.length === 0 && (
              <p className="muted small empty-hint">
                No labels yet. Draw a box (B)
                {assisted ? ', press S for AI suggestions or C for smart click' : ''}.
              </p>
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
                  <span
                    className="swatch"
                    style={{ background: classColor(classes, s.className) }}
                  />
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
                  <span className="source-tag">
                    {s.kind === 'polygon' ? 'poly' : 'box'}
                    {s.source !== 'manual' && (
                      <>
                        {' · '}
                        <b>AI</b>
                        {s.source === 'ai_edited' && ' ✎'}
                      </>
                    )}
                  </span>
                  {s.score != null && s.source !== 'manual' ? (
                    <span
                      className={`confidence-bar ${confidenceClass(s.score)}`}
                      title={`confidence ${s.score.toFixed(2)}`}
                    >
                      <span className="track">
                        <i style={{ width: `${Math.round(s.score * 100)}%` }} />
                      </span>
                      <small>{s.score.toFixed(2)}</small>
                    </span>
                  ) : (
                    <span className="confidence-bar empty" />
                  )}
                  <button
                    type="button"
                    className="icon"
                    title="Delete (Del)"
                    onClick={(e) => {
                      e.stopPropagation()
                      remove(s.id)
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
              ))}
            </ul>
            {shapes.some((s) => s.source !== 'manual') && (
              <p className="legend">
                dashed = AI proposal, untouched
                <br />
                solid = yours · ✎ = AI, corrected by you
              </p>
            )}

            <div className="sidebar-foot">
              {status === 'done' ? (
                <button
                  type="button"
                  className="wide-button"
                  onClick={() => void setImageStatus('in_progress')}
                >
                  <RotateCcw size={16} /> Reopen
                </button>
              ) : (
                <button
                  type="button"
                  className="primary wide-button"
                  onClick={() => void markDoneAndNext()}
                >
                  <Check size={16} /> Done <kbd>Enter</kbd>
                </button>
              )}
            </div>
          </aside>
        </div>
      </div>

      {showShortcuts && (
        <Modal title="Keyboard shortcuts" onClose={() => setShowShortcuts(false)} wide>
          <dl className="shortcuts">
            {[...SHORTCUTS, ...(assisted ? AI_SHORTCUTS : [])].map(([keys, action]) => (
              <div key={keys + action}>
                <dt>
                  <kbd>{keys}</kbd>
                </dt>
                <dd>{action}</dd>
              </div>
            ))}
          </dl>
        </Modal>
      )}
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
      // A project with a fixed mode (effort study) overrides the annotator's own choice.
      mode={project.annotation_mode ?? mode}
      modeLocked={project.annotation_mode != null}
      setMode={setMode}
      confidence={confidence}
      setConfidence={setConfidence}
      ai={ai}
    />
  )
}
