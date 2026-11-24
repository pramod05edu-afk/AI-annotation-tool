/**
 * The annotator's in-memory shapes and their conversion to and from the API.
 * All coordinates are ORIGINAL image pixels; zoom and pan only change how they are drawn.
 */
import type { AnnotationIn, AnnotationRead } from '../api/client'

export type Box = [number, number, number, number] // x_min, y_min, x_max, y_max
export type Point = [number, number]
export type Source = AnnotationIn['source']

interface ShapeBase {
  id: string // local id: the API replaces all annotations on save, so server ids change
  className: string
  source: Source
  score: number | null
  modelVersionId: number | null
}
export type BoxShape = ShapeBase & { kind: 'box'; box: Box }
export type PolygonShape = ShapeBase & { kind: 'polygon'; points: Point[] }
export type Shape = BoxShape | PolygonShape

export type Tool = 'select' | 'box' | 'polygon' | 'smart' | 'magic'

/** A magic-click point: label 1 = "this is the object", 0 = "this is not the object". */
export interface MagicPoint {
  point: Point
  label: 0 | 1
}
export interface View {
  scale: number // zoom: screen pixels per image pixel
  x: number // pan: where the image's top-left corner is on screen
  y: number
}

export const MIN_BOX_SIZE = 3 // px: smaller drags are treated as accidental clicks

export const newId = () => crypto.randomUUID()

const manual = { source: 'manual' as Source, score: null, modelVersionId: null }

export const newBox = (box: Box, className: string): BoxShape => ({
  id: newId(),
  kind: 'box',
  className,
  box: [round(box[0]), round(box[1]), round(box[2]), round(box[3])],
  ...manual,
})

export const newPolygon = (points: Point[], className: string): PolygonShape => ({
  id: newId(),
  kind: 'polygon',
  className,
  points: points.map(([x, y]) => [round(x), round(y)]),
  ...manual,
})

/** What the API sends back. Its JSON arrays have no fixed length, hence `number[]`. */
type ApiAnnotation = Pick<
  AnnotationRead,
  'class_name' | 'source' | 'score' | 'model_version_id'
> & {
  geometry: { type: 'box'; bbox: number[] } | { type: 'polygon'; points: number[][] }
}

/** Turn an API annotation into an editable shape (arrays back into fixed-size tuples). */
export function fromApi(a: ApiAnnotation): Shape {
  const base = {
    id: newId(),
    className: a.class_name,
    source: a.source,
    score: a.score,
    modelVersionId: a.model_version_id,
  }
  if (a.geometry.type === 'box') {
    const [x1 = 0, y1 = 0, x2 = 0, y2 = 0] = a.geometry.bbox
    return { ...base, kind: 'box', box: [x1, y1, x2, y2] }
  }
  const points = a.geometry.points.map(([x = 0, y = 0]): Point => [x, y])
  return { ...base, kind: 'polygon', points }
}

export function toApi(s: Shape): AnnotationIn {
  return {
    class_name: s.className,
    geometry:
      s.kind === 'box' ? { type: 'box', bbox: s.box } : { type: 'polygon', points: s.points },
    source: s.source,
    score: s.score,
    model_version_id: s.modelVersionId,
  }
}

/** A human changed this shape: an AI suggestion becomes "ai_edited" (for the effort study). */
export const markEdited = (source: Source): Source => (source === 'ai' ? 'ai_edited' : source)

const clamp = (v: number, min: number, max: number) => Math.min(Math.max(v, min), max)

export const clampPoint = ([x, y]: Point, width: number, height: number): Point => [
  clamp(x, 0, width),
  clamp(y, 0, height),
]

/** Box from two corner points in any drag direction, kept inside the image. */
export function boxFromCorners(a: Point, b: Point, width: number, height: number): Box {
  const [x1, y1] = clampPoint(a, width, height)
  const [x2, y2] = clampPoint(b, width, height)
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)]
}

export const isBigEnough = ([x1, y1, x2, y2]: Box) =>
  x2 - x1 >= MIN_BOX_SIZE && y2 - y1 >= MIN_BOX_SIZE

export function bounds(s: Shape): Box {
  if (s.kind === 'box') return s.box
  const xs = s.points.map((p) => p[0])
  const ys = s.points.map((p) => p[1])
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
}

/** Round to 0.1 px: keeps saved JSON small without any visible loss. */
export const round = (v: number) => Math.round(v * 10) / 10

/** Distinct, colour-blind-friendly colours (Okabe-Ito palette), one per class index. */
const PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
export const classColor = (classes: string[], name: string) =>
  PALETTE[Math.max(0, classes.indexOf(name)) % PALETTE.length] ?? '#E69F00'
