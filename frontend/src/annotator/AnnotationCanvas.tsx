/**
 * The drawing surface (Konva = HTML canvas with shapes, events and drag & drop).
 *
 * Coordinates: shapes are stored in ORIGINAL image pixels. The Stage is scaled and moved
 * (zoom / pan), so Konva converts for us: `stage.getRelativePointerPosition()` returns the
 * mouse position in image pixels, i.e. x_image = (x_screen - pan_x) / zoom.
 */
import type Konva from 'konva'
import { useEffect, useEffectEvent, useRef, useState } from 'react'
import {
  Circle,
  Image as KonvaImage,
  Layer,
  Line,
  Rect,
  Stage,
  Text,
  Transformer,
} from 'react-konva'

import { useElementSize } from '../hooks'
import {
  type Box,
  boxFromCorners,
  type BoxShape,
  classColor,
  clampPoint,
  isBigEnough,
  newBox,
  type MagicPoint,
  newPolygon,
  type Point,
  type PolygonShape,
  round,
  type Shape,
  type Tool,
  type View,
} from './shapes'

type MouseEvt = Konva.KonvaEventObject<MouseEvent>

export interface CanvasProps {
  image: HTMLImageElement
  shapes: Shape[]
  classes: string[]
  tool: Tool
  activeClass: string
  selectedId: string | null
  hoveredId: string | null
  panning: boolean // Space is held: drag moves the view instead of drawing
  view: View | null // null = fit the whole image
  draft: Point[] // polygon being drawn
  onViewChange: (view: View | null) => void
  onDraftChange: (points: Point[]) => void
  onSelect: (id: string | null) => void
  onHover: (id: string | null) => void
  onCreate: (shape: Shape) => void
  onEditStart: () => void // a drag/resize begins: remember the state for undo
  onEdit: (shape: Shape) => void
  magicPoints: MagicPoint[] // magic-click tool: the clicks so far
  magicPreview: Point[] | null // the outline SAM proposes for them
  onMagicClick: (point: Point, label: 0 | 1) => void
}

const CLOSE_DISTANCE_PX = 10 // clicking this close (on screen) to the first point closes a polygon
const MIN_SCALE = 0.05
const MAX_SCALE = 40

function fitView(width: number, height: number, image: HTMLImageElement): View {
  if (!width || !height) return { scale: 1, x: 0, y: 0 }
  const scale = Math.min(width / image.width, height / image.height) * 0.96
  return { scale, x: (width - image.width * scale) / 2, y: (height - image.height * scale) / 2 }
}

const withAlpha = (hex: string, alpha: number) =>
  hex +
  Math.round(alpha * 255)
    .toString(16)
    .padStart(2, '0')

const roundBox = (b: Box): Box => [round(b[0]), round(b[1]), round(b[2]), round(b[3])]

export default function AnnotationCanvas(p: CanvasProps) {
  const { image, tool } = p
  const { ref: containerRef, width, height } = useElementSize<HTMLDivElement>()
  const stageRef = useRef<Konva.Stage>(null)
  const transformerRef = useRef<Konva.Transformer>(null)
  const [boxStart, setBoxStart] = useState<Point | null>(null)
  const [cursor, setCursor] = useState<Point | null>(null)

  const view = p.view ?? fitView(width, height, image)
  const px = (screenPixels: number) => screenPixels / view.scale // constant on-screen size
  const clampToImage = (pt: Point) => clampPoint(pt, image.width, image.height)

  function pointer(): Point | null {
    const pos = stageRef.current?.getRelativePointerPosition()
    return pos ? [pos.x, pos.y] : null
  }

  // Attach the resize handles (Transformer) to the selected box.
  useEffect(() => {
    const transformer = transformerRef.current
    const stage = stageRef.current
    if (!transformer || !stage) return
    const selected = p.shapes.find((s) => s.id === p.selectedId)
    const node =
      selected?.kind === 'box' && tool === 'select'
        ? stage.findOne((n: Konva.Node) => n.id() === selected.id)
        : undefined
    transformer.nodes(node ? [node] : [])
    transformer.getLayer()?.batchDraw()
  }, [p.selectedId, p.shapes, tool])

  // Finish a box when the mouse button is released, even outside the canvas.
  const finishBox = useEffectEvent(() => {
    const end = pointer() ?? cursor
    if (boxStart && end) {
      const box = boxFromCorners(boxStart, end, image.width, image.height)
      if (isBigEnough(box)) {
        p.onCreate(newBox(box, p.activeClass))
      }
    }
    setBoxStart(null)
  })
  useEffect(() => {
    if (!boxStart) return
    const onUp = () => finishBox()
    window.addEventListener('mouseup', onUp)
    return () => window.removeEventListener('mouseup', onUp)
  }, [boxStart])

  // ---------- Stage events ----------

  function onMouseDown(e: MouseEvt) {
    if (p.panning || e.evt.button !== 0 || tool !== 'box') return
    const pt = pointer()
    if (pt) {
      setBoxStart(clampToImage(pt))
      setCursor(clampToImage(pt))
    }
  }

  function onMouseMove() {
    if (tool === 'select' || tool === 'magic') return // nothing follows the mouse: no re-render
    const pt = pointer()
    setCursor(pt ? clampToImage(pt) : null)
  }

  function onClick(e: MouseEvt) {
    if (p.panning || e.evt.button !== 0) return
    if (tool === 'select') {
      const target = e.target
      if (target === target.getStage() || target.name() === 'background') p.onSelect(null)
      return
    }
    if (tool === 'magic') {
      const pt = pointer()
      // Click = part of the object; Shift + click (or right click) = not part of it.
      if (pt) p.onMagicClick(clampToImage(pt), e.evt.shiftKey ? 0 : 1)
      return
    }
    if (tool !== 'polygon') return
    const pt = pointer()
    if (!pt) return
    const first = p.draft[0]
    const closeEnough =
      first && Math.hypot(pt[0] - first[0], pt[1] - first[1]) < px(CLOSE_DISTANCE_PX)
    if (p.draft.length >= 3 && closeEnough) {
      p.onCreate(newPolygon(p.draft, p.activeClass))
      p.onDraftChange([])
    } else {
      p.onDraftChange([...p.draft, clampToImage(pt)])
    }
  }

  function onContextMenu(e: Konva.KonvaEventObject<PointerEvent>) {
    e.evt.preventDefault() // no browser menu on the canvas
    if (tool !== 'magic' || p.panning) return
    const pt = pointer()
    if (pt) p.onMagicClick(clampToImage(pt), 0)
  }

  function onWheel(e: Konva.KonvaEventObject<WheelEvent>) {
    e.evt.preventDefault()
    const pos = stageRef.current?.getPointerPosition()
    if (!pos) return
    const factor = e.evt.deltaY < 0 ? 1.15 : 1 / 1.15
    const scale = Math.min(Math.max(view.scale * factor, MIN_SCALE), MAX_SCALE)
    // Keep the image point under the mouse in place while zooming.
    const imageX = (pos.x - view.x) / view.scale
    const imageY = (pos.y - view.y) / view.scale
    p.onViewChange({ scale, x: pos.x - imageX * scale, y: pos.y - imageY * scale })
  }

  function onStageDrag(e: Konva.KonvaEventObject<DragEvent>) {
    const stage = e.target.getStage()
    if (stage && e.target === stage)
      p.onViewChange({ scale: view.scale, x: stage.x(), y: stage.y() })
  }

  // ---------- Shape rendering ----------

  const selectable = tool === 'select' && !p.panning

  function renderBox(s: BoxShape, color: string, highlight: boolean) {
    const [x1, y1, x2, y2] = s.box
    const selected = s.id === p.selectedId
    return (
      <Rect
        key={s.id}
        id={s.id}
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        stroke={color}
        strokeWidth={selected ? 3 : 2}
        strokeScaleEnabled={false}
        dash={s.source === 'ai' ? [8, 5] : undefined}
        fill={withAlpha(color, highlight ? 0.35 : 0.1)}
        listening={selectable}
        draggable={selectable && selected}
        onMouseDown={() => selectable && p.onSelect(s.id)}
        onMouseEnter={() => p.onHover(s.id)}
        onMouseLeave={() => p.onHover(null)}
        onDragStart={p.onEditStart}
        onDragEnd={(e) => {
          const node = e.target
          const w = x2 - x1
          const h = y2 - y1
          // Keep the whole box inside the image.
          const x = Math.min(Math.max(node.x(), 0), image.width - w)
          const y = Math.min(Math.max(node.y(), 0), image.height - h)
          node.position({ x, y })
          p.onEdit({ ...s, box: roundBox([x, y, x + w, y + h]) })
        }}
        onTransformStart={p.onEditStart}
        onTransformEnd={(e) => {
          const node = e.target
          const box = boxFromCorners(
            [node.x(), node.y()],
            [node.x() + node.width() * node.scaleX(), node.y() + node.height() * node.scaleY()],
            image.width,
            image.height,
          )
          node.scale({ x: 1, y: 1 }) // Konva resizes by scaling; we store real sizes instead
          if (isBigEnough(box)) p.onEdit({ ...s, box: roundBox(box) })
        }}
      />
    )
  }

  function renderPolygon(s: PolygonShape, color: string, highlight: boolean) {
    const selected = s.id === p.selectedId
    return (
      <Line
        key={s.id}
        id={s.id}
        points={s.points.flat()}
        closed
        stroke={color}
        strokeWidth={selected ? 3 : 2}
        strokeScaleEnabled={false}
        dash={s.source === 'ai' ? [8, 5] : undefined}
        fill={withAlpha(color, highlight ? 0.35 : 0.1)}
        listening={selectable}
        draggable={selectable && selected}
        onMouseDown={() => selectable && p.onSelect(s.id)}
        onMouseEnter={() => p.onHover(s.id)}
        onMouseLeave={() => p.onHover(null)}
        onDragStart={p.onEditStart}
        onDragEnd={(e) => {
          const node = e.target
          const dx = node.x()
          const dy = node.y()
          node.position({ x: 0, y: 0 }) // the move is stored in the points instead
          p.onEdit({
            ...s,
            points: s.points.map(([x, y]) => {
              const [cx, cy] = clampToImage([x + dx, y + dy])
              return [round(cx), round(cy)]
            }),
          })
        }}
      />
    )
  }

  const selected = p.shapes.find((s) => s.id === p.selectedId)
  const draftBox =
    boxStart && cursor ? boxFromCorners(boxStart, cursor, image.width, image.height) : null
  const activeColor = classColor(p.classes, p.activeClass)
  const stageCursor = p.panning ? 'grab' : tool === 'select' ? 'default' : 'crosshair'

  return (
    <div ref={containerRef} className="canvas" style={{ cursor: stageCursor }}>
      {width > 0 && (
        <Stage
          ref={stageRef}
          width={width}
          height={height}
          scaleX={view.scale}
          scaleY={view.scale}
          x={view.x}
          y={view.y}
          draggable={tool === 'select' || p.panning}
          onDragMove={onStageDrag}
          onDragEnd={onStageDrag}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseLeave={() => setCursor(null)}
          onClick={onClick}
          onContextMenu={onContextMenu}
          onWheel={onWheel}
        >
          <Layer>
            <KonvaImage image={image} name="background" />
          </Layer>
          <Layer>
            {p.shapes.map((s) => {
              const color = classColor(p.classes, s.className)
              const highlight = s.id === p.hoveredId || s.id === p.selectedId
              return s.kind === 'box'
                ? renderBox(s, color, highlight)
                : renderPolygon(s, color, highlight)
            })}

            {/* class labels above each shape */}
            {p.shapes.map((s) => {
              const [x, y] = s.kind === 'box' ? s.box : (s.points[0] ?? [0, 0])
              return (
                <Text
                  key={`label-${s.id}`}
                  x={x}
                  y={y - px(15)}
                  text={s.score != null ? `${s.className} ${s.score.toFixed(2)}` : s.className}
                  fontSize={px(12)}
                  fill={classColor(p.classes, s.className)}
                  listening={false}
                />
              )
            })}

            {/* vertex handles of the selected polygon: drag to reshape */}
            {selectable &&
              selected?.kind === 'polygon' &&
              selected.points.map(([x, y], i) => (
                <Circle
                  key={`${selected.id}-${i}`}
                  x={x}
                  y={y}
                  radius={px(5)}
                  fill="white"
                  stroke={classColor(p.classes, selected.className)}
                  strokeWidth={2}
                  strokeScaleEnabled={false}
                  draggable
                  onDragStart={p.onEditStart}
                  onDragMove={(e) => {
                    const [nx, ny] = clampToImage([e.target.x(), e.target.y()])
                    e.target.position({ x: nx, y: ny })
                    const points = selected.points.map((pt, j): Point =>
                      j === i ? [round(nx), round(ny)] : pt,
                    )
                    p.onEdit({ ...selected, points })
                  }}
                />
              ))}

            {/* shape being drawn */}
            {draftBox && (
              <Rect
                x={draftBox[0]}
                y={draftBox[1]}
                width={draftBox[2] - draftBox[0]}
                height={draftBox[3] - draftBox[1]}
                stroke={activeColor}
                strokeWidth={2}
                strokeScaleEnabled={false}
                dash={[6, 4]}
                listening={false}
              />
            )}
            {tool === 'polygon' && p.draft.length > 0 && (
              <>
                <Line
                  points={[...p.draft.flat(), ...(cursor ?? [])]}
                  stroke={activeColor}
                  strokeWidth={2}
                  strokeScaleEnabled={false}
                  dash={[6, 4]}
                  listening={false}
                />
                {p.draft.map(([x, y], i) => (
                  <Circle
                    key={i}
                    x={x}
                    y={y}
                    radius={px(i === 0 && p.draft.length >= 3 ? 7 : 4)}
                    fill={i === 0 ? activeColor : 'white'}
                    stroke={activeColor}
                    strokeWidth={1}
                    strokeScaleEnabled={false}
                    listening={false}
                  />
                ))}
              </>
            )}

            {/* magic click: SAM's proposed outline and the clicks that produced it */}
            {tool === 'magic' && p.magicPreview && (
              <Line
                points={p.magicPreview.flat()}
                closed
                stroke={activeColor}
                strokeWidth={2}
                strokeScaleEnabled={false}
                dash={[8, 5]}
                fill={withAlpha(activeColor, 0.3)}
                listening={false}
              />
            )}
            {tool === 'magic' &&
              p.magicPoints.map(({ point: [x, y], label }, i) => (
                <Circle
                  key={`magic-${i}`}
                  x={x}
                  y={y}
                  radius={px(6)}
                  fill={label === 1 ? '#1a9f4b' : '#d93636'}
                  stroke="white"
                  strokeWidth={2}
                  strokeScaleEnabled={false}
                  listening={false}
                />
              ))}

            <Transformer
              ref={transformerRef}
              rotateEnabled={false}
              keepRatio={false}
              ignoreStroke
              flipEnabled={false}
              anchorSize={8}
            />
          </Layer>
        </Stage>
      )}
    </div>
  )
}
