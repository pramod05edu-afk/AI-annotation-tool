import { ArrowDown, ArrowUp, Plus, X } from 'lucide-react'
import { type FormEvent, useState } from 'react'

import { api, errorMessage, type Project, unwrap } from '../api/client'
import { classColor } from '../annotator/shapes'

interface Row {
  key: number // stable React key while rows are edited and reordered
  original: string | null // the saved name; null = added in this dialog
  name: string
}

let nextKey = 0
const row = (original: string | null, name: string): Row => ({ key: nextKey++, original, name })

/**
 * Manage a project's labels: add, rename, reorder (order = keyboard shortcuts 1-9), remove.
 * A rename keeps the existing boxes (the backend moves them to the new name); a label that
 * boxes still use cannot be removed, the backend refuses and the message is shown here.
 */
export default function EditLabels({
  project,
  onSaved,
}: {
  project: Project
  onSaved: (project: Project) => void
}) {
  const [rows, setRows] = useState(() => project.classes.map((c) => row(c, c)))
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string>()
  const [busy, setBusy] = useState(false)

  const names = rows.map((r) => r.name.trim())
  const duplicate = names.find((n, i) => n && names.indexOf(n) !== i)
  const problem = names.some((n) => !n)
    ? 'Labels cannot be empty'
    : duplicate
      ? `"${duplicate}" is in the list twice`
      : rows.length === 0
        ? 'A project needs at least one label'
        : undefined
  // Labels typed into the "add" field but not added yet: saved too, so nothing is lost.
  const pending = [
    ...new Set(
      draft
        .split(',')
        .map((n) => n.trim())
        .filter((n) => n && !names.includes(n)),
    ),
  ]
  const changed =
    names.join('\n') !== project.classes.join('\n') ||
    rows.some((r) => r.original === null) ||
    pending.length > 0

  function update(key: number, name: string) {
    setRows((all) => all.map((r) => (r.key === key ? { ...r, name } : r)))
  }

  function move(index: number, by: -1 | 1) {
    setRows((all) => {
      const next = [...all]
      const [item] = next.splice(index, 1)
      if (item) next.splice(index + by, 0, item)
      return next
    })
  }

  function add(event?: FormEvent) {
    event?.preventDefault()
    if (pending.length > 0) setRows((all) => [...all, ...pending.map((n) => row(null, n))])
    setDraft('')
  }

  async function save() {
    setBusy(true)
    setError(undefined)
    const renames = Object.fromEntries(
      rows
        .filter((r) => r.original !== null && r.original !== r.name.trim())
        .map((r) => [r.original, r.name.trim()]),
    )
    try {
      const saved = unwrap(
        await api.PATCH('/api/projects/{project_id}', {
          params: { path: { project_id: project.id } },
          body: { classes: [...names, ...pending], renames },
        }),
      )
      onSaved(saved)
    } catch (e) {
      setError(errorMessage(e))
      setBusy(false)
    }
  }

  return (
    <div className="label-editor">
      <ol>
        {rows.map((r, i) => (
          <li key={r.key}>
            <span className="swatch" style={{ background: classColor(names, names[i] ?? '') }} />
            <kbd>{i < 9 ? i + 1 : '–'}</kbd>
            <input
              value={r.name}
              onChange={(e) => update(r.key, e.target.value)}
              aria-label={`Label ${i + 1}`}
            />
            {r.original === null ? (
              <span className="label-note new">new</span>
            ) : r.original !== r.name.trim() ? (
              <span
                className="label-note"
                title={`Boxes labelled "${r.original}" keep their place`}
              >
                was {r.original}
              </span>
            ) : (
              <span className="label-note" />
            )}
            <button
              type="button"
              className="icon"
              onClick={() => move(i, -1)}
              disabled={i === 0}
              aria-label="Move up"
            >
              <ArrowUp size={14} />
            </button>
            <button
              type="button"
              className="icon"
              onClick={() => move(i, 1)}
              disabled={i === rows.length - 1}
              aria-label="Move down"
            >
              <ArrowDown size={14} />
            </button>
            <button
              type="button"
              className="icon"
              onClick={() => setRows((all) => all.filter((x) => x.key !== r.key))}
              aria-label={`Remove ${r.name}`}
              title="Remove (only possible if no box uses it)"
            >
              <X size={14} />
            </button>
          </li>
        ))}
      </ol>

      <form className="label-add" onSubmit={add}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Add labels, e.g. airplane, boat"
          aria-label="New labels"
        />
        <button type="submit" disabled={!draft.trim()}>
          <Plus size={15} /> Add
        </button>
      </form>

      <p className="muted small">
        Renaming keeps the existing boxes. A label can only be removed when no box uses it. The AI
        looks for labels by name, so plain words work best.
      </p>
      {(error ?? problem) && <p className="error">{error ?? problem}</p>}
      <div className="label-actions">
        <button
          type="button"
          className="primary"
          disabled={busy || !changed || problem !== undefined}
          onClick={() => void save()}
        >
          {busy ? 'Saving…' : 'Save labels'}
        </button>
      </div>
    </div>
  )
}
