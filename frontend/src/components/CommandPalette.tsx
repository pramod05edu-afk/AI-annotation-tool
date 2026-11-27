import { FolderOpen, LayoutGrid, Plus, Search } from 'lucide-react'
import { type KeyboardEvent, useCallback, useState } from 'react'
import { useNavigate } from 'react-router'

import { api, unwrap } from '../api/client'
import { useAsync } from '../hooks'

interface Command {
  id: string
  label: string
  hint: string
  icon: typeof Search
  run: () => void
}

/** Ctrl+K: type to jump to a project or run an action, like in modern editors. */
export default function CommandPalette({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const load = useCallback(async () => unwrap(await api.GET('/api/projects')), [])
  const { data: projects } = useAsync(load)

  const go = (path: string) => () => {
    onClose()
    void navigate(path)
  }
  const commands: Command[] = [
    { id: 'home', label: 'All projects', hint: 'Dashboard', icon: LayoutGrid, run: go('/') },
    { id: 'new', label: 'New project', hint: 'Create', icon: Plus, run: go('/?new=1') },
    ...(projects ?? []).map((p) => ({
      id: `p${p.id}`,
      label: p.name,
      hint: `${p.status_counts?.done ?? 0} / ${p.image_count} done`,
      icon: FolderOpen,
      run: go(`/projects/${p.id}`),
    })),
  ]
  const shown = commands.filter((c) => c.label.toLowerCase().includes(query.trim().toLowerCase()))
  const selected = Math.min(active, Math.max(shown.length - 1, 0))

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((selected + 1) % Math.max(shown.length, 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((selected - 1 + shown.length) % Math.max(shown.length, 1))
    } else if (e.key === 'Enter') {
      shown[selected]?.run()
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  return (
    <div className="modal-backdrop palette-backdrop" onMouseDown={onClose}>
      <div
        className="palette"
        role="dialog"
        aria-label="Command palette"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="palette-input">
          <Search size={18} />
          <input
            autoFocus
            value={query}
            placeholder="Search projects and actions…"
            onChange={(e) => {
              setQuery(e.target.value)
              setActive(0)
            }}
            onKeyDown={onKeyDown}
          />
          <kbd>Esc</kbd>
        </div>
        <ul className="palette-list">
          {shown.map((c, i) => (
            <li key={c.id}>
              <button
                type="button"
                className={i === selected ? 'active' : ''}
                onMouseEnter={() => setActive(i)}
                onClick={c.run}
              >
                <c.icon size={16} />
                <span>{c.label}</span>
                <small>{c.hint}</small>
              </button>
            </li>
          ))}
          {shown.length === 0 && <li className="palette-empty">Nothing found</li>}
        </ul>
      </div>
    </div>
  )
}
