import type { Project } from '../api/client'

/** Green = done, amber = in progress, grey = still to do. */
export default function ProgressBar({ project }: { project: Project }) {
  const total = project.image_count
  const done = project.status_counts?.done ?? 0
  const inProgress = project.status_counts?.in_progress ?? 0
  if (total === 0) return <p className="muted">No images yet</p>
  return (
    <div>
      <div className="progress" aria-label={`${done} of ${total} done`}>
        <div className="progress-done" style={{ width: `${(100 * done) / total}%` }} />
        <div className="progress-partial" style={{ width: `${(100 * inProgress) / total}%` }} />
      </div>
      <p className="muted small">
        {done} / {total} done{inProgress > 0 && ` · ${inProgress} in progress`}
      </p>
    </div>
  )
}
