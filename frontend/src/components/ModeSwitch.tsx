import { Lock } from 'lucide-react'

import type { Mode } from '../annotator/interactionLog'

interface ModeSwitchProps {
  mode: Mode
  /** The project fixes the mode (effort study): show it, but it cannot be changed. */
  locked: boolean
  onChange: (mode: Mode) => void
}

const LABEL: Record<Mode, string> = { manual: 'Manual', assisted: 'Assisted' }

/** Manual / Assisted switch; a lock badge when the project prescribes the mode. */
export default function ModeSwitch({ mode, locked, onChange }: ModeSwitchProps) {
  if (locked) {
    return (
      <span className={`mode-lock ${mode}`} title="This project prescribes the annotation mode">
        <Lock size={11} /> {LABEL[mode]} mode
      </span>
    )
  }
  return (
    <div className="segmented" role="group" aria-label="Annotation mode">
      {(['manual', 'assisted'] as const).map((m) => (
        <button
          key={m}
          type="button"
          className={mode === m ? 'active' : ''}
          onClick={() => onChange(m)}
          title={m === 'manual' ? 'Label by hand only' : 'AI suggestions, smart click and magic'}
        >
          {LABEL[m]}
        </button>
      ))}
    </div>
  )
}
