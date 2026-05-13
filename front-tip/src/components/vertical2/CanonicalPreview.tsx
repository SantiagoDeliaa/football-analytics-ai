import type { CanonicalEvent } from '../../types/eventData'

export function CanonicalPreview({ events }: { events: CanonicalEvent[] }) {
  const preview = JSON.stringify(events.slice(0, 10), null, 2)

  return (
    <details className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <summary className="cursor-pointer text-sm font-semibold text-slate-200">
        Ver preview del Canonical Event Model (JSON)
      </summary>
      <pre className="mt-3 max-h-64 overflow-auto rounded bg-slate-950 p-3 text-xs text-emerald-200">
        {preview}
      </pre>
    </details>
  )
}
