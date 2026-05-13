import type { EventHistoryEntry } from '../../types/eventData'
import { formatDateTime } from '../../utils/formatters'

interface EventHistoryPanelProps {
  entries: EventHistoryEntry[]
  activeMatchId?: string
  onLoad: (entry: EventHistoryEntry) => void
}

export function EventHistoryPanel({ entries, activeMatchId, onLoad }: EventHistoryPanelProps) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Historial local</h3>
          <p className="mt-1 text-sm text-slate-300">
            Guarda los últimos partidos procesados para recuperarlos rápido en modo demo.
          </p>
        </div>
        <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-emerald-200">
          {entries.length} guardados
        </span>
      </div>

      {!entries.length ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
          Todavía no hay partidos guardados localmente.
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {entries.map((entry) => {
            const isActive = entry.result.match_id === activeMatchId
            const buttonLabel = isActive ? 'Partido activo' : 'Cargar historial'
            const accessibleButtonLabel = isActive
              ? `${entry.match_label} activo`
              : `Cargar ${entry.match_label} desde historial local`

            return (
              <article
                className={`rounded-xl border p-4 transition ${
                  isActive
                    ? 'border-emerald-400/50 bg-emerald-500/10'
                    : 'border-slate-700 bg-slate-950/70'
                }`}
                key={entry.id}
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-slate-100">{entry.match_label}</p>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      {entry.competition_name} · {entry.provider}
                    </p>
                    <p className="text-xs text-slate-400">
                      Guardado: {formatDateTime(entry.saved_at)} · Equipo: {entry.selection.team} · Jugador:{' '}
                      {entry.selection.player}
                    </p>
                  </div>

                  <button
                    aria-label={accessibleButtonLabel}
                    className="rounded-md border border-slate-600 px-3 py-2 text-sm font-semibold text-slate-100 hover:border-emerald-400 hover:bg-slate-800"
                    onClick={() => onLoad(entry)}
                    type="button"
                  >
                    {buttonLabel}
                  </button>
                </div>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
