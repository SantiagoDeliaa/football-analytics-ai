import { useState } from 'react'
import type { ProcessedHistoryMatch, ProviderOption } from '../../types/eventData'
import { formatDateTime } from '../../utils/formatters'

interface ProcessedHistoryPanelProps {
  entries: ProcessedHistoryMatch[]
  loading: boolean
  activeMatchId?: string
  onLoad: (provider: ProviderOption, matchId: string) => void
  onDelete: (provider: ProviderOption, matchId: string) => void
  deletingKey?: string
  feedbackMessage?: string
}

export function ProcessedHistoryPanel({
  entries,
  loading,
  activeMatchId,
  onLoad,
  onDelete,
  deletingKey,
  feedbackMessage,
}: ProcessedHistoryPanelProps) {
  const [confirmingDeleteKey, setConfirmingDeleteKey] = useState<string>()

  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Historial local persistido</h3>
          <p className="mt-1 text-sm text-slate-300">
            Partidos guardados en SQLite + JSON para reutilizarlos entre sesiones y limpiar registros cuando ya no hagan falta.
          </p>
        </div>
        <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-sky-200">
          {entries.length} procesados
        </span>
      </div>

      {feedbackMessage ? (
        <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-100">
          {feedbackMessage}
        </div>
      ) : null}

      {loading ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
          Cargando historial persistido...
        </div>
      ) : !entries.length ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
          Todavía no hay partidos guardados en el historial persistido.
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {entries.map((entry) => {
            const isActive = entry.match_id === activeMatchId
            const matchLabel = [entry.home_team, entry.away_team].filter(Boolean).join(' vs ')
            const buttonLabel = isActive ? 'Partido activo' : 'Cargar partido'
            const accessibleButtonLabel = isActive
              ? `${matchLabel || `Partido ${entry.match_id}`} activo`
              : `Cargar ${matchLabel || `Partido ${entry.match_id}`} desde historial persistido`
            const entryKey = `${entry.provider}-${entry.match_id}`
            const isConfirmingDelete = confirmingDeleteKey === entryKey
            const isDeleting = deletingKey === entryKey

            return (
              <article
                className={`rounded-xl border p-4 transition ${
                  isActive
                    ? 'border-sky-400/50 bg-sky-500/10'
                    : 'border-slate-700 bg-slate-950/70'
                }`}
                key={`${entry.provider}-${entry.match_id}`}
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-slate-100">
                      {matchLabel || `Partido ${entry.match_id}`}
                    </p>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      {entry.competition_name} · {entry.season_name || 'Sin temporada'} · {entry.provider}
                    </p>
                    <p className="text-xs text-slate-400">
                      Actualizado: {formatDateTime(entry.updated_at)} · Fecha partido: {entry.match_date || 'N/D'}
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <button
                      aria-label={accessibleButtonLabel}
                      className="rounded-md border border-slate-600 px-3 py-2 text-sm font-semibold text-slate-100 hover:border-sky-400 hover:bg-slate-800"
                      onClick={() => onLoad(entry.provider, entry.match_id)}
                      type="button"
                    >
                      {buttonLabel}
                    </button>
                    {!isConfirmingDelete ? (
                      <button
                        className="rounded-md border border-rose-500/40 px-3 py-2 text-sm font-semibold text-rose-100 hover:bg-rose-500/10"
                        onClick={() => setConfirmingDeleteKey(entryKey)}
                        type="button"
                      >
                        Eliminar
                      </button>
                    ) : (
                      <>
                        <button
                          className="rounded-md border border-slate-600 px-3 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800"
                          onClick={() => setConfirmingDeleteKey(undefined)}
                          type="button"
                        >
                          Cancelar
                        </button>
                        <button
                          className="rounded-md border border-rose-500/50 bg-rose-500/10 px-3 py-2 text-sm font-semibold text-rose-100 hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                          disabled={isDeleting}
                          onClick={() => {
                            onDelete(entry.provider, entry.match_id)
                            setConfirmingDeleteKey(undefined)
                          }}
                          type="button"
                        >
                          {isDeleting ? 'Eliminando...' : 'Eliminar definitivamente'}
                        </button>
                      </>
                    )}
                  </div>
                </div>
                {isConfirmingDelete ? (
                  <p className="mt-3 text-sm text-rose-100">
                    Esta acción elimina la metadata del partido y los JSON asociados si existen.
                  </p>
                ) : null}
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
