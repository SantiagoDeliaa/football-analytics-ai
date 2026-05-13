import type { EventHistoryEntry, ProcessedHistoryMatch, ProviderOption } from '../../types/eventData'
import { formatDateTime } from '../../utils/formatters'

interface HistoryMatchHintProps {
  localEntry?: EventHistoryEntry
  backendEntry?: ProcessedHistoryMatch
  onLoadLocal: (entry: EventHistoryEntry) => void
  onLoadBackend: (provider: ProviderOption, matchId: string) => void
}

export function HistoryMatchHint({
  localEntry,
  backendEntry,
  onLoadLocal,
  onLoadBackend,
}: HistoryMatchHintProps) {
  if (!localEntry && !backendEntry) {
    return null
  }

  return (
    <section className="rounded-2xl border border-amber-700/40 bg-amber-900/20 p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h3 className="text-base font-semibold text-amber-50">Partido ya disponible en historial</h3>
          <p className="mt-1 text-sm text-amber-100">
            Podés reutilizar una versión ya procesada para acelerar la demo y evitar una nueva carga.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {localEntry ? (
            <span className="rounded-full border border-amber-300/30 bg-amber-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-amber-100">
              Local · {formatDateTime(localEntry.saved_at)}
            </span>
          ) : null}
          {backendEntry ? (
            <span className="rounded-full border border-sky-300/30 bg-sky-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-sky-100">
              Backend · {formatDateTime(backendEntry.updated_at)}
            </span>
          ) : null}
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 md:flex-row">
        {localEntry ? (
          <button
            className="rounded-md border border-amber-300/40 px-4 py-2 text-sm font-semibold text-amber-50 hover:bg-amber-800/30"
            onClick={() => onLoadLocal(localEntry)}
            type="button"
          >
            Cargar historial local
          </button>
        ) : null}
        {backendEntry ? (
          <button
            className="rounded-md border border-sky-300/40 px-4 py-2 text-sm font-semibold text-sky-50 hover:bg-sky-800/30"
            onClick={() => onLoadBackend(backendEntry.provider, backendEntry.match_id)}
            type="button"
          >
            Cargar historial backend
          </button>
        ) : null}
      </div>
    </section>
  )
}
