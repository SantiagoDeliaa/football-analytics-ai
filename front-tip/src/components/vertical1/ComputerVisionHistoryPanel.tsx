import { useState } from 'react'
import type { ComputerVisionHistoryItem } from '../../types/computerVision'
import { formatDateTime } from '../../utils/formatters'

interface ComputerVisionHistoryPanelProps {
  entries: ComputerVisionHistoryItem[]
  loading: boolean
  loadingProcessingId?: string
  deletingProcessingId?: string
  activeProcessingId?: string
  feedbackMessage?: string
  onLoad: (processingId: string) => void
  onDelete: (processingId: string) => void
}

export function ComputerVisionHistoryPanel({
  entries,
  loading,
  loadingProcessingId,
  deletingProcessingId,
  activeProcessingId,
  feedbackMessage,
  onLoad,
  onDelete,
}: ComputerVisionHistoryPanelProps) {
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string>()

  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-slate-100">Historial de procesamientos</h3>
          <p className="mt-1 text-sm text-slate-300">
            Reutilizá resultados guardados en SQLite + JSON sidecar sin reprocesar el video.
          </p>
        </div>
        <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-sky-200">
          {entries.length} guardados
        </span>
      </div>

      {feedbackMessage ? (
        <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-100">
          {feedbackMessage}
        </div>
      ) : null}

      {loading ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
          Cargando historial de procesamientos...
        </div>
      ) : !entries.length ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
          Todavía no hay procesamientos guardados.
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {entries.map((entry) => {
            const isActive = entry.processing_id === activeProcessingId
            const isLoading = entry.processing_id === loadingProcessingId
            const isDeleting = entry.processing_id === deletingProcessingId
            const isConfirmingDelete = entry.processing_id === confirmingDeleteId

            return (
              <article
                className={`rounded-xl border p-4 transition ${
                  isActive ? 'border-sky-400/50 bg-sky-500/10' : 'border-slate-700 bg-slate-950/70'
                }`}
                key={entry.processing_id}
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-slate-100">{entry.video_name || 'Video sin nombre'}</p>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      {entry.source_mode} · {entry.status}
                    </p>
                    <p className="text-xs text-slate-400">
                      Origen: {entry.source_label || 'No disponible'} · Actualizado: {formatDateTime(entry.updated_at)}
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <button
                      className="rounded-md border border-slate-600 px-3 py-2 text-sm font-semibold text-slate-100 hover:border-sky-400 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                      disabled={isLoading}
                      onClick={() => onLoad(entry.processing_id)}
                      type="button"
                    >
                      {isLoading ? 'Cargando...' : isActive ? 'Resultado activo' : 'Cargar resultado'}
                    </button>
                    {!isConfirmingDelete ? (
                      <button
                        className="rounded-md border border-rose-500/40 px-3 py-2 text-sm font-semibold text-rose-100 hover:bg-rose-500/10"
                        onClick={() => setConfirmingDeleteId(entry.processing_id)}
                        type="button"
                      >
                        Eliminar
                      </button>
                    ) : (
                      <>
                        <button
                          className="rounded-md border border-slate-600 px-3 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800"
                          onClick={() => setConfirmingDeleteId(undefined)}
                          type="button"
                        >
                          Cancelar
                        </button>
                        <button
                          className="rounded-md border border-rose-500/50 bg-rose-500/10 px-3 py-2 text-sm font-semibold text-rose-100 hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                          disabled={isDeleting}
                          onClick={() => {
                            onDelete(entry.processing_id)
                            setConfirmingDeleteId(undefined)
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
                    Esta acción elimina la metadata guardada y los archivos asociados si todavía existen.
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
