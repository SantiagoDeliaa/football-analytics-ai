import { useMemo } from 'react'
import type { ComputerVisionResult } from '../../types/computerVision'

export function ExportsPanel({ result }: { result: ComputerVisionResult }) {
  const jsonHref = useMemo(
    () =>
      result.artifacts?.stats_json_url
        ? result.artifacts.stats_json_url
        : `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`,
    [result],
  )

  const csvHref = useMemo(() => {
    const rows = [
      ['frame', 'pressure_team1', 'pressure_team2'],
      ...(result.timeline.pressure_height?.frames.map((frame, index) => [
        `${frame}`,
        `${result.timeline.pressure_height?.team1[index] ?? ''}`,
        `${result.timeline.pressure_height?.team2[index] ?? ''}`,
      ]) ?? []),
    ]

    const csv = rows.map((row) => row.join(',')).join('\n')
    return `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`
  }, [result])

  return (
    <section className="space-y-3 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <header>
        <h3 className="text-lg font-semibold text-slate-100">Exportes</h3>
        <p className="mt-2 text-sm text-slate-300">
          El frontend expone artefactos listos para demo y reutiliza archivos reales del backend cuando existen.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <DownloadCard
          available={Boolean(result.artifacts?.video_url)}
          href={result.artifacts?.video_url ?? undefined}
          label="Video procesado"
          note={
            result.artifacts?.video_url
              ? 'Descarga directa del archivo procesado por el backend.'
              : 'Disponible cuando la corrida viene del pipeline real.'
          }
        />
        <DownloadCard
          available={result.exports.json}
          href={jsonHref}
          label="Descargar JSON"
          note={
            result.artifacts?.stats_json_url
              ? 'Usa el `stats.json` real generado por el pipeline.'
              : 'Usa el resultado consolidado de la sesión actual.'
          }
        />
        <DownloadCard
          available={result.exports.csv}
          href={csvHref}
          label="Descargar CSV"
          note="Incluye la serie temporal disponible para comparar ambos equipos."
        />
        <DownloadCard
          available={result.exports.pdf && Boolean(result.artifacts?.pdf_url)}
          href={result.artifacts?.pdf_url ?? undefined}
          label="Exportar PDF"
          note="Sigue dependiendo de soporte backend específico de scouting."
        />
      </div>
    </section>
  )
}

function DownloadCard({
  label,
  available,
  href,
  note,
}: {
  label: string
  available: boolean
  href?: string
  note: string
}) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-sm font-semibold text-slate-100">{label}</p>
      <p className="mt-2 text-xs text-slate-400">{note}</p>
      {available && href ? (
        <a
          className="mt-4 inline-flex rounded-md border border-emerald-500 bg-emerald-700 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-600"
          download
          href={href}
        >
          Descargar
        </a>
      ) : (
        <button
          className="mt-4 rounded-md border border-slate-700 px-3 py-2 text-sm font-semibold text-slate-400"
          disabled
          type="button"
        >
          Próximamente
        </button>
      )}
    </article>
  )
}
