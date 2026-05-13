import { useMemo } from 'react'
import type { ComputerVisionResult } from '../../types/computerVision'

export function ExportsPanel({ result }: { result: ComputerVisionResult }) {
  const jsonHref = useMemo(
    () =>
      `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`,
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
          El JSON y CSV quedan disponibles en demo. El PDF depende del backend analítico.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <DownloadCard available={result.exports.json} href={jsonHref} label="Descargar JSON" />
        <DownloadCard available={result.exports.csv} href={csvHref} label="Descargar CSV" />
        <DownloadCard available={result.exports.pdf} label="Exportar PDF" />
      </div>
    </section>
  )
}

function DownloadCard({
  label,
  available,
  href,
}: {
  label: string
  available: boolean
  href?: string
}) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-sm font-semibold text-slate-100">{label}</p>
      <p className="mt-2 text-xs text-slate-400">
        {available ? 'Disponible para este resultado.' : 'Requiere soporte del backend.'}
      </p>
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
