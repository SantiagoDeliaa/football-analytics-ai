import type { ComputerVisionResult, TeamMetricRange } from '../../types/computerVision'

function formatMetric(metric?: TeamMetricRange, decimals = 1) {
  if (!metric || metric.mean === null || metric.mean === undefined) {
    return 'No aplica'
  }

  return metric.mean.toFixed(decimals)
}

export function TeamComparisonTable({ result }: { result: ComputerVisionResult }) {
  const rows = [
    {
      label: 'Presión (m)',
      team1: formatMetric(result.metrics.team1.pressure_height),
      team2: formatMetric(result.metrics.team2.pressure_height),
    },
    {
      label: 'Amplitud (m)',
      team1: formatMetric(result.metrics.team1.offensive_width),
      team2: formatMetric(result.metrics.team2.offensive_width),
    },
    {
      label: 'Compactación (m²)',
      team1: formatMetric(result.metrics.team1.compactness, 0),
      team2: formatMetric(result.metrics.team2.compactness, 0),
    },
  ]

  return (
    <section className="space-y-3 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <header>
        <h3 className="text-lg font-semibold text-slate-100">Comparativa táctica</h3>
        <p className="mt-2 text-sm text-slate-300">
          Resumen de métricas principales entre ambos equipos, alineado a la tabla del Streamlit original.
        </p>
      </header>

      <div className="overflow-hidden rounded-xl border border-slate-800">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-slate-950/70 text-slate-300">
            <tr>
              <th className="px-4 py-3 text-left font-semibold">Métrica</th>
              <th className="px-4 py-3 text-left font-semibold">Team 1</th>
              <th className="px-4 py-3 text-left font-semibold">Team 2</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 bg-slate-950/40 text-slate-100">
            {rows.map((row) => (
              <tr key={row.label}>
                <td className="px-4 py-3">{row.label}</td>
                <td className="px-4 py-3">{row.team1}</td>
                <td className="px-4 py-3">{row.team2}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FormationCard formation={result.formations.team1.most_common} team="Team 1" />
        <FormationCard formation={result.formations.team2.most_common} team="Team 2" />
      </div>
    </section>
  )
}

function FormationCard({ team, formation }: { team: string; formation: string }) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{team}</p>
      <p className="mt-2 text-lg font-semibold text-slate-100">{formation}</p>
      <p className="mt-2 text-xs text-slate-300">Formación más común detectada en el clip.</p>
    </article>
  )
}
