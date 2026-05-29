import type { MatchCenterExpectedMetrics } from '../../../types/matchCenter'
import { formatMetricValue } from '../../../utils/matchCenter'

interface ExpectedMetricsPanelProps {
  expectedMetrics?: MatchCenterExpectedMetrics | null
}

const metricRows = [
  { key: 'xg', label: 'Goles esperados' },
  { key: 'xgot', label: 'Goles esperados al arco' },
  { key: 'xpts', label: 'Puntos esperados' },
  { key: 'npxg', label: 'xG sin penales' },
  { key: 'xg_open_play', label: 'xG en jugada' },
  { key: 'xg_set_play', label: 'xG pelota parada' },
  { key: 'xg_free_kicks', label: 'xG de tiros libres' },
  { key: 'xga', label: 'xG concedido' },
] as const

export function ExpectedMetricsPanel({ expectedMetrics }: ExpectedMetricsPanelProps) {
  const home = expectedMetrics?.home
  const away = expectedMetrics?.away
  const hasMetrics = metricRows.some(
    (row) => home?.[row.key] !== null || home?.[row.key] !== undefined || away?.[row.key] !== null || away?.[row.key] !== undefined,
  )

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">Rendimiento esperado</h2>
          <p className="text-sm text-slate-300">
            Comparativa entre local y visitante según el contrato normalizado del backend.
          </p>
        </div>
      </div>

      {!hasMetrics ? (
        <p className="text-sm text-slate-300">No hay métricas esperadas disponibles para este partido.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-slate-300">
                <th className="px-3 py-2">Métrica</th>
                <th className="px-3 py-2">{home?.team_name || 'Local'}</th>
                <th className="px-3 py-2">{away?.team_name || 'Visitante'}</th>
              </tr>
            </thead>
            <tbody>
              {metricRows.map((row) => (
                <tr className="border-b border-slate-800/80" key={row.key}>
                  <td className="px-3 py-2 text-slate-200">{row.label}</td>
                  <td className="px-3 py-2 text-slate-100">{formatMetricValue(home?.[row.key])}</td>
                  <td className="px-3 py-2 text-slate-100">{formatMetricValue(away?.[row.key])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
