import type { ComputerVisionResult, TeamMetricRange } from '../../types/computerVision'

export function ScoutingPanels({ result }: { result: ComputerVisionResult }) {
  const warnings = result.warnings.length ? result.warnings : []
  const heatmapBins = result.scouting_heatmaps?.bins_shape?.join(' x ')

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div>
        <h3 className="text-lg font-semibold text-slate-100">Scouting</h3>
        <p className="mt-2 text-sm text-slate-300">
          Resume confianza, estructura, métricas tácticas y señal operativa del pipeline para lectura rápida.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ScoutingTeamCard
          bullets={result.scouting.bullets.team1}
          confidence={result.scouting.confidence.team1}
          formation={result.formations.team1.most_common}
          metrics={result.metrics.team1}
          team="Team 1"
        />
        <ScoutingTeamCard
          bullets={result.scouting.bullets.team2}
          confidence={result.scouting.confidence.team2}
          formation={result.formations.team2.most_common}
          metrics={result.metrics.team2}
          team="Team 2"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <SummaryCard
          label="Cobertura heatmap"
          value={heatmapBins ?? 'No disponible'}
          subtitle={
            result.scouting_heatmaps
              ? `Muestras Team 1: ${result.scouting_heatmaps.team1?.total_samples ?? 0} · Team 2: ${result.scouting_heatmaps.team2?.total_samples ?? 0}`
              : 'El backend no devolvió metadata de heatmaps.'
          }
        />
        <SummaryCard
          label="Homografía"
          value={`${result.health_summary.valid_frames ?? 0} frames válidos`}
          subtitle={`Fallback ratio: ${formatPercent(result.health_summary.fallback_ratio)}`}
        />
        <SummaryCard
          label="Warnings"
          value={`${warnings.length}`}
          subtitle={warnings[0] ?? 'Sin alertas críticas en esta corrida.'}
        />
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
        <h4 className="text-sm font-semibold text-slate-100">Alertas y señal operativa</h4>
        {warnings.length ? (
          <ul className="mt-3 space-y-2">
            {warnings.map((warning) => (
              <li className="text-sm text-slate-200" key={warning}>
                - {warning}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-slate-400">No hay alertas adicionales para este recorte.</p>
        )}
      </section>
    </section>
  )
}

export function PossessionPanel({ result }: { result: ComputerVisionResult }) {
  if (!result.possession) {
    return (
      <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Posesión</h3>
        <p className="mt-2 text-sm text-slate-300">
          La posesión no está disponible para este resultado o fue desactivada en la corrida.
        </p>
      </section>
    )
  }

  const bars = [
    { label: 'Team 1', value: result.possession.team1_pct, color: 'bg-emerald-400' },
    { label: 'Team 2', value: result.possession.team2_pct, color: 'bg-sky-400' },
    { label: 'Contestado', value: result.possession.contested_pct, color: 'bg-amber-400' },
  ]
  const passSummary = result.possession.passes
  const speedByTeam = result.speed_distance?.per_team
    ? Object.entries(result.speed_distance.per_team)
    : []
  const topPossessors = result.possession.top_possessors ?? []
  const playerDistanceRows = result.speed_distance?.per_player
    ? Object.entries(result.speed_distance.per_player)
        .sort(([, left], [, right]) => right.distance_m - left.distance_m)
        .slice(0, 6)
    : []

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div>
        <h3 className="text-lg font-semibold text-slate-100">Posesión y Físico</h3>
        <p className="mt-2 text-sm text-slate-300">
          Combina reparto de posesión, pases/turnovers y resumen físico por equipo/jugador.
        </p>
      </div>

      <div className="space-y-3">
        {bars.map((bar) => (
          <div key={bar.label}>
            <div className="mb-1 flex items-center justify-between text-sm text-slate-200">
              <span>{bar.label}</span>
              <span>{bar.value}%</span>
            </div>
            <div className="h-3 rounded-full bg-slate-950/80">
              <div className={`h-3 rounded-full ${bar.color}`} style={{ width: `${bar.value}%` }} />
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <SummaryCard
          label="Frames analizados"
          value={`${result.possession.total_frames_analyzed ?? 0}`}
          subtitle={result.possession.reason ?? 'Sin razón de degradación informada.'}
        />
        <SummaryCard
          label="Pases Team 1"
          value={`${passSummary?.team1_passes ?? 0}`}
          subtitle={`Pases Team 2: ${passSummary?.team2_passes ?? 0}`}
        />
        <SummaryCard
          label="Turnovers"
          value={`${passSummary?.turnovers ?? 0}`}
          subtitle={`Eventos de pase detectados: ${passSummary?.total ?? 0}`}
        />
        <SummaryCard
          label="Top poseedores"
          value={`${topPossessors.length}`}
          subtitle="Trackers con más frames de control de balón."
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <section className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
          <h4 className="text-sm font-semibold text-slate-100">Resumen físico por equipo</h4>
          {!speedByTeam.length ? (
            <p className="mt-3 text-sm text-slate-400">No hay resumen físico disponible para esta corrida.</p>
          ) : (
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                    <th className="px-3 py-2 font-semibold">Equipo</th>
                    <th className="px-3 py-2 font-semibold">Distancia</th>
                    <th className="px-3 py-2 font-semibold">Vel. max</th>
                    <th className="px-3 py-2 font-semibold">Sprints</th>
                  </tr>
                </thead>
                <tbody>
                  {speedByTeam.map(([team, stats]) => (
                    <tr className="border-b border-slate-900/80" key={team}>
                      <td className="px-3 py-2 text-slate-200">{team}</td>
                      <td className="px-3 py-2 text-slate-200">{stats.total_distance_m} m</td>
                      <td className="px-3 py-2 text-slate-200">{stats.max_speed_kmh} km/h</td>
                      <td className="px-3 py-2 text-slate-200">{stats.total_sprints}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
          <h4 className="text-sm font-semibold text-slate-100">Top poseedores</h4>
          {!topPossessors.length ? (
            <p className="mt-3 text-sm text-slate-400">No hay poseedores destacados para este recorte.</p>
          ) : (
            <ul className="mt-3 space-y-2">
              {topPossessors.slice(0, 6).map((player) => (
                <li className="rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2 text-sm text-slate-200" key={`${player.team}-${player.tracker_id}`}>
                  Tracker #{player.tracker_id} · {player.team} · {player.frames} frames
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
        <h4 className="text-sm font-semibold text-slate-100">Top físico por jugador</h4>
        {!playerDistanceRows.length ? (
          <p className="mt-3 text-sm text-slate-400">No hay datos por jugador disponibles.</p>
        ) : (
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-2 font-semibold">Tracker</th>
                  <th className="px-3 py-2 font-semibold">Equipo</th>
                  <th className="px-3 py-2 font-semibold">Distancia</th>
                  <th className="px-3 py-2 font-semibold">Vel. max</th>
                  <th className="px-3 py-2 font-semibold">Sprints</th>
                </tr>
              </thead>
              <tbody>
                {playerDistanceRows.map(([trackerId, stats]) => (
                  <tr className="border-b border-slate-900/80" key={trackerId}>
                    <td className="px-3 py-2 text-slate-200">#{trackerId}</td>
                    <td className="px-3 py-2 text-slate-200">{stats.team}</td>
                    <td className="px-3 py-2 text-slate-200">{stats.distance_m} m</td>
                    <td className="px-3 py-2 text-slate-200">{stats.max_speed_kmh} km/h</td>
                    <td className="px-3 py-2 text-slate-200">{stats.sprint_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  )
}

export function InterpretationPanel({ result }: { result: ComputerVisionResult }) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <h3 className="text-lg font-semibold text-slate-100">Interpretación</h3>
      <ul className="mt-4 space-y-2">
        {result.interpretation.map((item) => (
          <li className="text-sm text-slate-200" key={item}>
            - {item}
          </li>
        ))}
      </ul>
    </section>
  )
}

function ScoutingTeamCard({
  team,
  confidence,
  bullets,
  formation,
  metrics,
}: {
  team: string
  confidence: { label: string; score: number }
  bullets: string[]
  formation: string
  metrics: ComputerVisionResult['metrics']['team1']
}) {
  const metricRows = [
    { label: 'Altura presión', metric: metrics.pressure_height, suffix: 'm' },
    { label: 'Amplitud ofensiva', metric: metrics.offensive_width, suffix: 'm' },
    { label: 'Compactación', metric: metrics.compactness, suffix: 'm²' },
    { label: 'Profundidad bloque', metric: metrics.block_depth_m, suffix: 'm' },
    { label: 'Ancho bloque', metric: metrics.block_width_m, suffix: 'm' },
  ]

  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{team}</p>
          <p className="mt-2 text-sm font-semibold text-slate-100">
            Confianza {confidence.label} ({confidence.score}/100)
          </p>
          <p className="mt-1 text-sm text-slate-300">Formación detectada: {formation}</p>
        </div>
      </div>

      <ul className="mt-4 space-y-2">
        {bullets.map((item) => (
          <li className="text-sm text-slate-200" key={item}>
            - {item}
          </li>
        ))}
      </ul>

      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        {metricRows.map((row) => (
          <MetricStatCard key={row.label} label={row.label} metric={row.metric} suffix={row.suffix} />
        ))}
      </div>
    </article>
  )
}

function SummaryCard({ label, value, subtitle }: { label: string; value: string; subtitle: string }) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-xl font-semibold text-slate-100">{value}</p>
      <p className="mt-2 text-xs text-slate-400">{subtitle}</p>
    </article>
  )
}

function MetricStatCard({
  label,
  metric,
  suffix,
}: {
  label: string
  metric?: TeamMetricRange
  suffix: string
}) {
  return (
    <article className="rounded-lg border border-slate-800 bg-slate-900/80 p-3">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-sm font-semibold text-slate-100">{formatMetricRange(metric, suffix)}</p>
    </article>
  )
}

function formatMetricRange(metric: TeamMetricRange | undefined, suffix: string) {
  if (!metric || metric.mean == null) {
    return 'Sin datos'
  }

  const mean = `${metric.mean}${suffix}`
  const min = metric.min != null ? metric.min : 'N/D'
  const max = metric.max != null ? metric.max : 'N/D'
  return `Prom ${mean} · Min ${min}${suffix} · Max ${max}${suffix}`
}

function formatPercent(value: number | undefined) {
  if (value == null) {
    return 'N/D'
  }
  return `${Math.round(value * 100)}%`
}
