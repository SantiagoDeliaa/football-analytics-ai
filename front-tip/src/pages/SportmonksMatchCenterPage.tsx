import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { MetricCard } from '../components/common/MetricCard'
import { DataQualityBanner } from '../components/vertical2/match-center/DataQualityBanner'
import { ExpectedMetricsPanel } from '../components/vertical2/match-center/ExpectedMetricsPanel'
import { InsightsPanel } from '../components/vertical2/match-center/InsightsPanel'
import { LineupsPanel } from '../components/vertical2/match-center/LineupsPanel'
import { MatchHeader } from '../components/vertical2/match-center/MatchHeader'
import { MatchTimeline } from '../components/vertical2/match-center/MatchTimeline'
import { PlayerStatsTable } from '../components/vertical2/match-center/PlayerStatsTable'
import { useAsync } from '../hooks/useAsync'
import { getSportmonksMatchCenter } from '../services/matchCenterApi'
import {
  buildComparisonValue,
  formatMatchScore,
  formatMetricValue,
  getDataQualityLabel,
} from '../utils/matchCenter'

export function SportmonksMatchCenterPage() {
  const { matchId } = useParams<{ matchId: string }>()
  const { data, error, loading, run } = useAsync<Awaited<ReturnType<typeof getSportmonksMatchCenter>>>()

  useEffect(() => {
    if (!matchId) {
      return
    }
    void run(() => getSportmonksMatchCenter(matchId))
  }, [matchId, run])

  if (!matchId) {
    return <ErrorState message="Falta el identificador del partido para cargar el Match Center." />
  }

  if (loading) {
    return <LoadingState label="Cargando Match Center de Sportmonks..." />
  }

  if (error) {
    return (
      <ErrorState
        message={error}
        onRetry={() => {
          void run(() => getSportmonksMatchCenter(matchId))
        }}
      />
    )
  }

  const payload = data
  if (!payload) {
    return <ErrorState message="No se pudo cargar el Match Center del partido." />
  }

  const expectedMetrics = payload.expected_metrics
  const derivedMetrics = payload.derived_metrics
  const hasCoordinates = Boolean(payload.data_quality?.has_event_coordinates)
  const eventMapsEnabled = Boolean(payload.data_quality?.enabled_modules?.event_maps)

  return (
    <div className="space-y-6">
      <MatchHeader match={payload.match} />
      <DataQualityBanner dataQuality={payload.data_quality} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <MetricCard title="Resultado final" value={formatMatchScore(payload.match)} />
        <MetricCard
          title="Goles esperados"
          value={buildComparisonValue(expectedMetrics?.home?.xg, expectedMetrics?.away?.xg)}
          subtitle={`${payload.match.home_team.name || 'Local'} vs ${payload.match.away_team.name || 'Visitante'}`}
        />
        <MetricCard
          title="Goles esperados al arco"
          value={buildComparisonValue(expectedMetrics?.home?.xgot, expectedMetrics?.away?.xgot)}
        />
        <MetricCard
          title="Puntos esperados"
          value={buildComparisonValue(expectedMetrics?.home?.xpts, expectedMetrics?.away?.xpts)}
        />
        <MetricCard
          title="Eficacia ofensiva"
          value={buildComparisonValue(
            derivedMetrics?.home?.eficacia_ofensiva,
            derivedMetrics?.away?.eficacia_ofensiva,
          )}
        />
        <MetricCard
          title="Rendimiento de definición"
          value={buildComparisonValue(
            derivedMetrics?.home?.rendimiento_definicion,
            derivedMetrics?.away?.rendimiento_definicion,
          )}
        />
      </section>

      <ExpectedMetricsPanel expectedMetrics={expectedMetrics} />
      <MatchTimeline timeline={payload.timeline} />
      <LineupsPanel lineups={payload.lineups} />
      <PlayerStatsTable playerStats={payload.player_stats} />
      <InsightsPanel insights={payload.insights} />

      <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <h2 className="text-lg font-semibold text-slate-50">Resumen técnico</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            title="Cobertura"
            value={getDataQualityLabel(payload.data_quality?.level)}
            subtitle="Evaluación simple de calidad del backend"
          />
          <MetricCard
            title="Coordenadas de eventos"
            value={hasCoordinates ? 'Disponibles' : 'No disponibles'}
          />
          <MetricCard
            title="Mapas tácticos"
            value={eventMapsEnabled ? 'Habilitados' : 'No disponibles'}
          />
          <MetricCard
            title="xG local / visitante"
            value={`${formatMetricValue(expectedMetrics?.home?.xg)} / ${formatMetricValue(expectedMetrics?.away?.xg)}`}
          />
        </div>
        {!hasCoordinates ? (
          <p className="mt-4 text-sm text-slate-300">
            Las visualizaciones tácticas espaciales no están disponibles para Sportmonks en este
            partido porque el backend no confirmó coordenadas de eventos.
          </p>
        ) : null}
      </section>
    </div>
  )
}
