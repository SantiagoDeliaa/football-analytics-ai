import {
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  PointElement,
  RadialLinearScale,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from 'chart.js'
import { Radar } from 'react-chartjs-2'
import type { MatchMetrics } from '../../types/eventData'
import { buildPlayerRadarMetrics } from '../../utils/eventCharts'
import { MetricCard } from '../common/MetricCard'
import { formatMetricValue } from '../../utils/formatters'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

interface PlayerSpotlightProps {
  player: string
  metrics: MatchMetrics
}

export function PlayerSpotlight({ player, metrics }: PlayerSpotlightProps) {
  if (!player || player === 'Todos') {
    return (
      <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Análisis de jugador</h3>
        <p className="mt-2 text-sm text-slate-300">
          Seleccioná un jugador específico para ver su recorte táctico individual.
        </p>
      </section>
    )
  }

  const radarMetrics = buildPlayerRadarMetrics(metrics)
  const radarData: ChartData<'radar'> = {
    labels: radarMetrics.labels,
    datasets: [
      {
        label: player,
        data: radarMetrics.values,
        borderColor: '#34d399',
        backgroundColor: 'rgba(52, 211, 153, 0.18)',
        pointBackgroundColor: '#34d399',
        pointBorderColor: '#022c22',
        pointHoverBackgroundColor: '#86efac',
        borderWidth: 2,
      },
    ],
  }

  const radarOptions: ChartOptions<'radar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
      },
    },
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: {
          stepSize: 20,
          color: '#94a3b8',
          backdropColor: 'transparent',
        },
        grid: {
          color: 'rgba(71, 85, 105, 0.45)',
        },
        angleLines: {
          color: 'rgba(71, 85, 105, 0.35)',
        },
        pointLabels: {
          color: '#e2e8f0',
          font: {
            size: 12,
          },
        },
      },
    },
  }

  return (
    <section className="space-y-3">
      <header className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Análisis de jugador</h3>
        <p className="mt-2 text-sm text-slate-300">
          {player} queda aislado con foco en participación, progresión, amenaza y trabajo defensivo.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard title="Eventos" value={`${metrics.total_events}`} />
        <MetricCard title="Pases" value={`${metrics.total_passes}`} />
        <MetricCard title="Remates" value={`${metrics.total_shots}`} />
        <MetricCard title="Progresiones" value={`${metrics.progressive_actions}`} />
        <MetricCard title="Recuperaciones" value={`${metrics.recoveries}`} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <MetricCard
          subtitle={metrics.player_influence_label}
          title="Influencia"
          value={formatMetricValue(metrics.player_influence_score)}
        />
        <MetricCard
          subtitle={metrics.directness_label}
          title="Verticalidad"
          value={formatMetricValue(metrics.directness_index)}
        />
        <MetricCard
          subtitle={metrics.shot_quality_label}
          title="Calidad de remate"
          value={formatMetricValue(metrics.shot_quality_index)}
        />
      </div>

      <article className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h4 className="text-base font-semibold text-slate-100">Perfil radar del jugador</h4>
        <p className="mt-2 text-sm text-slate-300">
          Combina influencia, verticalidad, amenaza, recuperación y remate en una sola lectura visual.
        </p>
        <div className="mt-4 h-80 rounded-xl border border-slate-800 bg-slate-950/70 p-4">
          <Radar data={radarData} options={radarOptions} />
        </div>
      </article>
    </section>
  )
}
