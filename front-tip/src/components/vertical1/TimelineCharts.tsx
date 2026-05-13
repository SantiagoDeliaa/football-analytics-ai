import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import type { ComputerVisionResult } from '../../types/computerVision'
import { buildTimelineChartModel, formatValue } from '../../utils/computerVisionInsights'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

export function TimelineCharts({ result }: { result: ComputerVisionResult }) {
  const charts = [
    { title: 'Altura de presión', unit: 'm', series: result.timeline.pressure_height },
    { title: 'Compactación', unit: 'm²', series: result.timeline.compactness },
    { title: 'Amplitud ofensiva', unit: 'm', series: result.timeline.offensive_width },
  ]

  return (
    <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">
      {charts.map((chart) => (
        <article className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4" key={chart.title}>
          <h3 className="text-base font-semibold text-slate-100">{chart.title}</h3>
          <p className="mt-1 text-sm text-slate-300">Serie temporal profesional por tramo del clip para comparar ambos equipos.</p>
          <ProfessionalLineChart fps={result.fps} series={chart.series} unit={chart.unit} />
        </article>
      ))}
    </section>
  )
}

function ProfessionalLineChart({
  fps,
  series,
  unit,
}: {
  fps: number
  series: ComputerVisionResult['timeline'][keyof ComputerVisionResult['timeline']]
  unit: string
}) {
  const chartModel = buildTimelineChartModel(series, fps)

  if (!chartModel) {
    return <EmptyChart />
  }

  const data: ChartData<'line'> = {
    labels: chartModel.labels,
    datasets: [
      {
        label: 'Team 1',
        data: chartModel.team1,
        borderColor: '#34d399',
        backgroundColor: 'rgba(52, 211, 153, 0.14)',
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2.5,
        tension: 0.35,
        fill: true,
      },
      {
        label: 'Team 2',
        data: chartModel.team2,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96, 165, 250, 0.10)',
        pointRadius: 0,
        pointHoverRadius: 4,
        borderWidth: 2.5,
        tension: 0.35,
        fill: true,
      },
    ],
  }

  const padding = Math.max((chartModel.max - chartModel.min) * 0.1, 1)
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        align: 'start',
        labels: {
          color: '#cbd5e1',
          usePointStyle: true,
          boxWidth: 8,
          boxHeight: 8,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
        borderColor: 'rgba(148, 163, 184, 0.25)',
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        callbacks: {
          label(context) {
            const label = context.dataset.label ?? 'Serie'
            const value = context.parsed.y
            return `${label}: ${formatValue(typeof value === 'number' ? value : null, unit)}`
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(51, 65, 85, 0.35)',
        },
        ticks: {
          color: '#94a3b8',
          maxTicksLimit: 6,
        },
      },
      y: {
        min: chartModel.min - padding,
        max: chartModel.max + padding,
        grid: {
          color: 'rgba(51, 65, 85, 0.35)',
        },
        ticks: {
          color: '#94a3b8',
          callback(value) {
            return `${value} ${unit}`
          },
        },
      },
    },
  }

  return (
    <div className="mt-4 space-y-3">
      <div className="h-64 rounded-xl border border-slate-800 bg-slate-950/70 p-3">
        <Line data={data} options={options} />
      </div>
      <div className="grid grid-cols-1 gap-3 text-xs text-slate-300 md:grid-cols-3">
        <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-3 py-2">
          <span className="text-slate-500">Team 1 promedio</span>
          <p className="mt-1 text-sm font-semibold text-emerald-300">{formatValue(chartModel.averageTeam1, unit)}</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-3 py-2">
          <span className="text-slate-500">Team 2 promedio</span>
          <p className="mt-1 text-sm font-semibold text-sky-300">{formatValue(chartModel.averageTeam2, unit)}</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-3 py-2">
          <span className="text-slate-500">Rango observado</span>
          <p className="mt-1 text-sm font-semibold text-slate-100">
            {formatValue(chartModel.min, unit)} a {formatValue(chartModel.max, unit)}
          </p>
        </div>
      </div>
    </div>
  )
}

function EmptyChart() {
  return (
    <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
      Sin datos suficientes en este tramo.
    </div>
  )
}
