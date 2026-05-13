import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { buildMetricChartItems, parseMetricNumber } from '../../utils/eventCharts'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)

interface MetricItem {
  title: string
  value: string
  subtitle?: string
}

interface MetricsSectionProps {
  title: string
  items: MetricItem[]
  columns?: string
}

export function MetricsSection({
  title,
  items,
  columns = 'grid-cols-1 md:grid-cols-2 xl:grid-cols-4',
}: MetricsSectionProps) {
  const chartItems = buildMetricChartItems(items)
  const maxMetricValue = Math.max(...chartItems.map((item) => item.value), 1)

  const chartData: ChartData<'bar'> = {
    labels: chartItems.map((item) => item.label),
    datasets: [
      {
        label: 'Valor',
        data: chartItems.map((item) => item.value),
        backgroundColor: ['#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#22d3ee', '#f87171'],
        borderRadius: 10,
        borderSkipped: false,
      },
    ],
  }

  const chartOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y',
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#94a3b8',
        },
        grid: {
          color: 'rgba(51, 65, 85, 0.35)',
        },
      },
      y: {
        ticks: {
          color: '#cbd5e1',
        },
        grid: {
          display: false,
        },
      },
    },
  }

  return (
    <section className="space-y-3">
      <header>
        <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
      </header>

      {chartItems.length >= 2 ? (
        <article className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
          <p className="text-sm text-slate-300">
            Lectura visual rápida de los KPIs más relevantes del bloque actual.
          </p>
          <div className="mt-4 h-72 rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <Bar data={chartData} options={chartOptions} />
          </div>
        </article>
      ) : null}

      <div className={`grid gap-4 ${columns}`}>
        {items.map((item) => (
          <MetricInsightCard
            key={`${item.title}-${item.value}`}
            maxMetricValue={maxMetricValue}
            subtitle={item.subtitle}
            title={item.title}
            value={item.value}
          />
        ))}
      </div>
    </section>
  )
}

function MetricInsightCard({
  maxMetricValue,
  subtitle,
  title,
  value,
}: {
  maxMetricValue: number
  subtitle?: string
  title: string
  value: string
}) {
  const numericValue = parseMetricNumber(value)
  const intensity =
    numericValue === null || maxMetricValue <= 0 ? null : Math.max(6, Math.min(100, (numericValue / maxMetricValue) * 100))

  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-slate-400">{title}</p>
          <p className="mt-2 text-2xl font-semibold text-slate-50">{value}</p>
        </div>
        {subtitle ? (
          <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-semibold text-emerald-200">
            {subtitle}
          </span>
        ) : null}
      </div>
      {intensity !== null ? (
        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em] text-slate-500">
            <span>Intensidad relativa</span>
            <span>{Math.round(intensity)}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-800">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-emerald-400 via-sky-400 to-violet-400"
              style={{ width: `${intensity}%` }}
            />
          </div>
        </div>
      ) : null}
    </article>
  )
}
