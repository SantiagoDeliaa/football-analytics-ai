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
  summary?: string
  whatItMeasures?: string
  interpretation?: string
  highMeaning?: string
  lowMeaning?: string
  limitations?: string
  coachQuestion?: string
}

interface MetricsSectionProps {
  title: string
  items: MetricItem[]
  columns?: string
  onAskCoach?: (question: string) => void
}

export function MetricsSection({
  title,
  items,
  columns = 'grid-cols-1 md:grid-cols-2 xl:grid-cols-4',
  onAskCoach,
}: MetricsSectionProps) {
  const chartItems = buildMetricChartItems(items)
  const maxMetricValue = Math.max(...chartItems.map((item) => item.value), 1)

  const chartData: ChartData<'bar'> = {
    labels: chartItems.map((item) => item.label),
    datasets: [
      {
        label: 'Valor relativo',
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
            item={item}
            onAskCoach={onAskCoach}
          />
        ))}
      </div>
    </section>
  )
}

function MetricInsightCard({
  item,
  maxMetricValue,
  onAskCoach,
}: {
  item: MetricItem
  maxMetricValue: number
  onAskCoach?: (question: string) => void
}) {
  const { coachQuestion, highMeaning, interpretation, limitations, lowMeaning, subtitle, summary, title, value, whatItMeasures } =
    item
  const numericValue = parseMetricNumber(value)
  const relativeValue =
    numericValue === null || maxMetricValue <= 0 ? null : Math.max(0, Math.min(100, (numericValue / maxMetricValue) * 100))

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
      {summary ? <p className="mt-3 text-sm leading-6 text-slate-300">{summary}</p> : null}
      {relativeValue !== null ? (
        <p className="mt-2 text-xs text-slate-500">
          Referencia visual del bloque: {Math.round(relativeValue)}% respecto del valor más alto mostrado.
        </p>
      ) : null}
      {whatItMeasures || interpretation || highMeaning || lowMeaning || limitations ? (
        <details className="mt-4 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <summary className="cursor-pointer text-sm font-semibold text-slate-200">
            ¿Qué significa esta métrica?
          </summary>
          <div className="mt-3 space-y-2 text-sm leading-6 text-slate-300">
            {whatItMeasures ? <p><strong className="text-slate-100">Qué mide:</strong> {whatItMeasures}</p> : null}
            {interpretation ? <p><strong className="text-slate-100">Cómo leerla:</strong> {interpretation}</p> : null}
            {highMeaning ? <p><strong className="text-slate-100">Si está alta:</strong> {highMeaning}</p> : null}
            {lowMeaning ? <p><strong className="text-slate-100">Si está baja:</strong> {lowMeaning}</p> : null}
            {limitations ? <p><strong className="text-slate-100">Limitaciones:</strong> {limitations}</p> : null}
          </div>
        </details>
      ) : null}
      {coachQuestion && onAskCoach ? (
        <button
          className="mt-4 rounded-md border border-sky-500/40 bg-sky-500/10 px-3 py-2 text-sm font-semibold text-sky-100 hover:bg-sky-500/20"
          onClick={() => onAskCoach(coachQuestion)}
          type="button"
        >
          Preguntar al AI Coach
        </button>
      ) : null}
    </article>
  )
}
