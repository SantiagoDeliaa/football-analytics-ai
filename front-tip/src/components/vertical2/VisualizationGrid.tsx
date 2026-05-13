import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
  type Plugin,
} from 'chart.js'
import { Bar, Doughnut, Scatter } from 'react-chartjs-2'
import type { CanonicalEvent } from '../../types/eventData'
import {
  buildEventTypeDistribution,
  buildPitchScatterPoints,
  buildPressureBreakdown,
  buildProgressiveBreakdown,
} from '../../utils/eventCharts'

ChartJS.register(CategoryScale, LinearScale, PointElement, BarElement, ArcElement, Tooltip, Legend)

const pitchPlugin: Plugin<'scatter'> = {
  id: 'pitchPlugin',
  beforeDraw(chart) {
    const { ctx, chartArea, scales } = chart
    const xScale = scales.x
    const yScale = scales.y

    if (!chartArea || !xScale || !yScale) {
      return
    }

    const left = xScale.getPixelForValue(0)
    const right = xScale.getPixelForValue(120)
    const bottom = yScale.getPixelForValue(0)
    const top = yScale.getPixelForValue(80)
    const centerX = xScale.getPixelForValue(60)
    const centerY = yScale.getPixelForValue(40)
    const radius = Math.abs(xScale.getPixelForValue(50) - xScale.getPixelForValue(40))

    ctx.save()
    ctx.fillStyle = '#142130'
    ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top)
    ctx.strokeStyle = 'rgba(134, 239, 172, 0.5)'
    ctx.lineWidth = 1

    ctx.strokeRect(left, top, right - left, bottom - top)

    ctx.beginPath()
    ctx.moveTo(centerX, top)
    ctx.lineTo(centerX, bottom)
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
    ctx.stroke()

    drawPenaltyBox(ctx, xScale, yScale, 0, 18)
    drawPenaltyBox(ctx, xScale, yScale, 102, 120)

    ctx.restore()
  },
}

function drawPenaltyBox(
  ctx: CanvasRenderingContext2D,
  xScale: { getPixelForValue(value: number): number },
  yScale: { getPixelForValue(value: number): number },
  fromX: number,
  toX: number,
) {
  const top = yScale.getPixelForValue(62)
  const bottom = yScale.getPixelForValue(18)
  const left = xScale.getPixelForValue(fromX)
  const right = xScale.getPixelForValue(toX)
  ctx.strokeRect(Math.min(left, right), Math.min(top, bottom), Math.abs(right - left), Math.abs(bottom - top))
}

function PitchScatterCard({
  color,
  emptyLabel,
  events,
  title,
}: {
  color: string
  emptyLabel: string
  events: CanonicalEvent[]
  title: string
}) {
  const points = buildPitchScatterPoints(events)

  if (!points.length) {
    return (
      <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
        <div className="mt-3 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
          {emptyLabel}
        </div>
      </article>
    )
  }

  const data: ChartData<'scatter'> = {
    datasets: [
      {
        label: title,
        data: points,
        pointRadius: 4,
        pointHoverRadius: 5,
        pointBackgroundColor: color,
        pointBorderColor: 'rgba(15, 23, 42, 0.85)',
        pointBorderWidth: 1,
      },
    ],
  }

  const options: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
        borderColor: 'rgba(148, 163, 184, 0.25)',
        borderWidth: 1,
        callbacks: {
          label(context) {
            const x = typeof context.parsed.x === 'number' ? context.parsed.x : 0
            const y = typeof context.parsed.y === 'number' ? context.parsed.y : 0
            return `x ${x.toFixed(1)} · y ${y.toFixed(1)}`
          },
        },
      },
    },
    scales: {
      x: {
        min: 0,
        max: 120,
        display: false,
        grid: {
          display: false,
        },
      },
      y: {
        min: 0,
        max: 80,
        display: false,
        grid: {
          display: false,
        },
      },
    },
  }

  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
        <span className="text-xs text-slate-400">{points.length} eventos</span>
      </div>
      <div className="mt-3 h-56 rounded-xl border border-slate-800 bg-slate-950/70 p-2">
        <Scatter data={data} options={options} plugins={[pitchPlugin]} />
      </div>
    </article>
  )
}

function EventTypeBarCard({ events }: { events: CanonicalEvent[] }) {
  const distribution = buildEventTypeDistribution(events)

  if (!distribution.labels.length) {
    return null
  }

  const data: ChartData<'bar'> = {
    labels: distribution.labels,
    datasets: [
      {
        label: 'Eventos',
        data: distribution.values,
        backgroundColor: ['#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#f87171', '#22d3ee'],
        borderRadius: 10,
      },
    ],
  }

  const options: ChartOptions<'bar'> = {
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
      x: {
        ticks: {
          color: '#94a3b8',
        },
        grid: {
          display: false,
        },
      },
      y: {
        ticks: {
          color: '#94a3b8',
          precision: 0,
        },
        grid: {
          color: 'rgba(51, 65, 85, 0.35)',
        },
      },
    },
  }

  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h4 className="text-sm font-semibold text-slate-100">Distribución de eventos</h4>
      <p className="mt-1 text-xs text-slate-300">Top de acciones detectadas en el recorte filtrado.</p>
      <div className="mt-3 h-64 rounded-xl border border-slate-800 bg-slate-950/70 p-3">
        <Bar data={data} options={options} />
      </div>
    </article>
  )
}

function BreakdownDonutCard({
  colors,
  labels,
  title,
  values,
}: {
  colors: string[]
  labels: string[]
  title: string
  values: number[]
}) {
  if (!values.some((value) => value > 0)) {
    return null
  }

  const data: ChartData<'doughnut'> = {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: colors,
        borderColor: 'rgba(15, 23, 42, 0.8)',
        borderWidth: 2,
        hoverOffset: 6,
      },
    ],
  }

  const options: ChartOptions<'doughnut'> = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '62%',
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#cbd5e1',
          usePointStyle: true,
          boxWidth: 8,
          boxHeight: 8,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.94)',
      },
    },
  }

  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
      <div className="mt-3 h-64 rounded-xl border border-slate-800 bg-slate-950/70 p-3">
        <Doughnut data={data} options={options} />
      </div>
    </article>
  )
}

export function VisualizationGrid({ events }: { events: CanonicalEvent[] }) {
  const shots = events.filter((event) => event.event_type === 'Shot')
  const passes = events.filter((event) => event.event_type === 'Pass')
  const recoveries = events.filter((event) =>
    ['Ball Recovery', 'Interception', 'Duel'].includes(event.event_type),
  )
  const progressive = buildProgressiveBreakdown(events)
  const pressure = buildPressureBreakdown(events)

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <PitchScatterCard
        color="#38bdf8"
        emptyLabel="No hay eventos con coordenadas suficientes para este recorte."
        events={events}
        title="Mapa general de eventos"
      />
      <PitchScatterCard
        color="#f59e0b"
        emptyLabel="No se detectaron remates con coordenadas."
        events={shots}
        title="Mapa de remates"
      />
      <PitchScatterCard
        color="#34d399"
        emptyLabel="No se detectaron pases con coordenadas."
        events={passes}
        title="Mapa de pases"
      />
      <PitchScatterCard
        color="#a78bfa"
        emptyLabel="No se detectaron recuperaciones con coordenadas."
        events={recoveries}
        title="Mapa de recuperaciones"
      />
      <EventTypeBarCard events={events} />
      <BreakdownDonutCard
        colors={['#34d399', '#334155']}
        labels={progressive.labels}
        title="Peso de acciones progresivas"
        values={progressive.values}
      />
      <BreakdownDonutCard
        colors={['#60a5fa', '#334155']}
        labels={pressure.labels}
        title="Contexto de presión"
        values={pressure.values}
      />
    </div>
  )
}
