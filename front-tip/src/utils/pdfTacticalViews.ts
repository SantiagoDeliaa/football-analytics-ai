import type { Config, Data, Layout, Shape } from 'plotly.js'

export type Zone = {
  label: string
  value: number
  x: number
  y: number
  width: number
  height: number
  color: string
}

type PitchViewDefinition = {
  key: 'attack' | 'defense' | 'transitions'
  title: string
  subtitle: string
  zones: Zone[]
  hasSignal: boolean
}

export type PlotFigure = {
  data: Data[]
  layout: Partial<Layout>
  config: Partial<Config>
}

type PlotShape = Partial<Shape>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getSignal(payload: Record<string, unknown>, section: string, key: string) {
  const sectionValue = payload[section]
  if (!isRecord(sectionValue) || !isRecord(sectionValue.signals)) {
    return 0
  }

  const rawValue = sectionValue.signals[key]
  return typeof rawValue === 'number' ? rawValue : Number(rawValue ?? 0) || 0
}

export function buildPdfRadarMetrics(payload: Record<string, unknown>) {
  const finalThirdActivity = Math.max(
    0,
    Math.min(
      100,
      Math.round(
        40 +
          8 * getSignal(payload, 'attack', 'final third') +
          8 * getSignal(payload, 'attack', 'shots') +
          7 * getSignal(payload, 'attack', 'box entries'),
      ),
    ),
  )

  return [
    { label: 'Control territorial', value: finalThirdActivity },
    { label: 'Verticalidad', value: Math.min(100, Math.round(getSignal(payload, 'build_up', 'progression') * 12)) },
    { label: 'Impacto pressing', value: Math.min(100, Math.round(getSignal(payload, 'defense', 'pressing') * 18)) },
    { label: 'Riesgo en salida', value: Math.min(100, Math.round(getSignal(payload, 'transitions', 'turnover') * 18)) },
    { label: 'Actividad último tercio', value: Math.min(100, Math.round(getSignal(payload, 'attack', 'xg') * 20 + finalThirdActivity * 0.45)) },
  ]
}

function hexToRgba(hexColor: string, alpha: number) {
  const color = hexColor.replace('#', '')
  const r = Number.parseInt(color.slice(0, 2), 16)
  const g = Number.parseInt(color.slice(2, 4), 16)
  const b = Number.parseInt(color.slice(4, 6), 16)

  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha)).toFixed(2)})`
}

function buildPitchShapes(zones: Zone[], maxValue: number): PlotShape[] {
  const fieldShapes: PlotShape[] = [
    {
      type: 'rect',
      x0: 0,
      y0: 0,
      x1: 105,
      y1: 68,
      line: { color: '#e2e8f0', width: 2 },
      fillcolor: '#166534',
    },
    { type: 'line', x0: 52.5, y0: 0, x1: 52.5, y1: 68, line: { color: '#e2e8f0', width: 2 } },
    { type: 'circle', x0: 43.5, y0: 25, x1: 61.5, y1: 43, line: { color: '#e2e8f0', width: 2 } },
    { type: 'rect', x0: 0, y0: 13.84, x1: 16.5, y1: 54.16, line: { color: '#e2e8f0', width: 2 } },
    { type: 'rect', x0: 88.5, y0: 13.84, x1: 105, y1: 54.16, line: { color: '#e2e8f0', width: 2 } },
    { type: 'rect', x0: 0, y0: 24.84, x1: 5.5, y1: 43.16, line: { color: '#e2e8f0', width: 2 } },
    { type: 'rect', x0: 99.5, y0: 24.84, x1: 105, y1: 43.16, line: { color: '#e2e8f0', width: 2 } },
  ]

  return [
    ...fieldShapes,
    ...zones.map((zone) => {
      const opacity = zoneOpacity(zone.value, maxValue)
      return {
        type: 'rect' as const,
        x0: zone.x,
        y0: zone.y,
        x1: zone.x + zone.width,
        y1: zone.y + zone.height,
        line: {
          color: hexToRgba(zone.color, Math.min(0.95, opacity + 0.2)),
          width: 2,
        },
        fillcolor: hexToRgba(zone.color, opacity),
      }
    }),
  ]
}

export function buildPdfRadarFigure(payload: Record<string, unknown>): PlotFigure {
  const metrics = buildPdfRadarMetrics(payload)
  const labels = [...metrics.map((metric) => metric.label), metrics[0]?.label ?? 'Control territorial']
  const values = [...metrics.map((metric) => metric.value), metrics[0]?.value ?? 0]

  return {
    data: [
      {
        type: 'scatterpolar',
        r: values,
        theta: labels,
        fill: 'toself',
        fillcolor: 'rgba(52, 211, 153, 0.22)',
        line: {
          color: '#34d399',
          width: 3,
        },
        marker: {
          color: '#7dd3fc',
          size: 8,
        },
        hovertemplate: '%{theta}<br>%{r:.0f}/100<extra></extra>',
      },
    ],
    layout: {
      autosize: true,
      margin: { l: 56, r: 56, t: 28, b: 56 },
      paper_bgcolor: '#020617',
      plot_bgcolor: '#020617',
      font: { color: '#e2e8f0' },
      polar: {
        bgcolor: '#020617',
        radialaxis: {
          visible: true,
          range: [0, 100],
          tickfont: { color: '#94a3b8', size: 10 },
          gridcolor: 'rgba(148, 163, 184, 0.22)',
          linecolor: 'rgba(148, 163, 184, 0.22)',
          angle: 90,
        },
        angularaxis: {
          tickfont: { color: '#cbd5e1', size: 10 },
          gridcolor: 'rgba(148, 163, 184, 0.12)',
          linecolor: 'rgba(148, 163, 184, 0.12)',
        },
      },
      showlegend: false,
    },
    config: {
      displayModeBar: false,
      responsive: true,
    },
  }
}

export function buildPdfPitchViews(payload: Record<string, unknown>): PitchViewDefinition[] {
  const attackZones: Zone[] = [
    {
      label: 'Zonas de centros',
      value: getSignal(payload, 'attack', 'crosses'),
      x: 72,
      y: 0,
      width: 33,
      height: 18,
      color: '#60a5fa',
    },
    {
      label: 'Dribbles y entradas',
      value: (getSignal(payload, 'attack', 'box entries') + getSignal(payload, 'attack', 'final third')) / 2,
      x: 62,
      y: 18,
      width: 28,
      height: 32,
      color: '#34d399',
    },
    {
      label: 'Recuperaciones altas',
      value: getSignal(payload, 'defense', 'recoveries') + getSignal(payload, 'transitions', 'regain'),
      x: 70,
      y: 18,
      width: 35,
      height: 32,
      color: '#fbbf24',
    },
    {
      label: 'Origen de remates',
      value: getSignal(payload, 'attack', 'shots') + getSignal(payload, 'attack', 'xg'),
      x: 86,
      y: 24,
      width: 19,
      height: 20,
      color: '#f87171',
    },
  ]

  const defenseZones: Zone[] = [
    {
      label: 'Duelos en tercio propio',
      value: getSignal(payload, 'defense', 'duels'),
      x: 0,
      y: 18,
      width: 35,
      height: 32,
      color: '#60a5fa',
    },
    {
      label: 'Pérdidas peligrosas',
      value: getSignal(payload, 'transitions', 'turnover'),
      x: 0,
      y: 20,
      width: 52.5,
      height: 28,
      color: '#f87171',
    },
    {
      label: 'Fragilidad costado izq.',
      value: Math.max(0, getSignal(payload, 'transitions', 'turnover') - getSignal(payload, 'defense', 'pressing') * 0.4),
      x: 0,
      y: 52,
      width: 40,
      height: 16,
      color: '#fb7185',
    },
    {
      label: 'Fragilidad costado der.',
      value: Math.max(0, getSignal(payload, 'transitions', 'turnover') - getSignal(payload, 'defense', 'pressing') * 0.4),
      x: 0,
      y: 0,
      width: 40,
      height: 16,
      color: '#fb7185',
    },
  ]

  const transitionZones: Zone[] = [
    {
      label: 'Recuperaciones por zona',
      value: getSignal(payload, 'transitions', 'regain'),
      x: 35,
      y: 20,
      width: 35,
      height: 28,
      color: '#22c55e',
    },
    {
      label: 'Pérdidas por zona',
      value: getSignal(payload, 'transitions', 'turnover'),
      x: 20,
      y: 20,
      width: 40,
      height: 28,
      color: '#ef4444',
    },
    {
      label: 'Salida de contra',
      value: getSignal(payload, 'transitions', 'counter') + getSignal(payload, 'transitions', 'direct attack'),
      x: 60,
      y: 18,
      width: 35,
      height: 32,
      color: '#38bdf8',
    },
    {
      label: 'Ritmo de transición',
      value: getSignal(payload, 'transitions', 'transition'),
      x: 45,
      y: 0,
      width: 35,
      height: 68,
      color: '#f59e0b',
    },
  ]

  const views: PitchViewDefinition[] = [
    {
      key: 'attack',
      title: 'Attack View',
      subtitle: 'Dónde genera ventaja el equipo en fase ofensiva.',
      zones: attackZones,
      hasSignal: attackZones.some((zone) => zone.value > 0),
    },
    {
      key: 'defense',
      title: 'Defense View',
      subtitle: 'Dónde sufre más el equipo cuando defiende su propio arco.',
      zones: defenseZones,
      hasSignal: defenseZones.some((zone) => zone.value > 0),
    },
    {
      key: 'transitions',
      title: 'Transitions View',
      subtitle: 'Cómo responde el equipo al recuperar o perder el balón.',
      zones: transitionZones,
      hasSignal: transitionZones.some((zone) => zone.value > 0),
    },
  ]

  return views
}

export function zoneOpacity(value: number, maxValue: number) {
  if (maxValue <= 0) {
    return 0.18
  }

  const ratio = Math.max(0, Math.min(1, value / maxValue))
  return 0.18 + ratio * 0.5
}

export function buildPdfPitchFigure(view: PitchViewDefinition): PlotFigure {
  const maxValue = Math.max(...view.zones.map((zone) => zone.value), 0)
  const hasSignal = view.zones.some((zone) => zone.value > 0)

  return {
    data: hasSignal
      ? [
          {
            type: 'scatter',
            mode: 'markers',
            x: view.zones.map((zone) => zone.x + zone.width / 2),
            y: view.zones.map((zone) => zone.y + zone.height / 2),
            text: view.zones.map((zone) => zone.label),
            customdata: view.zones.map((zone) => [zone.value.toFixed(1), zone.color]),
            marker: {
              color: view.zones.map((zone) => zone.color),
              size: view.zones.map((zone) => 10 + zoneOpacity(zone.value, maxValue) * 24),
              opacity: 0.92,
              line: {
                color: '#f8fafc',
                width: 1,
              },
            },
            hovertemplate:
              '<b>%{text}</b><br>Intensidad: %{customdata[0]}<extra></extra>',
          },
        ]
      : [],
    layout: {
      autosize: true,
      margin: { l: 16, r: 16, t: 16, b: 16 },
      paper_bgcolor: '#0f1720',
      plot_bgcolor: '#166534',
      font: { color: '#f8fafc' },
      showlegend: false,
      xaxis: {
        range: [0, 105],
        visible: false,
        fixedrange: true,
      },
      yaxis: {
        range: [68, 0],
        visible: false,
        fixedrange: true,
        scaleanchor: 'x',
        scaleratio: 1,
      },
      shapes: buildPitchShapes(view.zones, maxValue) as Shape[],
      annotations: hasSignal
        ? []
        : [
            {
              x: 52.5,
              y: 34,
              text: 'Sin señales zonales suficientes',
              showarrow: false,
              font: { color: '#f8fafc', size: 16 },
            },
          ],
    },
    config: {
      displayModeBar: false,
      responsive: true,
    },
  }
}
