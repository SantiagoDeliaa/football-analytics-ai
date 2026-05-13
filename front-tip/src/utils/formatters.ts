import type { MatchMetrics } from '../types/eventData'

export function formatMetricValue(value: number | null | undefined, decimals = 1) {
  if (value === null || value === undefined) {
    return 'No aplica'
  }

  return Number.isInteger(value) ? `${value}` : value.toFixed(decimals)
}

export function formatPercentage(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return 'No aplica'
  }

  return `${Math.round(value)}%`
}

export function formatDateTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat('es-AR', {
    dateStyle: 'short',
    timeStyle: 'short',
  }).format(date)
}

export function buildPrimaryMetrics(metrics: MatchMetrics) {
  return [
    { title: 'Eventos analizados', value: `${metrics.total_events}` },
    { title: 'Pases', value: `${metrics.total_passes}` },
    { title: 'Remates', value: `${metrics.total_shots}` },
    { title: 'xG total', value: metrics.total_xg.toFixed(2) },
    { title: 'Acciones progresivas', value: `${metrics.progressive_actions}` },
    { title: 'Acciones en último tercio', value: `${metrics.final_third_actions}` },
    { title: 'Recuperaciones', value: `${metrics.recoveries}` },
    { title: 'Acciones bajo presión', value: `${metrics.total_under_pressure}` },
  ]
}

export function buildProprietaryMetrics(metrics: MatchMetrics) {
  return [
    {
      title: 'Field Tilt',
      value: formatMetricValue(metrics.field_tilt_index),
      subtitle: metrics.field_tilt_label,
    },
    {
      title: 'Directness',
      value: formatMetricValue(metrics.directness_index),
      subtitle: metrics.directness_label,
    },
    {
      title: 'Amenaza progresiva',
      value: formatMetricValue(metrics.progressive_threat_index),
      subtitle: metrics.progressive_threat_label,
    },
    {
      title: 'Altura de recuperación',
      value: formatMetricValue(metrics.recovery_height_index),
      subtitle: metrics.recovery_height_label,
    },
    {
      title: 'Calidad de remate',
      value: formatMetricValue(metrics.shot_quality_index),
      subtitle: metrics.shot_quality_label,
    },
    {
      title: 'Influencia jugador',
      value: formatMetricValue(metrics.player_influence_score),
      subtitle: metrics.player_influence_label,
    },
  ]
}
