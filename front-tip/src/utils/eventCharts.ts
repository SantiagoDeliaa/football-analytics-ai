import type { CanonicalEvent, MatchMetrics } from '../types/eventData'

export interface EventTypeDistribution {
  labels: string[]
  values: number[]
}

export interface PitchScatterPoint {
  x: number
  y: number
}

export interface MetricChartItem {
  label: string
  value: number
}

export function buildEventTypeDistribution(events: CanonicalEvent[], maxItems = 6): EventTypeDistribution {
  const counts = new Map<string, number>()

  events.forEach((event) => {
    const key = event.event_type?.trim() || 'Otro'
    counts.set(key, (counts.get(key) ?? 0) + 1)
  })

  const sorted = [...counts.entries()].sort((left, right) => right[1] - left[1]).slice(0, maxItems)

  return {
    labels: sorted.map(([label]) => label),
    values: sorted.map(([, value]) => value),
  }
}

export function buildPitchScatterPoints(events: CanonicalEvent[], limit = 200): PitchScatterPoint[] {
  return events
    .filter(
      (event): event is CanonicalEvent & { x: number; y: number } =>
        typeof event.x === 'number' && Number.isFinite(event.x) && typeof event.y === 'number' && Number.isFinite(event.y),
    )
    .slice(0, limit)
    .map((event) => ({
      x: clamp(event.x, 0, 120),
      y: clamp(event.y, 0, 80),
    }))
}

export function buildProgressiveBreakdown(events: CanonicalEvent[]) {
  const progressive = events.filter((event) => event.progressive).length
  const nonProgressive = Math.max(0, events.length - progressive)

  return {
    labels: ['Progresivas', 'No progresivas'],
    values: [progressive, nonProgressive],
  }
}

export function buildPressureBreakdown(events: CanonicalEvent[]) {
  const underPressure = events.filter((event) => event.under_pressure).length
  const withoutPressure = Math.max(0, events.length - underPressure)

  return {
    labels: ['Bajo presión', 'Sin presión'],
    values: [underPressure, withoutPressure],
  }
}

export function buildPlayerRadarMetrics(metrics: MatchMetrics) {
  return {
    labels: ['Influencia', 'Verticalidad', 'Amenaza', 'Recuperación', 'Remate'],
    values: [
      normalizeMetric(metrics.player_influence_score),
      normalizeMetric(metrics.directness_index),
      normalizeMetric(metrics.progressive_threat_index),
      normalizeMetric(metrics.recovery_height_index),
      normalizeMetric(metrics.shot_quality_index),
    ],
  }
}

export function parseMetricNumber(rawValue: string) {
  const normalized = rawValue.replace(',', '.').match(/-?\d+(\.\d+)?/)
  if (!normalized) {
    return null
  }

  const value = Number(normalized[0])
  return Number.isFinite(value) ? value : null
}

export function buildMetricChartItems(items: Array<{ title: string; value: string }>, maxItems = 6): MetricChartItem[] {
  return items
    .map((item) => ({
      label: item.title,
      value: parseMetricNumber(item.value),
    }))
    .filter((item): item is MetricChartItem => item.value !== null)
    .slice(0, maxItems)
}

function normalizeMetric(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 0
  }

  return clamp(value, 0, 100)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}
