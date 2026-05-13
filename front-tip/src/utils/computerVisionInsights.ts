import type { PipelineHealthSummary, TimelineSeries } from '../types/computerVision'

export interface HomographyOverview {
  badge: string
  title: string
  description: string
  tone: 'emerald' | 'amber' | 'rose'
}

export interface TimelineChartModel {
  labels: string[]
  team1: Array<number | null>
  team2: Array<number | null>
  min: number
  max: number
  averageTeam1: number | null
  averageTeam2: number | null
}

export function formatRatioAsPercentage(value?: number | null, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/D'
  }

  return `${(value * 100).toFixed(decimals)}%`
}

export function formatValue(value?: number | null, unit?: string, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/D'
  }

  const suffix = unit ? ` ${unit}` : ''
  return `${value.toFixed(decimals)}${suffix}`
}

export function formatFramesWithShare(validFrames?: number | null, totalFrames?: number | null) {
  if (validFrames === null || validFrames === undefined || Number.isNaN(validFrames)) {
    return 'N/D'
  }

  if (totalFrames === null || totalFrames === undefined || totalFrames <= 0 || Number.isNaN(totalFrames)) {
    return `${validFrames}`
  }

  return `${validFrames}/${totalFrames} (${((validFrames / totalFrames) * 100).toFixed(1)}%)`
}

export function getHomographyOverview(healthSummary: PipelineHealthSummary): HomographyOverview {
  const fallbackRatio = healthSummary.fallback_ratio ?? 0
  const invalidRatio = healthSummary.invalid_ratio ?? 0
  const warnRatio = healthSummary.warn_ratio ?? 0
  const reprojectionP95 = healthSummary.p95_reproj_error_m ?? healthSummary.avg_reproj_error_m ?? 0

  if (
    healthSummary.demo_mode === 'stable' &&
    fallbackRatio <= 0.08 &&
    invalidRatio <= 0.12 &&
    warnRatio <= 0.3 &&
    reprojectionP95 <= 1.1
  ) {
    return {
      badge: 'Estable',
      title: 'Homografía aplicada de forma estable',
      description:
        'La reproyección se mantiene controlada y la lectura táctica puede apoyarse mejor en radar, tracking y métricas avanzadas.',
      tone: 'emerald',
    }
  }

  if (invalidRatio >= 0.45 || fallbackRatio >= 0.28 || reprojectionP95 >= 2.2) {
    return {
      badge: 'Crítica',
      title: 'Homografía con señal insuficiente',
      description:
        'El pipeline detectó demasiados frames inválidos o fallback, así que conviene priorizar el video y tomar con mucha cautela las métricas finas.',
      tone: 'rose',
    }
  }

  if (
    healthSummary.demo_mode === 'degraded' ||
    fallbackRatio >= 0.1 ||
    invalidRatio >= 0.18 ||
    warnRatio >= 0.45 ||
    reprojectionP95 >= 1.4
  ) {
    return {
      badge: 'Degradada',
      title: 'Homografía aplicada con alertas',
      description:
        'La homografía está presente, pero el clip muestra fallback o warnings frecuentes. Usá métricas y radar como guía, no como verdad absoluta.',
      tone: 'amber',
    }
  }

  return {
    badge: 'Monitoreada',
    title: 'Homografía aplicada',
    description:
      'La señal es usable, aunque aparecen pequeñas oscilaciones. El contexto del video sigue siendo importante para validar conclusiones tácticas.',
    tone: 'amber',
  }
}

export function buildTimelineChartModel(series: TimelineSeries | undefined, fps: number): TimelineChartModel | null {
  if (!series) {
    return null
  }

  const length = Math.max(series.frames.length, series.team1.length, series.team2.length)
  if (!length) {
    return null
  }

  const team1 = normalizeSeries(series.team1, length)
  const team2 = normalizeSeries(series.team2, length)
  const values = [...team1, ...team2].filter((value): value is number => value !== null)

  if (!values.length) {
    return null
  }

  return {
    labels: Array.from({ length }, (_, index) => formatFrameLabel(series.frames[index] ?? index, fps)),
    team1,
    team2,
    min: Math.min(...values),
    max: Math.max(...values),
    averageTeam1: averageSeries(team1),
    averageTeam2: averageSeries(team2),
  }
}

function normalizeSeries(values: number[], length: number) {
  return Array.from({ length }, (_, index) => {
    const value = values[index]
    return typeof value === 'number' && Number.isFinite(value) ? value : null
  })
}

function averageSeries(values: Array<number | null>) {
  const filtered = values.filter((value): value is number => value !== null)
  if (!filtered.length) {
    return null
  }

  return filtered.reduce((accumulator, value) => accumulator + value, 0) / filtered.length
}

function formatFrameLabel(frame: number, fps: number) {
  if (!Number.isFinite(frame) || !Number.isFinite(fps) || fps <= 0) {
    return `Frame ${Math.max(0, Math.round(frame))}`
  }

  const totalSeconds = Math.max(0, Math.round(frame / fps))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}
