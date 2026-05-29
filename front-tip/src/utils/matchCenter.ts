import type {
  MatchCenterDataQuality,
  MatchCenterLineupSide,
  MatchCenterMatch,
  MatchCenterStatEntry,
} from '../types/matchCenter'

export function formatNullableValue(
  value: number | string | boolean | null | undefined,
  fallback = 'No disponible',
) {
  if (value === null || value === undefined || value === '') {
    return fallback
  }
  if (typeof value === 'boolean') {
    return value ? 'Sí' : 'No'
  }
  return `${value}`
}

export function formatMetricValue(value: number | null | undefined, fallback = 'No disponible') {
  if (value === null || value === undefined) {
    return fallback
  }
  return Number.isInteger(value) ? `${value}` : value.toFixed(2)
}

export function formatMatchScore(match: MatchCenterMatch) {
  const homeScore = match.home_team.score
  const awayScore = match.away_team.score
  if (homeScore === null || homeScore === undefined || awayScore === null || awayScore === undefined) {
    return 'No disponible'
  }
  return `${homeScore} - ${awayScore}`
}

export function formatMatchDate(date?: string | null) {
  if (!date) {
    return 'Fecha no disponible'
  }

  const parsedDate = new Date(date)
  if (Number.isNaN(parsedDate.getTime())) {
    return date
  }

  return new Intl.DateTimeFormat('es-AR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsedDate)
}

export function buildComparisonValue(
  home: number | null | undefined,
  away: number | null | undefined,
  fallback = 'No disponible',
) {
  if (home === null || home === undefined || away === null || away === undefined) {
    return fallback
  }
  return `${formatMetricValue(home)} vs ${formatMetricValue(away)}`
}

export function getDataQualityLabel(level?: MatchCenterDataQuality['level']) {
  const normalized = `${level ?? ''}`.trim().toLowerCase()
  if (normalized === 'alta') return 'Alta'
  if (normalized === 'media') return 'Media'
  if (normalized === 'baja') return 'Baja'
  return 'No disponible'
}

export function getLineupAvailability(lineup?: MatchCenterLineupSide | null) {
  return Boolean(lineup && (lineup.starters.length > 0 || lineup.substitutes.length > 0))
}

export function getPlayerStatValue(stats: MatchCenterStatEntry[] | undefined, keys: string[]) {
  if (!stats?.length) {
    return undefined
  }

  const normalizedKeys = keys.map((key) => key.trim().toLowerCase())
  return stats.find((stat) => normalizedKeys.includes(stat.key.trim().toLowerCase()))?.value
}
