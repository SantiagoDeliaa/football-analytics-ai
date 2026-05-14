import type { CanonicalEvent, MatchMetrics } from '../types/eventData'

const IGNORED_EVENT_TYPES = new Set([
  'Starting XI',
  'Half Start',
  'Half End',
  'Substitution',
  'Tactical Shift',
  'Bad Behaviour',
])

function safePercentage(numerator: number, denominator: number) {
  if (denominator <= 0) {
    return 0
  }

  return Math.max(0, Math.min(100, (numerator / denominator) * 100))
}

function getMetricLabel(score: number | null | undefined) {
  if (score === null || score === undefined) {
    return 'No aplica'
  }

  if (score < 40) {
    return 'Bajo'
  }

  if (score < 70) {
    return 'Medio'
  }

  return 'Alto'
}

function isFavorableDuel(outcome: string | null | undefined) {
  if (!outcome) {
    return false
  }

  const normalized = outcome.trim().toLowerCase()
  return ['won', 'success', 'success in play', 'tackle'].some((token) =>
    normalized.includes(token),
  )
}

export function filterAnalyticalEvents(events: CanonicalEvent[]) {
  return events.filter((event) => !IGNORED_EVENT_TYPES.has(event.event_type.trim()))
}

function calculateFieldTilt(events: CanonicalEvent[], selectedTeam?: string) {
  if (!selectedTeam || selectedTeam === 'Todos') {
    return null
  }

  const totalFinalThirdAll = events.filter(
    (event) => typeof event.x === 'number' && event.x >= 80,
  ).length

  const teamFinalThird = events.filter(
    (event) =>
      event.team_name === selectedTeam && typeof event.x === 'number' && event.x >= 80,
  ).length

  return Number(safePercentage(teamFinalThird, totalFinalThirdAll).toFixed(1))
}

function calculateDirectness(progressiveActions: number, totalPasses: number) {
  return Number(safePercentage(progressiveActions, totalPasses).toFixed(1))
}

function calculateProgressiveThreat(
  totalEvents: number,
  progressiveActions: number,
  finalThirdActions: number,
  totalShots: number,
  totalXg: number,
) {
  const rawPoints =
    progressiveActions * 2 +
    finalThirdActions * 1.5 +
    totalShots * 3 +
    totalXg * 25
  const maxPoints = Math.max(1, totalEvents * 3.5)

  return Number(Math.min(100, (rawPoints / maxPoints) * 100).toFixed(1))
}

function calculateRecoveryHeight(recoveryEvents: CanonicalEvent[]) {
  const xValues = recoveryEvents
    .map((event) => event.x)
    .filter((value): value is number => typeof value === 'number')

  if (!xValues.length) {
    return null
  }

  const avgX = xValues.reduce((sum, value) => sum + value, 0) / xValues.length
  return Number(safePercentage(avgX, 120).toFixed(1))
}

function calculateShotQuality(totalXg: number, totalShots: number) {
  if (totalShots <= 0) {
    return 0
  }

  return Number(((totalXg / totalShots) * 100).toFixed(1))
}

function calculatePlayerInfluence(
  selectedPlayer: string | undefined,
  totalEvents: number,
  totalPasses: number,
  progressiveActions: number,
  finalThirdActions: number,
  totalShots: number,
  recoveries: number,
) {
  if (!selectedPlayer || selectedPlayer === 'Todos') {
    return null
  }

  const activity = Math.min(1, totalEvents / 25)
  const creation = Math.min(1, (totalPasses + progressiveActions * 2) / 30)
  const threat = Math.min(1, (totalShots * 3 + finalThirdActions) / 20)
  const defensive = Math.min(1, recoveries / 8)
  const score = (0.35 * activity + 0.25 * creation + 0.25 * threat + 0.15 * defensive) * 100

  return Number(Math.min(100, Math.max(0, score)).toFixed(1))
}

export function calculateOpenEventMetrics(
  canonicalEvents: CanonicalEvent[],
  selection?: {
    team?: string
    player?: string
  },
): MatchMetrics {
  const analyticalEvents = filterAnalyticalEvents(canonicalEvents)
  const filteredEvents = analyticalEvents.filter((event) => {
    const matchesTeam =
      !selection?.team || selection.team === 'Todos'
        ? true
        : event.team_name === selection.team
    const matchesPlayer =
      !selection?.player || selection.player === 'Todos'
        ? true
        : event.player_name === selection.player

    return matchesTeam && matchesPlayer
  })

  const totalEvents = filteredEvents.length
  const totalPasses = filteredEvents.filter((event) => event.event_type === 'Pass').length
  const totalShots = filteredEvents.filter((event) => event.event_type === 'Shot').length
  const progressiveActions = filteredEvents.filter((event) => event.progressive).length
  const totalUnderPressure = filteredEvents.filter((event) => event.under_pressure).length
  const totalXg = filteredEvents
    .filter((event) => event.event_type === 'Shot')
    .reduce((sum, event) => sum + (event.xG ?? 0), 0)
  const finalThirdActions = filteredEvents.filter(
    (event) => typeof event.x === 'number' && event.x >= 80,
  ).length

  const recoveryEvents = filteredEvents.filter((event) => {
    if (event.event_type === 'Ball Recovery' || event.event_type === 'Interception') {
      return true
    }

    return event.event_type === 'Duel' && isFavorableDuel(event.outcome)
  })

  const recoveries = recoveryEvents.length
  const fieldTiltIndex = calculateFieldTilt(analyticalEvents, selection?.team)
  const directnessIndex = calculateDirectness(progressiveActions, totalPasses)
  const progressiveThreatIndex = calculateProgressiveThreat(
    totalEvents,
    progressiveActions,
    finalThirdActions,
    totalShots,
    totalXg,
  )
  const recoveryHeightIndex = calculateRecoveryHeight(recoveryEvents)
  const shotQualityIndex = calculateShotQuality(totalXg, totalShots)
  const playerInfluenceScore = calculatePlayerInfluence(
    selection?.player,
    totalEvents,
    totalPasses,
    progressiveActions,
    finalThirdActions,
    totalShots,
    recoveries,
  )

  return {
    total_events: totalEvents,
    total_passes: totalPasses,
    total_shots: totalShots,
    progressive_actions: progressiveActions,
    final_third_actions: finalThirdActions,
    recoveries,
    total_under_pressure: totalUnderPressure,
    total_xg: Number(totalXg.toFixed(3)),
    field_tilt_index: fieldTiltIndex,
    field_tilt_label: getMetricLabel(fieldTiltIndex),
    directness_index: directnessIndex,
    directness_label: getMetricLabel(directnessIndex),
    progressive_threat_index: progressiveThreatIndex,
    progressive_threat_label: getMetricLabel(progressiveThreatIndex),
    recovery_height_index: recoveryHeightIndex,
    recovery_height_label: getMetricLabel(recoveryHeightIndex),
    shot_quality_index: shotQualityIndex,
    shot_quality_label: getMetricLabel(shotQualityIndex),
    player_influence_score: playerInfluenceScore,
    player_influence_label: getMetricLabel(playerInfluenceScore),
  }
}

export function generateOpenEventInsights(
  metrics: MatchMetrics,
  selection?: {
    team?: string
    player?: string
  },
) {
  const scope = selection?.player && selection.player !== 'Todos'
    ? 'El jugador seleccionado'
    : selection?.team && selection.team !== 'Todos'
      ? 'El equipo seleccionado'
      : 'El recorte seleccionado'

  const insights: string[] = []

  if (metrics.total_events === 0) {
    return ['No se detectaron eventos para los filtros seleccionados.']
  }

  insights.push(`${scope} participó en ${metrics.total_events} eventos registrados.`)
  insights.push(
    `Se registraron ${metrics.total_passes} pases y ${metrics.total_shots} remates en el recorte analizado.`,
  )
  insights.push(
    `Se detectaron ${metrics.progressive_actions} acciones progresivas y ${metrics.final_third_actions} acciones en último tercio.`,
  )

  if ((metrics.field_tilt_index ?? 0) >= 70) {
    insights.push(
      'El equipo seleccionado mostró un dominio territorial alto, señal de presencia sostenida en campo rival.',
    )
  }

  if ((metrics.directness_index ?? 0) >= 65) {
    insights.push('El índice de verticalidad indica una progresión ofensiva agresiva.')
  }

  if ((metrics.progressive_threat_index ?? 0) >= 65) {
    insights.push('La amenaza progresiva se mantuvo en niveles altos durante el recorte.')
  }

  if ((metrics.recovery_height_index ?? 0) >= 60) {
    insights.push('La altura promedio de recuperación sugiere presión efectiva en zonas adelantadas.')
  }

  if (
    selection?.player &&
    selection.player !== 'Todos' &&
    (metrics.player_influence_score ?? 0) >= 70
  ) {
    insights.push('El jugador seleccionado muestra alta influencia en las acciones ofensivas del equipo.')
  }

  if (metrics.recoveries > 0 && insights.length < 5) {
    insights.push(`El volumen de recuperaciones defensivas fue de ${metrics.recoveries} acciones.`)
  }

  return insights.slice(0, 5)
}
