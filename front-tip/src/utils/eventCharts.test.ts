import {
  buildEventTypeDistribution,
  buildMetricChartItems,
  buildPitchScatterPoints,
  buildPlayerRadarMetrics,
  buildPressureBreakdown,
  buildProgressiveBreakdown,
  parseMetricNumber,
} from './eventCharts'

const events = [
  {
    event_id: '1',
    match_id: '99',
    team_name: 'Argentina',
    player_name: 'Jugador 1',
    minute: 1,
    second: 0,
    event_type: 'Pass',
    x: 15,
    y: 25,
    progressive: true,
    under_pressure: true,
    xG: 0,
  },
  {
    event_id: '2',
    match_id: '99',
    team_name: 'Argentina',
    player_name: 'Jugador 2',
    minute: 2,
    second: 0,
    event_type: 'Shot',
    x: 100,
    y: 40,
    progressive: false,
    under_pressure: false,
    xG: 0.2,
  },
  {
    event_id: '3',
    match_id: '99',
    team_name: 'Argentina',
    player_name: 'Jugador 3',
    minute: 3,
    second: 0,
    event_type: 'Pass',
    x: 60,
    y: 10,
    progressive: true,
    under_pressure: false,
    xG: 0,
  },
] as const

describe('eventCharts', () => {
  it('agrupa distribución por tipo de evento', () => {
    const distribution = buildEventTypeDistribution([...events])

    expect(distribution.labels[0]).toBe('Pases')
    expect(distribution.values[0]).toBe(2)
    expect(distribution.labels[1]).toBe('Remates')
  })

  it('normaliza puntos válidos para scatter de cancha', () => {
    const points = buildPitchScatterPoints([...events], 2)

    expect(points).toEqual([
      { x: 15, y: 25 },
      { x: 100, y: 40 },
    ])
  })

  it('arma breakdowns de progresión y presión', () => {
    expect(buildProgressiveBreakdown([...events]).values).toEqual([2, 1])
    expect(buildPressureBreakdown([...events]).values).toEqual([1, 2])
  })

  it('arma métricas radar de jugador', () => {
    const radar = buildPlayerRadarMetrics({
      total_events: 12,
      total_passes: 8,
      total_shots: 2,
      progressive_actions: 4,
      final_third_actions: 5,
      recoveries: 3,
      total_under_pressure: 2,
      total_xg: 0.4,
      player_influence_score: 78,
      directness_index: 65,
      progressive_threat_index: 72,
      recovery_height_index: 58,
      shot_quality_index: 44,
    })

    expect(radar.labels).toHaveLength(5)
    expect(radar.values).toEqual([78, 65, 72, 58, 44])
  })

  it('parsea valores numéricos y arma items para charts de KPIs', () => {
    expect(parseMetricNumber('1.30')).toBe(1.3)
    expect(parseMetricNumber('82%')).toBe(82)
    expect(parseMetricNumber('No aplica')).toBeNull()

    const chartItems = buildMetricChartItems([
      { title: 'Eventos', value: '100' },
      { title: 'xG total', value: '1.30' },
      { title: 'Estado', value: 'No aplica' },
    ])

    expect(chartItems).toEqual([
      { label: 'Eventos', value: 100 },
      { label: 'xG total', value: 1.3 },
    ])
  })
})
