import { buildTimelineChartModel, formatFramesWithShare, getHomographyOverview } from './computerVisionInsights'

describe('computerVisionInsights', () => {
  it('marca la homografía como degradada cuando hay mucho warning y fallback', () => {
    const overview = getHomographyOverview({
      demo_mode: 'degraded',
      fallback_ratio: 0.14,
      invalid_ratio: 0.253,
      warn_ratio: 0.794,
      p95_reproj_error_m: 1.82,
    })

    expect(overview.badge).toBe('Degradada')
    expect(overview.title).toMatch(/homografía aplicada con alertas/i)
  })

  it('arma un modelo temporal listo para graficar', () => {
    const model = buildTimelineChartModel(
      {
        frames: [0, 50, 100],
        team1: [38, 42, 44],
        team2: [30, 31, 35],
      },
      25,
    )

    expect(model).not.toBeNull()
    expect(model?.labels).toEqual(['00:00', '00:02', '00:04'])
    expect(model?.min).toBe(30)
    expect(model?.max).toBe(44)
    expect(model?.averageTeam1).toBeCloseTo(41.33, 2)
  })

  it('formatea frames válidos con share cuando existe total', () => {
    expect(formatFramesWithShare(747, 1000)).toBe('747/1000 (74.7%)')
  })
})
