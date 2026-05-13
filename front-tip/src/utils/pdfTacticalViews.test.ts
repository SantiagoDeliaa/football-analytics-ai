import {
  buildPdfPitchFigure,
  buildPdfPitchViews,
  buildPdfRadarFigure,
  buildPdfRadarMetrics,
} from './pdfTacticalViews'

const samplePayload = {
  attack: {
    signals: {
      crosses: 6,
      'box entries': 4,
      'final third': 8,
      shots: 5,
      xg: 1.4,
    },
  },
  defense: {
    signals: {
      duels: 7,
      pressing: 4,
      recoveries: 5,
    },
  },
  transitions: {
    signals: {
      regain: 3,
      turnover: 2,
      counter: 2,
      transition: 4,
      'direct attack': 1,
    },
  },
  build_up: {
    signals: {
      progression: 6,
    },
  },
}

describe('pdfTacticalViews', () => {
  it('construye métricas radar dentro del rango esperado', () => {
    const metrics = buildPdfRadarMetrics(samplePayload)

    expect(metrics).toHaveLength(5)
    expect(metrics.every((metric) => metric.value >= 0 && metric.value <= 100)).toBe(true)
  })

  it('arma una figura radar con una única serie polar', () => {
    const figure = buildPdfRadarFigure(samplePayload)

    expect(figure.data).toHaveLength(1)
    expect(figure.data[0].type).toBe('scatterpolar')
  })

  it('arma la cancha con overlays cuando hay señal', () => {
    const attackView = buildPdfPitchViews(samplePayload)[0]
    const figure = buildPdfPitchFigure(attackView)

    expect(figure.data).toHaveLength(1)
    expect(figure.layout.shapes).toBeDefined()
    expect(Array.isArray(figure.layout.shapes)).toBe(true)
  })

  it('muestra un mensaje cuando no hay señal zonal', () => {
    const emptyView = buildPdfPitchViews({})[0]
    const figure = buildPdfPitchFigure(emptyView)

    expect(figure.data).toHaveLength(0)
    expect(figure.layout.annotations?.[0]?.text).toContain('Sin señales zonales suficientes')
  })
})
