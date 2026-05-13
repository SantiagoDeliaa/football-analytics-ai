import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Vertical1Page } from './Vertical1Page'
import * as computerVisionApi from '../services/computerVisionApi'

vi.mock('../services/computerVisionApi')

const mockComputerVisionApi = vi.mocked(computerVisionApi)

describe('Vertical1Page', () => {
  beforeEach(() => {
    const result = {
      source: 'mock' as const,
      status_message: 'Demo',
      video_name: 'demo.mp4',
      duration_seconds: 12.5,
      total_frames: 312,
      fps: 25,
      health_summary: {
        demo_mode: 'stable' as const,
        fallback_ratio: 0.12,
        p95_reproj_error_m: 1.1,
        p95_churn_ratio: 0.25,
        p95_max_speed_mps: 8.2,
        ball_detected_ratio: 0.44,
      },
      formations: {
        team1: { most_common: '4-3-3' },
        team2: { most_common: '4-4-2' },
      },
      metrics: {
        team1: {
          pressure_height: { mean: 42, min: 36, max: 48 },
          offensive_width: { mean: 34, min: 29, max: 39 },
          compactness: { mean: 812, min: 760, max: 840 },
        },
        team2: {
          pressure_height: { mean: 35, min: 31, max: 40 },
          offensive_width: { mean: 29, min: 24, max: 33 },
          compactness: { mean: 748, min: 700, max: 780 },
        },
      },
      timeline: {
        pressure_height: {
          frames: [0, 60, 120],
          team1: [38, 42, 44],
          team2: [31, 34, 36],
        },
      },
      possession: {
        team1_pct: 52,
        team2_pct: 40,
        contested_pct: 8,
      },
      scouting: {
        confidence: {
          team1: { label: 'Alta', score: 82 },
          team2: { label: 'Media', score: 68 },
        },
        bullets: {
          team1: ['Presión alta'],
          team2: ['Bloque medio'],
        },
      },
      exports: {
        json: true,
        csv: true,
        pdf: false,
      },
      warnings: ['Warning demo'],
      interpretation: ['Interpretación demo'],
    }

    mockComputerVisionApi.createComputerVisionJob.mockResolvedValue({
      job_id: 'job-1',
      status: 'queued',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      video_name: 'clip.mp4',
      result: null,
      error: null,
    })
    mockComputerVisionApi.getComputerVisionJob.mockResolvedValue({
      job_id: 'job-1',
      status: 'completed',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      video_name: 'clip.mp4',
      result,
      error: null,
    })

    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:demo')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('valida que haya una fuente antes de procesar', async () => {
    render(
      <MemoryRouter>
        <Vertical1Page />
      </MemoryRouter>,
    )

    fireEvent.click(screen.getByRole('button', { name: /procesar video/i }))

    expect(await screen.findByText(/debés seleccionar un archivo de video/i)).toBeInTheDocument()
    expect(mockComputerVisionApi.createComputerVisionJob).not.toHaveBeenCalled()
  })

  it('procesa un video y renderiza tabs de resultados', async () => {
    render(
      <MemoryRouter>
        <Vertical1Page />
      </MemoryRouter>,
    )

    const file = new File(['video-demo'], 'clip.mp4', { type: 'video/mp4' })
    const input = screen.getByLabelText(/archivo de video/i)
    await userEvent.upload(input, file)

    fireEvent.click(screen.getByRole('button', { name: /procesar video/i }))

    await waitFor(() => expect(mockComputerVisionApi.createComputerVisionJob).toHaveBeenCalled())
    await waitFor(() => expect(mockComputerVisionApi.getComputerVisionJob).toHaveBeenCalled())
    expect(await screen.findByText(/duración analizada/i)).toBeInTheDocument()
    expect(screen.getByText('312')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /estadísticas/i }))
    expect(await screen.findByText(/comparativa táctica/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /exportar/i }))
    expect(await screen.findByText(/descargar json/i)).toBeInTheDocument()
  })
})
