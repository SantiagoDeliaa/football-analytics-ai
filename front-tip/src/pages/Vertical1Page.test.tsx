import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { Vertical1Page } from './Vertical1Page'
import * as computerVisionApi from '../services/computerVisionApi'

vi.mock('../services/computerVisionApi')

const mockComputerVisionApi = vi.mocked(computerVisionApi)

describe('Vertical1Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
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
        total_frames_analyzed: 120,
        top_possessors: [
          { tracker_id: 8, frames: 40, team: 'team1' },
          { tracker_id: 4, frames: 27, team: 'team2' },
        ],
        passes: {
          team1_passes: 12,
          team2_passes: 8,
          turnovers: 3,
          total: 23,
        },
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
      quality_control: {
        confidence_grade_team1: 'Alta',
        confidence_grade_team2: 'Media',
        warnings: ['Warning demo'],
      },
      speed_distance: {
        per_team: {
          team1: {
            total_distance_m: 950,
            avg_distance_m: 118.7,
            max_speed_kmh: 28.4,
            player_count: 8,
            total_sprints: 11,
            total_sprint_distance_m: 121.2,
          },
          team2: {
            total_distance_m: 880,
            avg_distance_m: 110,
            max_speed_kmh: 27.1,
            player_count: 8,
            total_sprints: 8,
            total_sprint_distance_m: 96.5,
          },
        },
        per_player: {
          '8': {
            distance_m: 160.2,
            max_speed_kmh: 28.4,
            team: 'team1',
            sprint_count: 3,
            sprint_distance_m: 31.2,
            intensity_zones_m: { walk: 32, jog: 74, run: 38, sprint: 16.2 },
          },
        },
      },
      scouting_heatmaps: {
        team1: { total_samples: 24, downsampled_shape: [20, 20] },
        team2: { total_samples: 21, downsampled_shape: [20, 20] },
        bins_shape: [26, 17],
      },
      artifacts: {
        video_url: '/api/static/computer-vision/demo_processed.mp4',
        stats_json_url: '/api/static/computer-vision/demo_stats.json',
        pdf_url: null,
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
      processing_id: 'cv-1',
      video_name: 'clip.mp4',
      result,
      error: null,
    })
    mockComputerVisionApi.getComputerVisionHistory.mockResolvedValue({ items: [] })
    mockComputerVisionApi.getComputerVisionHistoryItem.mockResolvedValue({
      metadata: {
        processing_id: 'cv-1',
        job_id: 'job-1',
        source_mode: 'upload',
        source_label: 'clip.mp4',
        video_name: 'clip.mp4',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        video_url: '/api/static/computer-vision/demo_processed.mp4',
        stats_json_url: '/api/static/computer-vision/demo_stats.json',
      },
      result,
    })
    mockComputerVisionApi.deleteComputerVisionHistoryItem.mockResolvedValue({
      ok: true,
      message: 'Se eliminó el procesamiento guardado cv-1 del historial local.',
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

  it('renderiza el historial vacío de procesamientos', async () => {
    render(
      <MemoryRouter>
        <Vertical1Page />
      </MemoryRouter>,
    )

    expect(await screen.findByText(/historial de procesamientos/i)).toBeInTheDocument()
    expect(await screen.findByText(/todavía no hay procesamientos guardados/i)).toBeInTheDocument()
    expect(mockComputerVisionApi.getComputerVisionHistory).toHaveBeenCalled()
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
    await waitFor(() => expect(mockComputerVisionApi.getComputerVisionHistory).toHaveBeenCalled())
    expect(await screen.findByText(/duración analizada/i)).toBeInTheDocument()
    expect(screen.getByText('312')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /estadísticas/i }))
    expect(await screen.findByText(/comparativa táctica/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /exportar/i }))
    expect(await screen.findByText(/descargar json/i)).toBeInTheDocument()
    expect(screen.getByText(/video procesado/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /scouting/i }))
    expect(await screen.findByText(/alertas y señal operativa/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /posesión/i }))
    expect(await screen.findByText(/posesión y físico/i)).toBeInTheDocument()
    expect(screen.getByText(/top físico por jugador/i)).toBeInTheDocument()
  })

  it('exige el archivo cuando se elige modelo custom de jugadores', async () => {
    render(
      <MemoryRouter>
        <Vertical1Page />
      </MemoryRouter>,
    )

    const file = new File(['video-demo'], 'clip.mp4', { type: 'video/mp4' })
    await userEvent.upload(screen.getByLabelText(/archivo de video/i), file)
    fireEvent.change(screen.getByRole('combobox', { name: /^modelo de jugadores$/i }), {
      target: { value: 'custom' },
    })
    expect(await screen.findByText(/todavía no se cargó ningún archivo/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /procesar video/i }))

    expect(await screen.findByText(/debés subir un modelo custom de jugadores/i)).toBeInTheDocument()
    expect(mockComputerVisionApi.createComputerVisionJob).not.toHaveBeenCalled()
  })

  it('carga un resultado guardado desde el historial', async () => {
    mockComputerVisionApi.getComputerVisionHistory.mockResolvedValue({
      items: [
        {
          processing_id: 'cv-1',
          job_id: 'job-1',
          source_mode: 'upload',
          source_label: 'clip.mp4',
          video_name: 'clip.mp4',
          status: 'completed',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          video_url: '/api/static/computer-vision/demo_processed.mp4',
          stats_json_url: '/api/static/computer-vision/demo_stats.json',
        },
      ],
    })

    render(
      <MemoryRouter>
        <Vertical1Page />
      </MemoryRouter>,
    )

    fireEvent.click(await screen.findByRole('button', { name: /cargar resultado/i }))

    await waitFor(() => expect(mockComputerVisionApi.getComputerVisionHistoryItem).toHaveBeenCalledWith('cv-1'))
    expect(await screen.findByText(/se cargó el procesamiento guardado para clip\.mp4/i)).toBeInTheDocument()
    expect(await screen.findByText(/duración analizada/i)).toBeInTheDocument()
  })

  it('elimina un procesamiento guardado con confirmación', async () => {
    mockComputerVisionApi.getComputerVisionHistory.mockResolvedValue({
      items: [
        {
          processing_id: 'cv-1',
          job_id: 'job-1',
          source_mode: 'upload',
          source_label: 'clip.mp4',
          video_name: 'clip.mp4',
          status: 'completed',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          video_url: '/api/static/computer-vision/demo_processed.mp4',
          stats_json_url: '/api/static/computer-vision/demo_stats.json',
        },
      ],
    })

    render(
      <MemoryRouter>
        <Vertical1Page />
      </MemoryRouter>,
    )

    fireEvent.click(await screen.findByRole('button', { name: /^eliminar$/i }))
    fireEvent.click(screen.getByRole('button', { name: /eliminar definitivamente/i }))

    await waitFor(() => expect(mockComputerVisionApi.deleteComputerVisionHistoryItem).toHaveBeenCalledWith('cv-1'))
    expect(await screen.findByText(/se eliminó el procesamiento guardado cv-1/i)).toBeInTheDocument()
  })
})
