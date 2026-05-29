import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { SportmonksMatchCenterPage } from './SportmonksMatchCenterPage'
import * as matchCenterApi from '../services/matchCenterApi'

vi.mock('../services/matchCenterApi')

const mockApi = vi.mocked(matchCenterApi)

function renderPage(initialEntry = '/match-center/sportmonks/19636404') {
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route element={<SportmonksMatchCenterPage />} path="/match-center/sportmonks/:matchId" />
      </Routes>
    </MemoryRouter>,
  )
}

describe('SportmonksMatchCenterPage', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('renderiza el Match Center desde FastAPI y no muestra mapas cuando faltan coordenadas', async () => {
    mockApi.getSportmonksMatchCenter.mockResolvedValue({
      provider: 'sportmonks',
      match: {
        match_id: '19636404',
        competition: 'Liga Profesional de Futbol',
        season: '2026',
        date: '2026-05-02 19:15:00',
        status: 'FT',
        venue: {
          name: 'Estadio Unico Madre de Ciudades',
          city: 'Santiago del Estero',
        },
        home_team: {
          id: '14212',
          name: 'Central Cordoba SdE',
          score: 1,
        },
        away_team: {
          id: '587',
          name: 'Boca Juniors',
          score: 2,
        },
      },
      expected_metrics: {
        home: {
          team_name: 'Central Cordoba SdE',
          xg: 1.46,
          xgot: 2.43,
          xpts: 1.39,
          npxg: 1.46,
          xg_open_play: null,
          xg_set_play: null,
          xg_free_kicks: null,
          shooting_performance: null,
          xga: 1.44,
        },
        away: {
          team_name: 'Boca Juniors',
          xg: 1.44,
          xgot: 1.6,
          xpts: 1.37,
          npxg: 1.44,
          xg_open_play: null,
          xg_set_play: null,
          xg_free_kicks: null,
          shooting_performance: null,
          xga: 1.46,
        },
      },
      timeline: [
        {
          minute: 43,
          extra_minute: null,
          team_name: 'Boca Juniors',
          player_name: 'Alan Velasco',
          related_player_name: 'Williams Alarcon',
          event_type: 'goal',
          event_label: 'Gol',
          result: '0-1',
          description: 'Gol de Boca Juniors',
        },
      ],
      lineups: {
        home: {
          team_id: '14212',
          team_name: 'Central Cordoba SdE',
          formation: '4-4-2',
          coach: 'Omar De Felippe',
          starters: [],
          substitutes: [],
        },
        away: {
          team_id: '587',
          team_name: 'Boca Juniors',
          formation: '4-3-3',
          coach: 'Diego Martinez',
          starters: [],
          substitutes: [],
        },
      },
      team_stats: {
        home: { team_id: '14212', team_name: 'Central Cordoba SdE', stats: [] },
        away: { team_id: '587', team_name: 'Boca Juniors', stats: [] },
      },
      player_stats: [
        {
          player_id: '10',
          player_name: 'Alan Velasco',
          team_id: '587',
          team_name: 'Boca Juniors',
          position: 'FW',
          jersey_number: 10,
          is_starter: true,
          minutes_played: 87,
          rating: 7.8,
          stats: [
            { key: 'passes', label: 'Pases', value: 22 },
            { key: 'accurate_passes_percentage', label: 'Precision de pase', value: 81 },
          ],
          insights: ['El jugador registro 22 pases.'],
        },
      ],
      derived_metrics: {
        home: {
          eficacia_ofensiva: 0.68,
          rendimiento_definicion: -0.46,
          amenaza_jugada: null,
          amenaza_pelota_parada: null,
        },
        away: {
          eficacia_ofensiva: 1.39,
          rendimiento_definicion: 0.56,
          amenaza_jugada: null,
          amenaza_pelota_parada: null,
        },
      },
      insights: [
        'Boca Juniors gano 2-1 en un partido equilibrado segun goles esperados.',
        'No hay coordenadas de eventos confirmadas, por lo que no corresponden mapas tacticos espaciales.',
      ],
      data_quality: {
        level: 'media',
        has_xg: true,
        has_xgot: true,
        has_xpts: true,
        has_lineups: true,
        has_player_stats: true,
        has_team_stats: true,
        has_event_timeline: true,
        has_event_coordinates: false,
        enabled_modules: {
          match_center: true,
          expected_metrics: true,
          timeline: true,
          lineups: true,
          team_stats: true,
          player_stats: true,
          event_maps: false,
          shot_map: false,
          pass_network: false,
        },
        message:
          'Este partido tiene estadísticas, lineups, xG y eventos principales. No incluye coordenadas de eventos, por lo que los mapas tácticos espaciales no están disponibles.',
      },
    })

    renderPage()

    await waitFor(() => expect(mockApi.getSportmonksMatchCenter).toHaveBeenCalledWith('19636404'))

    expect(await screen.findByText(/sportmonks match center/i)).toBeInTheDocument()
    expect(screen.getByText(/central cordoba sde 1 - 2 boca juniors/i)).toBeInTheDocument()
    expect(screen.getByText(/cobertura de datos: media/i)).toBeInTheDocument()
    expect(screen.getByText(/timeline de eventos/i)).toBeInTheDocument()
    expect(screen.getByText(/estadísticas de jugadores/i)).toBeInTheDocument()
    expect(screen.getAllByText(/alan velasco/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/mapas tácticos/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/no disponibles/i).length).toBeGreaterThan(0)
  })

  it('muestra estado de error si falla la carga', async () => {
    mockApi.getSportmonksMatchCenter.mockRejectedValue(new Error('No se pudo cargar el partido.'))

    renderPage()

    expect(await screen.findByText(/no se pudo cargar el partido/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /reintentar/i })).toBeInTheDocument()
  })
})
