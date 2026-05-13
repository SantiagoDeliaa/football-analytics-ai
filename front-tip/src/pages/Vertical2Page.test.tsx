import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { EventDataProvider } from '../app/EventDataContext'
import { Vertical2Page } from './Vertical2Page'
import * as api from '../services/eventDataApi'

vi.mock('../services/eventDataApi')

const mockApi = vi.mocked(api)

function buildCanonicalEvents(count: number, team = 'Argentina') {
  return Array.from({ length: count }, (_, index) => ({
    event_id: `event-${index}`,
    match_id: '99',
    team_name: team,
    player_name: `Jugador ${index % 3}`,
    minute: index,
    second: 0,
    event_type: index % 10 === 0 ? 'Shot' : 'Pass',
    x: 82,
    y: 40,
    end_x: 95,
    end_y: 42,
    outcome: 'won',
    progressive: index % 2 === 0,
    under_pressure: index % 4 === 0,
    xG: index % 10 === 0 ? 0.13 : 0,
  }))
}

describe('Vertical2Page', () => {
  beforeEach(() => {
    window.localStorage.clear()
    mockApi.fetchCompetitions.mockResolvedValue({
      competitions: [
        {
          competition_id: 1,
          season_id: 10,
          competition_name: 'Liga',
          season_name: '2025',
          display_name: 'Liga - 2025',
        },
      ],
      status: { source: 'api' },
    })
    mockApi.fetchMatches.mockResolvedValue({
      matches: [{ match_id: 99, display_name: 'A vs B' }],
      status: { source: 'api' },
    })
    mockApi.fetchProcessedHistory.mockResolvedValue([])
    mockApi.loadEventData.mockResolvedValue({
      match_id: '99',
      competition_name: 'Liga',
      match_label: 'A vs B',
      canonical_events: buildCanonicalEvents(100),
      insights: ['Insight demo'],
      metrics: {
        total_events: 100,
        total_passes: 40,
        total_shots: 10,
        progressive_actions: 7,
        final_third_actions: 12,
        recoveries: 8,
        total_under_pressure: 5,
        total_xg: 1.3,
      },
      raw_events_count: 100,
      used_fallback_events: false,
      events_status_message: '',
    })
    mockApi.uploadPdfReport.mockResolvedValue({
      normalized_payload: {},
      insights: [],
      metrics: {
        total_events: 0,
        total_passes: 0,
        total_shots: 0,
        progressive_actions: 0,
        final_third_actions: 0,
        recoveries: 0,
        total_under_pressure: 0,
        total_xg: 0,
      },
    })
    mockApi.loadProcessedHistoryEntry.mockResolvedValue({
      match_id: '99',
      competition_name: 'Liga',
      match_label: 'A vs B',
      canonical_events: buildCanonicalEvents(100),
      insights: ['Insight backend'],
      metrics: {
        total_events: 100,
        total_passes: 40,
        total_shots: 10,
        progressive_actions: 7,
        final_third_actions: 12,
        recoveries: 8,
        total_under_pressure: 5,
        total_xg: 1.3,
      },
      raw_events_count: 100,
      used_fallback_events: false,
      events_status_message: '',
    })
  })

  it('carga métricas API al presionar "Cargar datos"', async () => {
    render(
      <MemoryRouter initialEntries={['/vertical2']}>
        <EventDataProvider>
          <Routes>
            <Route element={<Vertical2Page />} path="/vertical2" />
            <Route element={<Vertical2Page />} path="/vertical2/match/:matchId" />
          </Routes>
        </EventDataProvider>
      </MemoryRouter>,
    )

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalledWith('StatsBomb Open Data'))
    await waitFor(() =>
      expect(mockApi.fetchMatches).toHaveBeenCalledWith('StatsBomb Open Data', 1, 10),
    )

    fireEvent.click(screen.getByRole('button', { name: /cargar datos/i }))

    await waitFor(() => expect(mockApi.loadEventData).toHaveBeenCalled())
    expect(await screen.findByText(/eventos analizados/i)).toBeInTheDocument()
    expect(screen.getAllByText('100').length).toBeGreaterThan(0)
  })

  it('recupera un partido desde historial local al entrar por ruta dinámica', async () => {
    window.localStorage.setItem(
      'tip:event-history',
      JSON.stringify([
        {
          id: 'history-1',
          provider: 'StatsBomb Open Data',
          competition_name: 'Liga',
          match_label: 'A vs B',
          saved_at: new Date().toISOString(),
          selection: {
            team: 'Argentina',
            player: 'Todos',
          },
          result: {
            match_id: '99',
            competition_name: 'Liga',
            match_label: 'A vs B',
            canonical_events: buildCanonicalEvents(22),
            insights: ['Insight local'],
            metrics: {
              total_events: 22,
              total_passes: 10,
              total_shots: 3,
              progressive_actions: 4,
              final_third_actions: 5,
              recoveries: 2,
              total_under_pressure: 6,
              total_xg: 0.6,
            },
            raw_events_count: 22,
            used_fallback_events: false,
            events_status_message: '',
          },
        },
      ]),
    )

    render(
      <MemoryRouter initialEntries={['/vertical2/match/99']}>
        <EventDataProvider>
          <Routes>
            <Route element={<Vertical2Page />} path="/vertical2" />
            <Route element={<Vertical2Page />} path="/vertical2/match/:matchId" />
          </Routes>
        </EventDataProvider>
      </MemoryRouter>,
    )

    expect(await screen.findByText(/resumen del partido/i)).toBeInTheDocument()
    expect(screen.getAllByText('22').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /partido activo/i })).toBeInTheDocument()
  })

  it('permite cargar un partido desde historial backend', async () => {
    mockApi.fetchProcessedHistory.mockResolvedValue([
      {
        provider: 'StatsBomb Open Data',
        match_id: '99',
        competition_name: 'Liga',
        season_name: '2025',
        home_team: 'Argentina',
        away_team: 'Francia',
        match_date: '2025-01-01',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ])

    render(
      <MemoryRouter initialEntries={['/vertical2']}>
        <EventDataProvider>
          <Routes>
            <Route element={<Vertical2Page />} path="/vertical2" />
            <Route element={<Vertical2Page />} path="/vertical2/match/:matchId" />
          </Routes>
        </EventDataProvider>
      </MemoryRouter>,
    )

    expect(await screen.findByText(/historial backend/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /cargar backend/i }))

    await waitFor(() => expect(mockApi.loadProcessedHistoryEntry).toHaveBeenCalled())
    expect(mockApi.loadProcessedHistoryEntry).toHaveBeenCalledWith('StatsBomb Open Data', '99')
  })

  it('recarga filtros al cambiar a API-Football', async () => {
    mockApi.fetchCompetitions.mockResolvedValueOnce({
      competitions: [
        {
          competition_id: 10,
          season_id: 2024,
          competition_name: 'Liga Profesional',
          season_name: '2024',
          display_name: 'Liga Profesional (Argentina) - 2024',
        },
      ],
      status: { source: 'api' },
    })
    mockApi.fetchMatches.mockResolvedValueOnce({
      matches: [
        {
          match_id: 555,
          display_name: 'River Plate vs Boca Juniors — 2024-05-12',
          home_team: 'River Plate',
          away_team: 'Boca Juniors',
          match_date: '2024-05-12',
        },
      ],
      status: { source: 'api' },
    })

    render(
      <MemoryRouter initialEntries={['/vertical2']}>
        <EventDataProvider>
          <Routes>
            <Route element={<Vertical2Page />} path="/vertical2" />
            <Route element={<Vertical2Page />} path="/vertical2/match/:matchId" />
          </Routes>
        </EventDataProvider>
      </MemoryRouter>,
    )

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalledWith('StatsBomb Open Data'))
    fireEvent.change(screen.getByLabelText(/proveedor/i), { target: { value: 'API-Football' } })

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenLastCalledWith('API-Football'))
  })
})
