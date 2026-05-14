import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { AiCoachChatProvider } from '../app/AiCoachChatContext'
import { EventDataProvider, eventDataReducer, initialEventDataState } from '../app/EventDataContext'
import { AiCoachChatWidget } from '../components/coach/AiCoachChatWidget'
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

function renderVertical2Page(initialEntries = ['/vertical2']) {
  render(
    <MemoryRouter initialEntries={initialEntries}>
      <AiCoachChatProvider>
        <EventDataProvider>
          <AiCoachChatWidget />
          <Routes>
            <Route element={<Vertical2Page />} path="/vertical2" />
            <Route element={<Vertical2Page />} path="/vertical2/match/:matchId" />
          </Routes>
        </EventDataProvider>
      </AiCoachChatProvider>
    </MemoryRouter>,
  )
}

describe('Vertical2Page', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    window.localStorage.clear()
    window.sessionStorage.clear()
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
    mockApi.fetchApiFootballCountries.mockResolvedValue({
      countries: [{ name: 'Argentina', code: 'AR', flag: '', display_name: 'Argentina' }],
      status: { source: 'api' },
    })
    mockApi.fetchApiFootballLeagues.mockResolvedValue({
      leagues: [
        {
          league_id: 10,
          league_name: 'Liga Profesional',
          country_name: 'Argentina',
          type: 'League',
          logo: '',
          seasons: [2024, 2023],
          current_season: 2024,
          display_name: 'Liga Profesional (Argentina)',
        },
      ],
      status: { source: 'api' },
    })
    mockApi.fetchApiFootballFixtures.mockResolvedValue({
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
    mockApi.fetchMatches.mockResolvedValue({
      matches: [{ match_id: 99, display_name: 'A vs B' }],
      status: { source: 'api' },
    })
    mockApi.fetchProcessedHistory.mockResolvedValue([])
    mockApi.deleteProcessedHistoryEntry.mockResolvedValue({
      ok: true,
      message: 'Partido eliminado del historial persistido.',
    })
    mockApi.fetchCoachStatus.mockResolvedValue({
      configured: true,
      api_key_configured: true,
      model_configured: true,
      base_url_configured: true,
      model: 'gpt-4o-mini',
      base_url: 'https://api.openai.com/v1/chat/completions',
      message: 'AI Tactical Coach configurado correctamente.',
    })
    mockApi.generateCoachDiagnosis.mockResolvedValue({
      ok: true,
      diagnosis: 'Diagnóstico táctico de prueba.',
      error: '',
      suggested_questions: ['¿Dónde generó más peligro?'],
    })
    mockApi.askCoachQuestion.mockResolvedValue({
      ok: true,
      answer: 'Respuesta táctica de prueba.',
      error: '',
      suggested_questions: ['¿Qué debería corregir el cuerpo técnico?'],
    })
    mockApi.loadEventData.mockResolvedValue({
      provider: 'StatsBomb Open Data',
      match_id: '99',
      competition_name: 'Liga',
      season_name: '2025',
      match_label: 'A vs B',
      home_team: 'A',
      away_team: 'B',
      match_date: '2025-01-01',
      canonical_events: buildCanonicalEvents(100),
      insights: ['Insight demo'],
      raw_payload: [{ type: 'Pass', minute: 1, team: { name: 'Argentina' } }],
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
      provider: 'StatsBomb Open Data',
      match_id: '99',
      competition_name: 'Liga',
      season_name: '2025',
      match_label: 'A vs B',
      home_team: 'Argentina',
      away_team: 'Francia',
      match_date: '2025-01-01',
      canonical_events: buildCanonicalEvents(100),
      insights: ['Insight backend'],
      raw_payload: [{ type: 'Pass', minute: 1, team: { name: 'Argentina' } }],
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
    renderVertical2Page()

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalledWith('StatsBomb Open Data'))
    await waitFor(() =>
      expect(mockApi.fetchMatches).toHaveBeenCalledWith('StatsBomb Open Data', 1, 10),
    )

    fireEvent.click(screen.getByRole('button', { name: /cargar datos/i }))

    await waitFor(() => expect(mockApi.loadEventData).toHaveBeenCalled())
    expect(screen.queryByText(/vertical 2/i)).not.toBeInTheDocument()
    expect(await screen.findByText(/eventos analizados/i)).toBeInTheDocument()
    expect(screen.getAllByText('100').length).toBeGreaterThan(0)
    expect(screen.getByText(/panel técnico del provider/i)).toBeInTheDocument()
    expect(screen.getByText(/timeline raw de statsbomb/i)).toBeInTheDocument()
  })

  it('recupera un partido desde historial local al cargarlo manualmente', async () => {
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
            provider: 'StatsBomb Open Data',
            match_id: '99',
            competition_name: 'Liga',
            season_name: '2025',
            match_label: 'A vs B',
            home_team: 'A',
            away_team: 'B',
            match_date: '2025-01-01',
            canonical_events: buildCanonicalEvents(22),
            insights: ['Insight local'],
            raw_payload: [{ type: 'Pass', minute: 1, team: { name: 'Argentina' } }],
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

    renderVertical2Page()

    expect(await screen.findByText(/historial local del navegador/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /desde historial local/i }))

    expect(await screen.findByText(/resumen del partido/i)).toBeInTheDocument()
    expect(screen.getAllByText('22').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /activo/i })).toBeInTheDocument()
  })

  it('muestra una sugerencia contextual cuando el partido ya existe en historial', async () => {
    window.localStorage.setItem(
      'tip:event-history',
      JSON.stringify([
        {
          id: 'history-1',
          provider: 'StatsBomb Open Data',
          competition_name: 'Liga',
          match_label: 'A vs B',
          saved_at: new Date().toISOString(),
          selection: { team: 'Todos', player: 'Todos' },
          result: {
            provider: 'StatsBomb Open Data',
            match_id: '99',
            competition_name: 'Liga',
            season_name: '2025',
            match_label: 'A vs B',
            home_team: 'A',
            away_team: 'B',
            match_date: '2025-01-01',
            canonical_events: buildCanonicalEvents(22),
            insights: ['Insight local'],
            raw_payload: [{ type: 'Pass', minute: 1, team: { name: 'A' } }],
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

    renderVertical2Page()

    expect(await screen.findByText(/partido ya disponible en historial/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /cargar historial local/i })).toBeInTheDocument()
  })

  it('resetea equipo y jugador al cambiar de partido para no arrastrar filtros viejos', () => {
    const stateWithSelection = {
      ...initialEventDataState,
      selectedTeam: 'Bayer Leverkusen',
      selectedPlayer: 'Florian Wirtz',
      result: {
        provider: 'StatsBomb Open Data' as const,
        match_id: '3895302',
        competition_name: 'Bundesliga',
        season_name: '2024',
        match_label: 'Bayer Leverkusen vs Werder Bremen',
        home_team: 'Bayer Leverkusen',
        away_team: 'Werder Bremen',
        match_date: '2024-02-10',
        canonical_events: [
          ...buildCanonicalEvents(8, 'Bayer Leverkusen'),
          ...buildCanonicalEvents(8, 'Werder Bremen'),
        ],
        insights: ['Insight local'],
        metrics: {
          total_events: 16,
          total_passes: 10,
          total_shots: 2,
          progressive_actions: 4,
          final_third_actions: 5,
          recoveries: 2,
          total_under_pressure: 3,
          total_xg: 0.5,
        },
        raw_events_count: 16,
        used_fallback_events: false,
        events_status_message: '',
      },
    }

    const nextState = eventDataReducer(stateWithSelection, {
      type: 'setSelectedMatch',
      match: {
        match_id: 555,
        display_name: 'Boca Juniors vs Racing Club — 2024-03-11',
        home_team: 'Boca Juniors',
        away_team: 'Racing Club',
        match_date: '2024-03-11',
      },
    })

    expect(nextState.selectedTeam).toBe('Todos')
    expect(nextState.selectedPlayer).toBe('Todos')
    expect(nextState.result).toBeUndefined()
  })

  it('resetea jugador al cambiar de equipo para recalcular opciones limpias', () => {
    const stateWithSelection = {
      ...initialEventDataState,
      selectedTeam: 'Bayer Leverkusen',
      selectedPlayer: 'Florian Wirtz',
    }

    const nextState = eventDataReducer(stateWithSelection, {
      type: 'setTeam',
      team: 'Werder Bremen',
    })

    expect(nextState.selectedTeam).toBe('Werder Bremen')
    expect(nextState.selectedPlayer).toBe('Todos')
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

    renderVertical2Page()

    expect(await screen.findByText(/historial local persistido/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /historial persistido/i }))

    await waitFor(() => expect(mockApi.loadProcessedHistoryEntry).toHaveBeenCalled())
    expect(mockApi.loadProcessedHistoryEntry).toHaveBeenCalledWith('StatsBomb Open Data', '99')
  })

  it('permite eliminar un partido del historial persistido con confirmación', async () => {
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

    renderVertical2Page()

    expect(await screen.findByText(/historial local persistido/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /^eliminar$/i }))
    fireEvent.click(screen.getByRole('button', { name: /eliminar definitivamente/i }))

    await waitFor(() =>
      expect(mockApi.deleteProcessedHistoryEntry).toHaveBeenCalledWith('StatsBomb Open Data', '99'),
    )
    expect(await screen.findByText(/partido eliminado del historial persistido/i)).toBeInTheDocument()
  })

  it('carga el flujo específico de API-Football al cambiar de provider', async () => {
    renderVertical2Page()

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalledWith('StatsBomb Open Data'))
    fireEvent.change(screen.getByLabelText(/proveedor/i), { target: { value: 'API-Football' } })

    await waitFor(() => expect(mockApi.fetchApiFootballCountries).toHaveBeenCalled())
    await waitFor(() =>
      expect(mockApi.fetchApiFootballLeagues).toHaveBeenCalledWith({ country: 'Argentina' }),
    )
    expect(await screen.findByLabelText(/^país$/i)).toHaveValue('Argentina')
    expect(screen.getByLabelText(/^liga$/i)).toHaveValue('10')
    expect(screen.getByLabelText(/temporada/i)).toHaveValue('2024')
  })

  it('permite buscar partidos y cargar datos con API-Football', async () => {
    mockApi.loadEventData.mockResolvedValueOnce({
      provider: 'API-Football',
      match_id: '555',
      competition_name: 'Liga Profesional',
      season_name: '2024',
      match_label: 'River Plate vs Boca Juniors — 2024-05-12',
      home_team: 'River Plate',
      away_team: 'Boca Juniors',
      match_date: '2024-05-12',
      canonical_events: buildCanonicalEvents(24, 'River Plate'),
      insights: ['Insight API-Football'],
      raw_payload: {
        events: [{ type: 'Goal', time: { elapsed: 15 }, team: { name: 'River Plate' }, player: { name: 'Borja' } }],
        lineups: [{ team: { name: 'River Plate' } }],
        statistics: [{ team: { name: 'River Plate' }, statistics: [{ type: 'Shots on Goal', value: 3 }] }],
        players: [{ team: { name: 'River Plate' }, players: [{ player: { name: 'Borja', age: 31 } }] }],
      },
      metrics: {
        total_events: 24,
        total_passes: 10,
        total_shots: 4,
        progressive_actions: 3,
        final_third_actions: 6,
        recoveries: 5,
        total_under_pressure: 2,
        total_xg: 0,
      },
      raw_events_count: 24,
      used_fallback_events: false,
      events_status_message: '',
    })

    renderVertical2Page()

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalled())
    fireEvent.change(screen.getByLabelText(/proveedor/i), { target: { value: 'API-Football' } })

    await waitFor(() => expect(mockApi.fetchApiFootballCountries).toHaveBeenCalled())
    await waitFor(() => expect(mockApi.fetchApiFootballLeagues).toHaveBeenCalled())

    fireEvent.click(await screen.findByRole('button', { name: /buscar partidos/i }))

    await waitFor(() =>
      expect(mockApi.fetchApiFootballFixtures).toHaveBeenCalledWith({ leagueId: 10, season: 2024 }),
    )
    expect(await screen.findByText(/river plate vs boca juniors — 2024-05-12/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /cargar datos/i }))

    await waitFor(() =>
      expect(mockApi.loadEventData).toHaveBeenCalledWith(
        expect.objectContaining({
          provider: 'API-Football',
          matchId: 555,
          competitionName: 'Liga Profesional',
          seasonName: '2024',
        }),
      ),
    )
    expect(await screen.findByText(/resumen del partido/i)).toBeInTheDocument()
    expect(screen.getByText(/panel técnico del provider/i)).toBeInTheDocument()
    expect(screen.getByText(/tabla técnica de estadísticas/i)).toBeInTheDocument()
    expect(screen.getByText(/vistas raw/i)).toBeInTheDocument()
  })

  it('permite generar diagnóstico y hacer preguntas al AI Coach', async () => {
    renderVertical2Page()

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalled())
    await waitFor(() =>
      expect(mockApi.fetchMatches).toHaveBeenCalledWith('StatsBomb Open Data', 1, 10),
    )
    fireEvent.click(screen.getByRole('button', { name: /cargar datos/i }))

    await waitFor(() => expect(mockApi.fetchCoachStatus).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: /abrir ai coach/i }))
    await waitFor(() => expect(screen.getByText(/a vs b \| statsbomb open data/i)).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /generar diagnóstico táctico/i }))

    await waitFor(() => expect(mockApi.generateCoachDiagnosis).toHaveBeenCalled())
    expect(await screen.findByText(/diagnóstico táctico de prueba\./i)).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText(/preguntale algo al ai coach/i), {
      target: { value: '¿Dónde estuvo la mayor amenaza?' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^preguntar$/i }))

    await waitFor(() => expect(mockApi.askCoachQuestion).toHaveBeenCalled())
    expect(await screen.findByText(/respuesta táctica de prueba\./i)).toBeInTheDocument()
  })

  it('permite disparar una pregunta sugerida desde el chat global con el contexto cargado', async () => {
    mockApi.loadEventData.mockResolvedValueOnce({
      provider: 'StatsBomb Open Data',
      match_id: '99',
      competition_name: 'Liga',
      season_name: '2025',
      match_label: 'A vs B',
      home_team: 'Argentina',
      away_team: 'Francia',
      match_date: '2025-01-01',
      canonical_events: buildCanonicalEvents(100),
      insights: ['Insight demo'],
      raw_payload: [{ type: 'Pass', minute: 1, team: { name: 'Argentina' } }],
      metrics: {
        total_events: 100,
        total_passes: 40,
        total_shots: 10,
        progressive_actions: 7,
        final_third_actions: 12,
        recoveries: 8,
        total_under_pressure: 5,
        total_xg: 1.3,
        field_tilt_index: 72,
        field_tilt_label: 'Alto',
        directness_index: 60,
        directness_label: 'Medio',
        progressive_threat_index: 68,
        progressive_threat_label: 'Alto',
        recovery_height_index: 55,
        recovery_height_label: 'Medio',
        shot_quality_index: 48,
        shot_quality_label: 'Medio',
      },
      raw_events_count: 100,
      used_fallback_events: false,
      events_status_message: '',
    })

    renderVertical2Page()

    await waitFor(() => expect(mockApi.fetchCompetitions).toHaveBeenCalled())
    await waitFor(() =>
      expect(mockApi.fetchMatches).toHaveBeenCalledWith('StatsBomb Open Data', 1, 10),
    )
    fireEvent.click(screen.getByRole('button', { name: /cargar datos/i }))

    await waitFor(() => expect(mockApi.loadEventData).toHaveBeenCalled())
    expect(await screen.findByText(/resumen del partido/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /abrir ai coach/i }))
    await waitFor(() => expect(screen.getByText(/a vs b \| statsbomb open data/i)).toBeInTheDocument())
    fireEvent.click(await screen.findByRole('button', { name: /¿dónde generó más peligro\?/i }))

    await waitFor(() => expect(mockApi.askCoachQuestion).toHaveBeenCalled())
    expect(mockApi.askCoachQuestion).toHaveBeenCalledWith(
      expect.objectContaining({
        matchId: '99',
        provider: 'StatsBomb Open Data',
        question: '¿Dónde generó más peligro?',
      }),
    )
  })
})
