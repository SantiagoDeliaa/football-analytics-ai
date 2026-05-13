import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { AiCoachPanel } from './AiCoachPanel'
import * as api from '../../services/eventDataApi'
import type { EventDataResult } from '../../types/eventData'

vi.mock('../../services/eventDataApi')

const mockApi = vi.mocked(api)

const sampleResult: EventDataResult = {
  provider: 'StatsBomb Open Data',
  match_id: '99',
  competition_name: 'Liga',
  season_name: '2025',
  match_label: 'Argentina vs Francia',
  home_team: 'Argentina',
  away_team: 'Francia',
  match_date: '2025-01-01',
  canonical_events: [
    {
      event_id: '1',
      match_id: '99',
      team_name: 'Argentina',
      player_name: 'Lionel Messi',
      minute: 10,
      second: 0,
      event_type: 'Pass',
      x: 80,
      y: 40,
      end_x: 95,
      end_y: 42,
      outcome: 'Complete',
      progressive: true,
      under_pressure: false,
      xG: 0,
    },
  ],
  metrics: {
    total_events: 1,
    total_passes: 1,
    total_shots: 0,
    progressive_actions: 1,
    final_third_actions: 1,
    recoveries: 0,
    total_under_pressure: 0,
    total_xg: 0,
  },
  insights: ['Insight demo'],
  raw_events_count: 1,
  used_fallback_events: false,
  events_status_message: '',
}

describe('AiCoachPanel', () => {
  beforeEach(() => {
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
      diagnosis: 'Diagnóstico táctico listo.',
      error: '',
      suggested_questions: ['¿Qué debería corregir el cuerpo técnico?'],
    })
    mockApi.askCoachQuestion.mockResolvedValue({
      ok: true,
      answer: 'Debe ajustar la altura de la presión.',
      error: '',
      suggested_questions: ['¿Qué jugador fue más influyente?'],
    })
  })

  it('muestra estado degradado cuando el coach no está configurado', async () => {
    mockApi.fetchCoachStatus.mockResolvedValueOnce({
      configured: false,
      api_key_configured: false,
      model_configured: true,
      base_url_configured: true,
      model: 'gpt-4o-mini',
      base_url: 'https://api.openai.com/v1/chat/completions',
      message: 'Falta configurar AI_COACH_API_KEY en el entorno.',
    })

    render(<AiCoachPanel result={sampleResult} selectedPlayer="Todos" selectedTeam="Todos" />)

    expect((await screen.findAllByText(/falta configurar ai_coach_api_key/i)).length).toBeGreaterThan(
      0,
    )
    expect(screen.getByRole('button', { name: /generar diagnóstico táctico/i })).toBeDisabled()
  })

  it('genera diagnóstico y permite preguntar con historial conversacional', async () => {
    render(<AiCoachPanel result={sampleResult} selectedPlayer="Todos" selectedTeam="Argentina" />)

    await waitFor(() => expect(mockApi.fetchCoachStatus).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: /generar diagnóstico táctico/i }))
    await waitFor(() => expect(mockApi.generateCoachDiagnosis).toHaveBeenCalled())
    expect(await screen.findByText(/diagnóstico táctico listo\./i)).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText(/preguntale algo al ai coach/i), {
      target: { value: '¿Qué debería corregir el cuerpo técnico?' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^preguntar$/i }))

    await waitFor(() => expect(mockApi.askCoachQuestion).toHaveBeenCalled())
    expect(mockApi.askCoachQuestion).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'StatsBomb Open Data',
        matchId: '99',
        team: 'Argentina',
        player: undefined,
        conversationHistory: [],
      }),
    )
    expect(await screen.findByText(/debe ajustar la altura de la presión\./i)).toBeInTheDocument()
  })
})
