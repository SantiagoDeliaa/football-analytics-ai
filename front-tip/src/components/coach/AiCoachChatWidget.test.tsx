import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { Link, MemoryRouter, Route, Routes } from 'react-router-dom'
import { AiCoachChatProvider, useAiCoachChat } from '../../app/AiCoachChatContext'
import * as api from '../../services/eventDataApi'
import type { EventDataResult } from '../../types/eventData'
import { AiCoachChatWidget } from './AiCoachChatWidget'

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
  canonical_events: [],
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
  insights: [],
  raw_events_count: 0,
  used_fallback_events: false,
  events_status_message: '',
}

function Vertical2Harness() {
  const { registerEventDataContext } = useAiCoachChat()

  useEffect(() => {
    registerEventDataContext({
      result: sampleResult,
      selectedTeam: 'Argentina',
      selectedPlayer: 'Todos',
    })
  }, [registerEventDataContext])

  return (
    <div>
      <p>Vertical 2 lista</p>
      <Link to="/vertical1">Ir a Vertical 1</Link>
    </div>
  )
}

function Vertical1Harness() {
  return (
    <div>
      <p>Vertical 1 lista</p>
      <Link to="/vertical2">Volver a Vertical 2</Link>
    </div>
  )
}

describe('AiCoachChatWidget', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    window.sessionStorage.clear()
    mockApi.fetchCoachStatus.mockResolvedValue({
      configured: true,
      api_key_configured: true,
      model_configured: true,
      base_url_configured: true,
      model: 'gpt-4o-mini',
      base_url: 'https://api.openai.com/v1/chat/completions',
      message: 'AI Tactical Coach configurado correctamente.',
    })
    mockApi.askCoachQuestion.mockResolvedValue({
      ok: true,
      answer: 'El equipo generó más peligro por derecha.',
      error: '',
      suggested_questions: ['¿Cómo sostuvo la presión alta?'],
    })
    mockApi.generateCoachDiagnosis.mockResolvedValue({
      ok: true,
      diagnosis: 'Diagnóstico persistente.',
      error: '',
      suggested_questions: ['¿Qué debería ajustar el mediocampo?'],
    })
  })

  it('mantiene historial y contexto al navegar entre verticales', async () => {
    render(
      <MemoryRouter initialEntries={['/vertical2']}>
        <AiCoachChatProvider>
          <AiCoachChatWidget />
          <Routes>
            <Route element={<Vertical2Harness />} path="/vertical2" />
            <Route element={<Vertical1Harness />} path="/vertical1" />
          </Routes>
        </AiCoachChatProvider>
      </MemoryRouter>,
    )

    await waitFor(() => expect(mockApi.fetchCoachStatus).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: /abrir ai coach/i }))

    expect(await screen.findByText(/argentina vs francia \| statsbomb open data \| equipo argentina/i)).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText(/preguntale algo al ai coach/i), {
      target: { value: '¿Dónde generó más peligro?' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^preguntar$/i }))

    await waitFor(() => expect(mockApi.askCoachQuestion).toHaveBeenCalled())
    expect(await screen.findByText(/el equipo generó más peligro por derecha\./i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('link', { name: /ir a vertical 1/i }))

    expect(await screen.findByText(/vertical 1 lista/i)).toBeInTheDocument()
    expect(screen.getByText(/seguís conversando con el contexto cargado en data analytics/i)).toBeInTheDocument()
    expect(screen.getByText(/el equipo generó más peligro por derecha\./i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /minimizar ai coach/i }))
    fireEvent.click(screen.getByRole('button', { name: /abrir ai coach/i }))

    expect(screen.getByText(/el equipo generó más peligro por derecha\./i)).toBeInTheDocument()
  })
})
