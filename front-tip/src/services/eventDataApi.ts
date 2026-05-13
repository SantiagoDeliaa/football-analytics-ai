import { apiRequest, API_BASE_URL } from './apiClient'
import { ApiError } from './apiClient'
import type {
  Competition,
  EventDataResult,
  IngestionStatus,
  Match,
  PdfAnalysisResult,
  ProcessedHistoryMatch,
  ProviderOption,
} from '../types/eventData'

const MOCK_COMPETITIONS: Competition[] = [
  {
    competition_id: 43,
    season_id: 3,
    competition_name: 'FIFA World Cup',
    season_name: '2018',
    display_name: 'FIFA World Cup - 2018',
  },
]

const MOCK_MATCHES: Match[] = [
  {
    match_id: 7585,
    display_name: 'Argentina vs Francia',
    home_team: 'Argentina',
    away_team: 'Francia',
    match_date: '2018-06-30',
  },
]

function buildErrorStatus(error: unknown, fallbackMessage: string): IngestionStatus {
  if (error instanceof ApiError) {
    return {
      source: 'error',
      message: error.message || fallbackMessage,
    }
  }

  return {
    source: 'error',
    message: fallbackMessage,
  }
}

export async function fetchCompetitions(provider: ProviderOption): Promise<{
  competitions: Competition[]
  status: IngestionStatus
}> {
  try {
    const competitions = await apiRequest<Competition[]>(
      `/api/v1/event-data/competitions?provider=${encodeURIComponent(provider)}`,
    )
    return { competitions, status: { source: 'api' } }
  } catch (error) {
    if (provider === 'StatsBomb Open Data') {
      return {
        competitions: MOCK_COMPETITIONS,
        status: { source: 'mock', message: 'No se pudo conectar con backend; usando fallback.' },
      }
    }

    return {
      competitions: [],
      status: buildErrorStatus(error, 'No se pudieron cargar ligas desde API-Football.'),
    }
  }
}

export async function fetchMatches(
  provider: ProviderOption,
  competitionId: number,
  seasonId: number,
): Promise<{ matches: Match[]; status: IngestionStatus }> {
  try {
    const matches = await apiRequest<Match[]>(
      `/api/v1/event-data/matches?provider=${encodeURIComponent(provider)}&competition_id=${competitionId}&season_id=${seasonId}`,
    )
    return { matches, status: { source: 'api' } }
  } catch (error) {
    if (provider === 'StatsBomb Open Data') {
      return {
        matches: MOCK_MATCHES,
        status: { source: 'mock', message: 'No se pudo conectar con backend; usando fallback.' },
      }
    }

    return {
      matches: [],
      status: buildErrorStatus(error, 'No se pudieron cargar partidos desde API-Football.'),
    }
  }
}

export interface LoadEventDataRequest {
  provider: ProviderOption
  matchId: number
  team?: string
  player?: string
  competitionName?: string
  seasonName?: string
  matchLabel?: string
  homeTeam?: string
  awayTeam?: string
  matchDate?: string
}

export async function loadEventData(req: LoadEventDataRequest): Promise<EventDataResult> {
  const payload = await apiRequest<EventDataResult>('/api/v1/event-data/analyze', {
    method: 'POST',
    body: JSON.stringify({
      provider: req.provider,
      match_id: req.matchId,
      team: req.team || null,
      player: req.player || null,
      competition_name: req.competitionName || null,
      season_name: req.seasonName || null,
      match_label: req.matchLabel || null,
      home_team: req.homeTeam || null,
      away_team: req.awayTeam || null,
      match_date: req.matchDate || null,
    }),
  })
  return payload
}

export async function uploadPdfReport(file: File): Promise<PdfAnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/v1/event-data/pdf`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('No se pudo procesar el PDF.')
  }

  return (await response.json()) as PdfAnalysisResult
}

export async function fetchProcessedHistory(limit = 20): Promise<ProcessedHistoryMatch[]> {
  try {
    return await apiRequest<ProcessedHistoryMatch[]>(`/api/v1/event-data/history?limit=${limit}`)
  } catch {
    return []
  }
}

export async function loadProcessedHistoryEntry(
  provider: ProviderOption,
  matchId: string,
): Promise<EventDataResult> {
  return apiRequest<EventDataResult>(
    `/api/v1/event-data/history/${encodeURIComponent(provider)}/${encodeURIComponent(matchId)}`,
  )
}
