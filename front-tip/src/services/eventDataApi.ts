import { apiRequest, API_BASE_URL } from './apiClient'
import { ApiError } from './apiClient'
import type {
  ApiFootballCountry,
  ApiFootballLeague,
  CoachAnswerResult,
  CoachConfigStatus,
  CoachDiagnosisResult,
  CoachQuestionRequest,
  CoachRequest,
  DeleteProcessedMatchResult,
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

export async function fetchApiFootballCountries(): Promise<{
  countries: ApiFootballCountry[]
  status: IngestionStatus
}> {
  try {
    const countries = await apiRequest<ApiFootballCountry[]>('/api/v1/event-data/api-football/countries')
    return { countries, status: { source: 'api' } }
  } catch (error) {
    return {
      countries: [],
      status: buildErrorStatus(error, 'No se pudieron cargar países desde API-Football.'),
    }
  }
}

export async function fetchApiFootballLeagues(params: {
  country: string
  season?: number
  search?: string
}): Promise<{ leagues: ApiFootballLeague[]; status: IngestionStatus }> {
  const query = new URLSearchParams({ country: params.country })
  if (params.season !== undefined) {
    query.set('season', `${params.season}`)
  }
  if (params.search?.trim()) {
    query.set('search', params.search.trim())
  }

  try {
    const leagues = await apiRequest<ApiFootballLeague[]>(
      `/api/v1/event-data/api-football/leagues?${query.toString()}`,
    )
    return { leagues, status: { source: 'api' } }
  } catch (error) {
    return {
      leagues: [],
      status: buildErrorStatus(error, 'No se pudieron cargar ligas desde API-Football.'),
    }
  }
}

export async function fetchApiFootballFixtures(params: {
  leagueId: number
  season: number
}): Promise<{ matches: Match[]; status: IngestionStatus }> {
  const query = new URLSearchParams({
    league_id: `${params.leagueId}`,
    season: `${params.season}`,
  })

  try {
    const matches = await apiRequest<Match[]>(
      `/api/v1/event-data/api-football/fixtures?${query.toString()}`,
    )
    return { matches, status: { source: 'api' } }
  } catch (error) {
    return {
      matches: [],
      status: buildErrorStatus(error, 'No se pudieron cargar partidos desde API-Football.'),
    }
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

export async function deleteProcessedHistoryEntry(
  provider: ProviderOption,
  matchId: string,
): Promise<DeleteProcessedMatchResult> {
  return apiRequest<DeleteProcessedMatchResult>(
    `/api/v1/event-data/history/${encodeURIComponent(provider)}/${encodeURIComponent(matchId)}`,
    {
      method: 'DELETE',
    },
  )
}

function buildCoachPayload(req: CoachRequest) {
  return {
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
  }
}

export async function fetchCoachStatus(): Promise<CoachConfigStatus> {
  return apiRequest<CoachConfigStatus>('/api/v1/event-data/coach/status')
}

export async function generateCoachDiagnosis(req: CoachRequest): Promise<CoachDiagnosisResult> {
  return apiRequest<CoachDiagnosisResult>('/api/v1/event-data/coach/diagnosis', {
    method: 'POST',
    body: JSON.stringify(buildCoachPayload(req)),
  })
}

export async function askCoachQuestion(req: CoachQuestionRequest): Promise<CoachAnswerResult> {
  return apiRequest<CoachAnswerResult>('/api/v1/event-data/coach/question', {
    method: 'POST',
    body: JSON.stringify({
      ...buildCoachPayload(req),
      question: req.question,
      conversation_history: req.conversationHistory ?? [],
    }),
  })
}
