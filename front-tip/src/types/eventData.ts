export type ProviderOption = 'StatsBomb Open Data' | 'API-Football'

export interface Competition {
  competition_id: number
  season_id: number
  competition_name: string
  season_name: string
  country_name?: string
  display_name: string
}

export interface ApiFootballCountry {
  name: string
  code: string
  flag: string
  display_name: string
}

export interface ApiFootballLeague {
  league_id: number
  league_name: string
  country_name: string
  type: string
  logo: string
  seasons: number[]
  current_season?: number | null
  display_name: string
}

export interface Match {
  match_id: number
  display_name: string
  home_team?: string
  away_team?: string
  match_date?: string
}

export interface CanonicalEvent {
  event_id: string
  match_id: string
  team_name: string
  player_name: string
  minute: number
  second: number
  event_type: string
  x?: number | null
  y?: number | null
  end_x?: number | null
  end_y?: number | null
  outcome?: string | null
  progressive: boolean
  under_pressure: boolean
  xG: number
  xA?: number | null
}

export interface MatchMetrics {
  total_events: number
  total_passes: number
  total_shots: number
  progressive_actions: number
  final_third_actions: number
  recoveries: number
  total_under_pressure: number
  total_xg: number
  field_tilt_index?: number | null
  field_tilt_label?: string
  directness_index?: number | null
  directness_label?: string
  progressive_threat_index?: number | null
  progressive_threat_label?: string
  recovery_height_index?: number | null
  recovery_height_label?: string
  shot_quality_index?: number | null
  shot_quality_label?: string
  player_influence_score?: number | null
  player_influence_label?: string
}

export interface IngestionStatus {
  source: 'api' | 'mock' | 'error'
  message?: string
}

export interface EventDataResult {
  provider: ProviderOption
  match_id: string
  competition_name: string
  season_name: string
  match_label: string
  home_team: string
  away_team: string
  match_date: string
  canonical_events: CanonicalEvent[]
  metrics: MatchMetrics
  insights: string[]
  raw_events_count: number
  raw_payload?: unknown
  used_fallback_events: boolean
  events_status_message: string
}

export interface PdfAnalysisResult {
  normalized_payload: Record<string, unknown>
  metrics: MatchMetrics
  insights: string[]
  ingestion?: {
    status: string
    parser?: string | null
    page_count: number
    bytes_size: number
    messages: string[]
  }
}

export interface EventSelection {
  team: string
  player: string
}

export interface EventHistoryEntry {
  id: string
  provider: ProviderOption
  competition_name: string
  match_label: string
  saved_at: string
  selection: EventSelection
  result: EventDataResult
}

export interface ProcessedHistoryMatch {
  provider: ProviderOption
  match_id: string
  competition_name: string
  season_name: string
  home_team: string
  away_team: string
  match_date: string
  created_at: string
  updated_at: string
}

export interface CoachConversationMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface CoachRequest {
  provider: ProviderOption
  matchId: string | number
  team?: string
  player?: string
  competitionName?: string
  seasonName?: string
  matchLabel?: string
  homeTeam?: string
  awayTeam?: string
  matchDate?: string
}

export interface CoachQuestionRequest extends CoachRequest {
  question: string
  conversationHistory?: CoachConversationMessage[]
}

export interface CoachConfigStatus {
  configured: boolean
  api_key_configured: boolean
  model_configured: boolean
  base_url_configured: boolean
  model: string
  base_url: string
  message: string
}

export interface CoachDiagnosisResult {
  ok: boolean
  diagnosis: string
  error: string
  suggested_questions: string[]
}

export interface CoachAnswerResult {
  ok: boolean
  answer: string
  error: string
  suggested_questions: string[]
}
