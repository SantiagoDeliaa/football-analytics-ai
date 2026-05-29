export interface MatchCenterVenue {
  name?: string | null
  city?: string | null
}

export interface MatchCenterTeamSummary {
  id?: string | null
  name?: string | null
  score?: number | null
}

export interface MatchCenterMatch {
  match_id: string
  competition?: string | null
  season?: string | null
  date?: string | null
  status?: string | null
  venue?: MatchCenterVenue | null
  home_team: MatchCenterTeamSummary
  away_team: MatchCenterTeamSummary
}

export interface TeamExpectedMetrics {
  team_name?: string | null
  xg?: number | null
  xgot?: number | null
  xpts?: number | null
  npxg?: number | null
  xg_open_play?: number | null
  xg_set_play?: number | null
  xg_free_kicks?: number | null
  shooting_performance?: number | null
  xga?: number | null
}

export interface MatchCenterExpectedMetrics {
  home?: TeamExpectedMetrics | null
  away?: TeamExpectedMetrics | null
}

export interface MatchTimelineEvent {
  minute?: number | null
  extra_minute?: number | null
  team_name?: string | null
  player_name?: string | null
  related_player_name?: string | null
  event_type?: string | null
  event_label?: string | null
  result?: string | null
  description?: string | null
}

export interface MatchCenterLineupPlayer {
  player_id?: string | null
  player_name?: string | null
  position?: string | null
  jersey_number?: number | null
  minutes_played?: number | null
  rating?: number | null
  is_starter?: boolean | null
}

export interface MatchCenterLineupSide {
  team_id?: string | null
  team_name?: string | null
  formation?: string | null
  coach?: string | null
  starters: MatchCenterLineupPlayer[]
  substitutes: MatchCenterLineupPlayer[]
}

export interface MatchCenterLineups {
  home?: MatchCenterLineupSide | null
  away?: MatchCenterLineupSide | null
}

export interface MatchCenterStatEntry {
  key: string
  label: string
  value?: string | number | boolean | null
}

export interface MatchCenterTeamStatsSide {
  team_id?: string | null
  team_name?: string | null
  stats: MatchCenterStatEntry[]
}

export interface MatchCenterTeamStats {
  home?: MatchCenterTeamStatsSide | null
  away?: MatchCenterTeamStatsSide | null
}

export interface PlayerMatchStats {
  player_id?: string | null
  player_name?: string | null
  team_id?: string | null
  team_name?: string | null
  position?: string | null
  jersey_number?: number | null
  is_starter?: boolean | null
  minutes_played?: number | null
  rating?: number | null
  stats: MatchCenterStatEntry[]
  insights?: string[]
}

export interface DerivedMetricsSide {
  eficacia_ofensiva?: number | null
  rendimiento_definicion?: number | null
  amenaza_jugada?: number | null
  amenaza_pelota_parada?: number | null
}

export interface MatchCenterDerivedMetrics {
  home?: DerivedMetricsSide | null
  away?: DerivedMetricsSide | null
}

export interface MatchCenterEnabledModules {
  match_center?: boolean
  expected_metrics?: boolean
  timeline?: boolean
  lineups?: boolean
  team_stats?: boolean
  player_stats?: boolean
  event_maps?: boolean
  shot_map?: boolean
  pass_network?: boolean
}

export interface MatchCenterDataQuality {
  level?: 'alta' | 'media' | 'baja' | string
  has_xg?: boolean
  has_xgot?: boolean
  has_xpts?: boolean
  has_lineups?: boolean
  has_player_stats?: boolean
  has_team_stats?: boolean
  has_event_timeline?: boolean
  has_event_coordinates?: boolean
  enabled_modules?: MatchCenterEnabledModules
  message?: string
}

export interface MatchCenterResponse {
  provider: string
  match: MatchCenterMatch
  expected_metrics?: MatchCenterExpectedMetrics | null
  timeline?: MatchTimelineEvent[]
  lineups?: MatchCenterLineups | null
  team_stats?: MatchCenterTeamStats | null
  player_stats?: PlayerMatchStats[]
  derived_metrics?: MatchCenterDerivedMetrics | null
  insights?: string[]
  data_quality?: MatchCenterDataQuality | null
}
