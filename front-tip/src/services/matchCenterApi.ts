import { apiRequest } from './apiClient'
import type { MatchCenterResponse } from '../types/matchCenter'

export async function getSportmonksMatchCenter(matchId: string): Promise<MatchCenterResponse> {
  return apiRequest<MatchCenterResponse>(
    `/api/v1/event-data/providers/sportmonks/matches/${encodeURIComponent(matchId)}/match-center`,
  )
}
