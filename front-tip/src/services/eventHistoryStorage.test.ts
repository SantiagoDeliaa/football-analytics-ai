import {
  getEventHistoryEntry,
  listEventHistoryEntries,
  saveEventHistoryEntry,
} from './eventHistoryStorage'
import type { EventDataResult } from '../types/eventData'

function buildResult(matchId: string): EventDataResult {
  return {
    match_id: matchId,
    competition_name: 'Liga',
    match_label: `Equipo A vs Equipo B ${matchId}`,
    canonical_events: [],
    metrics: {
      total_events: 100,
      total_passes: 40,
      total_shots: 8,
      progressive_actions: 10,
      final_third_actions: 12,
      recoveries: 7,
      total_under_pressure: 9,
      total_xg: 1.5,
    },
    insights: ['Insight demo'],
    raw_events_count: 100,
    used_fallback_events: false,
    events_status_message: '',
  }
}

describe('eventHistoryStorage', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('guarda y lista partidos procesados', () => {
    saveEventHistoryEntry({
      provider: 'StatsBomb Open Data',
      result: buildResult('101'),
      selection: {
        team: 'Argentina',
        player: 'Todos',
      },
    })

    const entries = listEventHistoryEntries()
    expect(entries).toHaveLength(1)
    expect(entries[0]?.result.match_id).toBe('101')
    expect(entries[0]?.selection.team).toBe('Argentina')
  })

  it('reemplaza la entrada de un mismo match y permite recuperarla', () => {
    saveEventHistoryEntry({
      provider: 'StatsBomb Open Data',
      result: buildResult('202'),
      selection: {
        team: 'Todos',
        player: 'Jugador 1',
      },
    })

    saveEventHistoryEntry({
      provider: 'StatsBomb Open Data',
      result: buildResult('202'),
      selection: {
        team: 'Francia',
        player: 'Jugador 2',
      },
    })

    const entries = listEventHistoryEntries()
    const entry = getEventHistoryEntry('202')

    expect(entries).toHaveLength(1)
    expect(entry?.selection.team).toBe('Francia')
    expect(entry?.selection.player).toBe('Jugador 2')
  })
})
