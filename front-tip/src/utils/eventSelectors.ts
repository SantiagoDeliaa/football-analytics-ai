import type { CanonicalEvent } from '../types/eventData'

export function filterEventsBySelection(
  events: CanonicalEvent[],
  selection?: { team?: string; player?: string },
) {
  return events.filter((event) => {
    const matchesTeam =
      !selection?.team || selection.team === 'Todos' ? true : event.team_name === selection.team
    const matchesPlayer =
      !selection?.player || selection.player === 'Todos'
        ? true
        : event.player_name === selection.player

    return matchesTeam && matchesPlayer
  })
}

export function extractTeams(events: CanonicalEvent[]): string[] {
  return [...new Set(events.map((event) => event.team_name).filter(Boolean))].sort()
}

export function extractPlayers(events: CanonicalEvent[], team: string): string[] {
  return [
    ...new Set(
      filterEventsBySelection(events, { team })
        .map((event) => event.player_name)
        .filter((player) => Boolean(player) && player !== 'Jugador desconocido'),
    ),
  ].sort()
}
