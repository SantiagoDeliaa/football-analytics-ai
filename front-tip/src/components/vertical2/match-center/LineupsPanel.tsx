import type { MatchCenterLineupSide, MatchCenterLineups } from '../../../types/matchCenter'
import { formatNullableValue, getLineupAvailability } from '../../../utils/matchCenter'

interface LineupsPanelProps {
  lineups?: MatchCenterLineups | null
}

function TeamLineupColumn({
  title,
  lineup,
}: {
  title: string
  lineup?: MatchCenterLineupSide | null
}) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <h3 className="text-base font-semibold text-slate-50">{title}</h3>
      <p className="mt-1 text-sm text-slate-300">
        Formación: {formatNullableValue(lineup?.formation, 'No disponible')}
      </p>
      <p className="mt-1 text-sm text-slate-300">
        DT: {formatNullableValue(lineup?.coach, 'No disponible')}
      </p>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <h4 className="text-sm font-semibold text-slate-100">Titulares</h4>
          <ul className="mt-2 space-y-2 text-sm text-slate-300">
            {lineup?.starters.length ? (
              lineup.starters.map((player) => (
                <li className="rounded-lg border border-slate-800 px-3 py-2" key={player.player_id || player.player_name}>
                  <span className="font-medium text-slate-100">
                    {formatNullableValue(player.jersey_number, '-')}{' '}
                    {formatNullableValue(player.player_name)}
                  </span>
                  <span className="ml-2 text-slate-400">
                    {formatNullableValue(player.position, '-')} · {formatNullableValue(player.minutes_played, '-')} min
                  </span>
                </li>
              ))
            ) : (
              <li className="text-slate-400">No disponibles.</li>
            )}
          </ul>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-slate-100">Suplentes</h4>
          <ul className="mt-2 space-y-2 text-sm text-slate-300">
            {lineup?.substitutes.length ? (
              lineup.substitutes.map((player) => (
                <li className="rounded-lg border border-slate-800 px-3 py-2" key={player.player_id || player.player_name}>
                  <span className="font-medium text-slate-100">
                    {formatNullableValue(player.jersey_number, '-')}{' '}
                    {formatNullableValue(player.player_name)}
                  </span>
                  <span className="ml-2 text-slate-400">{formatNullableValue(player.position, '-')}</span>
                </li>
              ))
            ) : (
              <li className="text-slate-400">No disponibles.</li>
            )}
          </ul>
        </div>
      </div>
    </article>
  )
}

export function LineupsPanel({ lineups }: LineupsPanelProps) {
  const hasAnyLineup = getLineupAvailability(lineups?.home) || getLineupAvailability(lineups?.away)

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="text-lg font-semibold text-slate-50">Lineups</h2>
      {!hasAnyLineup ? (
        <p className="mt-3 text-sm text-slate-300">Lineups no disponibles para este partido.</p>
      ) : (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          <TeamLineupColumn lineup={lineups?.home} title={lineups?.home?.team_name || 'Local'} />
          <TeamLineupColumn lineup={lineups?.away} title={lineups?.away?.team_name || 'Visitante'} />
        </div>
      )}
    </section>
  )
}
