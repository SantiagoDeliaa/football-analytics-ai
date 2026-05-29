import type { PlayerMatchStats } from '../../../types/matchCenter'
import { formatMetricValue, formatNullableValue, getPlayerStatValue } from '../../../utils/matchCenter'

interface PlayerStatsTableProps {
  playerStats?: PlayerMatchStats[]
}

const columns = [
  { label: 'Toques', keys: ['touches', 'ball_touches'] },
  { label: 'Pases', keys: ['passes'] },
  { label: 'Precisión de pase', keys: ['accurate_passes_percentage', 'pass_accuracy'] },
  { label: 'Pases clave', keys: ['key_passes'] },
  { label: 'Chances creadas', keys: ['chances_created', 'big_chances_created'] },
  { label: 'Remates', keys: ['shots_total', 'shots', 'shots-total'] },
  { label: 'Remates al arco', keys: ['shots_on_target', 'shots_on_goal', 'shots-on-target'] },
  { label: 'Duelos ganados', keys: ['duels_won'] },
  { label: 'Recuperaciones', keys: ['recoveries', 'ball_recoveries'] },
  { label: 'Pérdidas de posesión', keys: ['possession_lost', 'dispossessed'] },
] as const

function renderStat(player: PlayerMatchStats, keys: string[]) {
  const value = getPlayerStatValue(player.stats, keys)
  if (typeof value === 'number') {
    return formatMetricValue(value, '-')
  }
  return formatNullableValue(value, '-')
}

export function PlayerStatsTable({ playerStats }: PlayerStatsTableProps) {
  if (!playerStats?.length) {
    return (
      <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <h2 className="text-lg font-semibold text-slate-50">Estadísticas de jugadores</h2>
        <p className="mt-3 text-sm text-slate-300">
          No hay estadísticas de jugadores disponibles para este partido.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="text-lg font-semibold text-slate-50">Estadísticas de jugadores</h2>
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-slate-300">
              <th className="px-3 py-2">Jugador</th>
              <th className="px-3 py-2">Equipo</th>
              <th className="px-3 py-2">Minutos</th>
              <th className="px-3 py-2">Rating</th>
              {columns.map((column) => (
                <th className="px-3 py-2" key={column.label}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {playerStats.map((player) => (
              <tr className="border-b border-slate-800/80 align-top" key={player.player_id || `${player.team_name}-${player.player_name}`}>
                <td className="px-3 py-2 text-slate-100">{formatNullableValue(player.player_name)}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(player.team_name)}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(player.minutes_played, '-')}</td>
                <td className="px-3 py-2 text-slate-300">
                  {player.rating === null || player.rating === undefined ? '-' : formatMetricValue(player.rating, '-')}
                </td>
                {columns.map((column) => (
                  <td className="px-3 py-2 text-slate-300" key={column.label}>
                    {renderStat(player, [...column.keys])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
