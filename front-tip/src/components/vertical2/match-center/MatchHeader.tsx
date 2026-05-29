import type { MatchCenterMatch } from '../../../types/matchCenter'
import { formatMatchDate, formatMatchScore, formatNullableValue } from '../../../utils/matchCenter'

interface MatchHeaderProps {
  match: MatchCenterMatch
}

export function MatchHeader({ match }: MatchHeaderProps) {
  const homeTeamName = match.home_team.name || 'Equipo local'
  const awayTeamName = match.away_team.name || 'Equipo visitante'

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-emerald-300">Sportmonks Match Center</p>
          <h1 className="mt-2 text-2xl font-semibold text-slate-50 md:text-3xl">
            {homeTeamName} {formatMatchScore(match)} {awayTeamName}
          </h1>
          <p className="mt-2 text-sm text-slate-300">
            {formatNullableValue(match.competition)} · {formatNullableValue(match.season, 'Temporada no disponible')}
          </p>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-950/60 px-4 py-3 text-sm text-slate-200">
          <p>
            <span className="text-slate-400">Estado:</span> {formatNullableValue(match.status)}
          </p>
          <p className="mt-1">
            <span className="text-slate-400">Fecha:</span> {formatMatchDate(match.date)}
          </p>
          <p className="mt-1">
            <span className="text-slate-400">Estadio:</span>{' '}
            {formatNullableValue(match.venue?.name, 'No disponible')}
          </p>
          <p className="mt-1">
            <span className="text-slate-400">Ciudad:</span>{' '}
            {formatNullableValue(match.venue?.city, 'No disponible')}
          </p>
        </div>
      </div>
    </section>
  )
}
