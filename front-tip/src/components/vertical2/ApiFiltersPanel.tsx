import type { Competition, Match, ProviderOption } from '../../types/eventData'

interface ApiFiltersPanelProps {
  provider: ProviderOption
  competitions: Competition[]
  matches: Match[]
  selectedCompetition?: Competition
  selectedMatch?: Match
  selectedTeam: string
  selectedPlayer: string
  teamOptions: string[]
  playerOptions: string[]
  loading: boolean
  warning?: string
  onProviderChange: (provider: ProviderOption) => void
  onCompetitionChange: (competitionId: string) => void
  onMatchChange: (matchId: string) => void
  onTeamChange: (team: string) => void
  onPlayerChange: (player: string) => void
  onSubmit: () => void
}

export function ApiFiltersPanel(props: ApiFiltersPanelProps) {
  const {
    provider,
    competitions,
    matches,
    selectedCompetition,
    selectedMatch,
    selectedTeam,
    selectedPlayer,
    teamOptions,
    playerOptions,
    loading,
    warning,
    onProviderChange,
    onCompetitionChange,
    onMatchChange,
    onTeamChange,
    onPlayerChange,
    onSubmit,
  } = props

  const competitionValue =
    selectedCompetition && selectedCompetition.competition_id !== undefined && selectedCompetition.competition_id !== null
      ? `${selectedCompetition.competition_id}`
      : ''
  const matchValue =
    selectedMatch && selectedMatch.match_id !== undefined && selectedMatch.match_id !== null
      ? `${selectedMatch.match_id}`
      : ''
  const safeTeamValue = teamOptions.includes(selectedTeam) ? selectedTeam : teamOptions[0] ?? ''
  const safePlayerValue = playerOptions.includes(selectedPlayer) ? selectedPlayer : playerOptions[0] ?? ''

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h3 className="text-base font-semibold text-slate-100">Datos por API</h3>
      <p className="mt-1 text-sm text-slate-300">
        Conectá event data para generar métricas tácticas propietarias.
      </p>

      {warning ? (
        <p className="mt-3 rounded-lg border border-amber-700/50 bg-amber-900/30 p-2 text-xs text-amber-100">
          {warning}
        </p>
      ) : null}

      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="text-sm text-slate-200">
          Proveedor
          <select
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            onChange={(event) => onProviderChange(event.target.value as ProviderOption)}
            value={provider}
          >
            <option value="StatsBomb Open Data">StatsBomb Open Data</option>
            <option value="API-Football">API-Football</option>
          </select>
        </label>

        <label className="text-sm text-slate-200">
          Competición / temporada
          <select
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            onChange={(event) => onCompetitionChange(event.target.value)}
            value={competitionValue}
          >
            <option value="">Seleccionar</option>
            {competitions.map((competition, index) => (
              <option
                key={`${competition.competition_id}-${competition.season_id}-${competition.display_name}-${index}`}
                value={competition.competition_id}
              >
                {competition.display_name}
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm text-slate-200">
          Partido
          <select
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            onChange={(event) => onMatchChange(event.target.value)}
            value={matchValue}
          >
            <option value="">Seleccionar</option>
            {matches.map((match, index) => (
              <option key={`${match.match_id}-${match.display_name}-${index}`} value={match.match_id}>
                {match.display_name}
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm text-slate-200">
          Equipo
          <select
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            onChange={(event) => onTeamChange(event.target.value)}
            value={safeTeamValue}
          >
            {teamOptions.map((team, index) => (
              <option key={`${team}-${index}`} value={team}>
                {team}
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm text-slate-200 md:col-span-2">
          Jugador
          <select
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            onChange={(event) => onPlayerChange(event.target.value)}
            value={safePlayerValue}
          >
            {playerOptions.map((player, index) => (
              <option key={`${player}-${index}`} value={player}>
                {player}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        className="mt-4 rounded-md border border-emerald-500 bg-emerald-700 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
        disabled={loading || !selectedMatch}
        onClick={onSubmit}
        type="button"
      >
        {loading ? 'Cargando...' : 'Cargar datos'}
      </button>
    </section>
  )
}
