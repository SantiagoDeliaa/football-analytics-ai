import type {
  ApiFootballCountry,
  ApiFootballLeague,
  Competition,
  Match,
  ProviderOption,
} from '../../types/eventData'

function inferTeamsFromSelectedMatch(selectedMatch?: Match): string[] {
  const directTeams = [selectedMatch?.home_team, selectedMatch?.away_team].filter(
    (team): team is string => Boolean(team?.trim()),
  )

  if (directTeams.length > 0) {
    return directTeams
  }

  const label = String(selectedMatch?.display_name ?? '').trim()
  if (!label) {
    return []
  }

  const [teamsChunk] = label.split(' — ')
  const inferredTeams = teamsChunk
    .split(' vs ')
    .map((team) => team.trim())
    .filter(Boolean)

  return inferredTeams.length >= 2 ? inferredTeams.slice(0, 2) : []
}

interface ApiFiltersPanelProps {
  provider: ProviderOption
  competitions: Competition[]
  matches: Match[]
  apiFootballCountries: ApiFootballCountry[]
  apiFootballLeagues: ApiFootballLeague[]
  apiFootballSelectedCountry: string
  apiFootballSelectedLeagueId: string
  apiFootballSelectedSeason: string
  apiFootballLoadingFixtures: boolean
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
  onApiFootballCountryChange: (country: string) => void
  onApiFootballLeagueChange: (leagueId: string) => void
  onApiFootballSeasonChange: (season: string) => void
  onApiFootballSearch: () => void
  onTeamChange: (team: string) => void
  onPlayerChange: (player: string) => void
  onSubmit: () => void
}

export function ApiFiltersPanel(props: ApiFiltersPanelProps) {
  const {
    provider,
    competitions,
    matches,
    apiFootballCountries,
    apiFootballLeagues,
    apiFootballSelectedCountry,
    apiFootballSelectedLeagueId,
    apiFootballSelectedSeason,
    apiFootballLoadingFixtures,
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
    onApiFootballCountryChange,
    onApiFootballLeagueChange,
    onApiFootballSeasonChange,
    onApiFootballSearch,
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
  const resolvedTeamOptions =
    teamOptions.length > 1 ? teamOptions : ['Todos', ...inferTeamsFromSelectedMatch(selectedMatch)]
  const safeTeamValue = resolvedTeamOptions.includes(selectedTeam) ? selectedTeam : resolvedTeamOptions[0] ?? ''
  const safePlayerValue = playerOptions.includes(selectedPlayer) ? selectedPlayer : playerOptions[0] ?? ''
  const selectedApiFootballLeague =
    provider === 'API-Football'
      ? apiFootballLeagues.find((league) => `${league.league_id}` === apiFootballSelectedLeagueId)
      : undefined
  const apiFootballSeasonOptions = selectedApiFootballLeague?.seasons ?? []
  const canSearchApiFootballFixtures =
    provider === 'API-Football' &&
    Boolean(apiFootballSelectedCountry) &&
    Boolean(apiFootballSelectedLeagueId) &&
    Boolean(apiFootballSelectedSeason)
  const teamGridSpanClass = provider === 'API-Football' ? '' : 'md:col-span-2'

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

        {provider === 'API-Football' ? (
          <label className="text-sm text-slate-200">
            País
            <select
              className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
              onChange={(event) => onApiFootballCountryChange(event.target.value)}
              value={apiFootballSelectedCountry}
            >
              <option value="">Seleccionar</option>
              {apiFootballCountries.map((country) => (
                <option key={country.name} value={country.name}>
                  {country.display_name}
                </option>
              ))}
            </select>
          </label>
        ) : (
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
        )}

        {provider === 'API-Football' ? (
          <>
            <label className="text-sm text-slate-200">
              Liga
              <select
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
                onChange={(event) => onApiFootballLeagueChange(event.target.value)}
                value={apiFootballSelectedLeagueId}
              >
                <option value="">Seleccionar</option>
                {apiFootballLeagues.map((league) => (
                  <option key={league.league_id} value={league.league_id}>
                    {league.display_name}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-sm text-slate-200">
              Temporada
              <select
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
                onChange={(event) => onApiFootballSeasonChange(event.target.value)}
                value={apiFootballSelectedSeason}
              >
                <option value="">Seleccionar</option>
                {apiFootballSeasonOptions.map((season) => (
                  <option key={season} value={season}>
                    {season}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : null}

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
            {resolvedTeamOptions.map((team, index) => (
              <option key={`${team}-${index}`} value={team}>
                {team}
              </option>
            ))}
          </select>
        </label>

        <label className={`text-sm text-slate-200 ${teamGridSpanClass}`}>
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

      <div className="mt-4 flex flex-col gap-3 md:flex-row">
        {provider === 'API-Football' ? (
          <button
            className="rounded-md border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-100 hover:border-sky-400 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={loading || apiFootballLoadingFixtures || !canSearchApiFootballFixtures}
            onClick={onApiFootballSearch}
            type="button"
          >
            {apiFootballLoadingFixtures ? 'Buscando partidos...' : 'Buscar partidos'}
          </button>
        ) : null}

        <button
          className="rounded-md border border-emerald-500 bg-emerald-700 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={loading || !selectedMatch}
          onClick={onSubmit}
          type="button"
        >
          {loading ? 'Cargando...' : 'Cargar datos'}
        </button>
      </div>
    </section>
  )
}
