import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  loadPdfTacticalBoard,
  loadMetricsSection,
  loadPlayerSpotlight,
  loadVisualizationGrid,
  scheduleModulePrefetch,
} from '../app/modulePreload'
import { useEventDataContext } from '../app/EventDataContext'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { Tabs } from '../components/common/Tabs'
import { ApiFiltersPanel } from '../components/vertical2/ApiFiltersPanel'
import { AiCoachPanel } from '../components/vertical2/AiCoachPanel'
import { CanonicalPreview } from '../components/vertical2/CanonicalPreview'
import { EventHistoryPanel } from '../components/vertical2/EventHistoryPanel'
import { HistoryMatchHint } from '../components/vertical2/HistoryMatchHint'
import { InsightsList } from '../components/vertical2/InsightsList'
import { MatchSummary } from '../components/vertical2/MatchSummary'
import { ProcessedHistoryPanel } from '../components/vertical2/ProcessedHistoryPanel'
import { ProviderDebugPanel } from '../components/vertical2/ProviderDebugPanel'
import { PdfUploadForm } from '../components/vertical2/PdfUploadForm'
import { useAsync } from '../hooks/useAsync'
import {
  fetchApiFootballCountries,
  fetchApiFootballFixtures,
  fetchApiFootballLeagues,
  fetchCompetitions,
  fetchMatches,
  fetchProcessedHistory,
  loadEventData,
  loadProcessedHistoryEntry,
  uploadPdfReport,
} from '../services/eventDataApi'
import {
  listEventHistoryEntries,
  saveEventHistoryEntry,
} from '../services/eventHistoryStorage'
import type {
  ApiFootballCountry,
  ApiFootballLeague,
  EventHistoryEntry,
  PdfAnalysisResult,
  ProcessedHistoryMatch,
  ProviderOption,
} from '../types/eventData'
import {
  buildPrimaryMetrics,
  buildProprietaryMetrics,
} from '../utils/formatters'
import {
  extractPlayers,
  extractTeams,
  filterEventsBySelection,
} from '../utils/eventSelectors'
import {
  calculateOpenEventMetrics,
  generateOpenEventInsights,
} from '../utils/openEventAnalytics'

const MetricsSection = lazy(async () => {
  const module = await loadMetricsSection()
  return { default: module.MetricsSection }
})

const PlayerSpotlight = lazy(async () => {
  const module = await loadPlayerSpotlight()
  return { default: module.PlayerSpotlight }
})

const VisualizationGrid = lazy(async () => {
  const module = await loadVisualizationGrid()
  return { default: module.VisualizationGrid }
})

const PdfTacticalBoard = lazy(async () => {
  const module = await loadPdfTacticalBoard()
  return { default: module.PdfTacticalBoard }
})

type ActiveTab = 'pdf' | 'api'

function inferTeamsFromMatch(selectedMatch?: { home_team?: string; away_team?: string; display_name?: string }) {
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

export function Vertical2Page() {
  const navigate = useNavigate()
  const params = useParams()
  const [activeTab, setActiveTab] = useState<ActiveTab>('api')
  const [fallbackWarning, setFallbackWarning] = useState<string>()
  const [apiFootballCountries, setApiFootballCountries] = useState<ApiFootballCountry[]>([])
  const [apiFootballLeagues, setApiFootballLeagues] = useState<ApiFootballLeague[]>([])
  const [apiFootballSelectedCountry, setApiFootballSelectedCountry] = useState('')
  const [apiFootballSelectedLeagueId, setApiFootballSelectedLeagueId] = useState('')
  const [apiFootballSelectedSeason, setApiFootballSelectedSeason] = useState('')
  const [pdfResult, setPdfResult] = useState<PdfAnalysisResult>()
  const [pdfError, setPdfError] = useState<string>()
  const [historyEntries, setHistoryEntries] = useState<EventHistoryEntry[]>(() => listEventHistoryEntries())
  const [processedHistoryEntries, setProcessedHistoryEntries] = useState<ProcessedHistoryMatch[]>([])
  const { state, dispatch } = useEventDataContext()
  const competitionsTask = useAsync<Awaited<ReturnType<typeof fetchCompetitions>>>()
  const apiFootballCountriesTask = useAsync<Awaited<ReturnType<typeof fetchApiFootballCountries>>>()
  const apiFootballLeaguesTask = useAsync<Awaited<ReturnType<typeof fetchApiFootballLeagues>>>()
  const apiFootballFixturesTask = useAsync<Awaited<ReturnType<typeof fetchApiFootballFixtures>>>()
  const matchesTask = useAsync<Awaited<ReturnType<typeof fetchMatches>>>()
  const apiTask = useAsync<Awaited<ReturnType<typeof loadEventData>>>()
  const pdfTask = useAsync<Awaited<ReturnType<typeof uploadPdfReport>>>()
  const processedHistoryTask = useAsync<Awaited<ReturnType<typeof fetchProcessedHistory>>>()
  const historyEntryTask = useAsync<Awaited<ReturnType<typeof loadProcessedHistoryEntry>>>()

  useEffect(() => {
    if (state.provider !== 'StatsBomb Open Data') {
      return
    }

    void competitionsTask.run(() => fetchCompetitions(state.provider)).then((payload) => {
      if (!payload) return
      dispatch({ type: 'setCompetitions', competitions: payload.competitions })
      setFallbackWarning(
        payload.status.source === 'mock' || payload.status.source === 'error'
          ? payload.status.message
          : undefined,
      )
      if (payload.competitions.length > 0) {
        dispatch({ type: 'setSelectedCompetition', competition: payload.competitions[0] })
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.provider, dispatch])

  useEffect(() => {
    if (state.provider !== 'API-Football') {
      return
    }

    void apiFootballCountriesTask.run(() => fetchApiFootballCountries()).then((payload) => {
      if (!payload) {
        return
      }

      setApiFootballCountries(payload.countries)
      setFallbackWarning(payload.status.source === 'api' ? undefined : payload.status.message)
      setApiFootballSelectedCountry((currentCountry) => {
        if (currentCountry && payload.countries.some((country) => country.name === currentCountry)) {
          return currentCountry
        }
        return payload.countries[0]?.name ?? ''
      })
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.provider])

  useEffect(() => {
    if (state.provider !== 'API-Football') {
      return
    }
    if (!apiFootballSelectedCountry) {
      setApiFootballLeagues([])
      setApiFootballSelectedLeagueId('')
      setApiFootballSelectedSeason('')
      return
    }

    void apiFootballLeaguesTask
      .run(() => fetchApiFootballLeagues({ country: apiFootballSelectedCountry }))
      .then((payload) => {
        if (!payload) {
          return
        }

        setApiFootballLeagues(payload.leagues)
        setFallbackWarning(payload.status.source === 'api' ? undefined : payload.status.message)
        setApiFootballSelectedLeagueId((currentLeagueId) => {
          if (currentLeagueId && payload.leagues.some((league) => `${league.league_id}` === currentLeagueId)) {
            return currentLeagueId
          }
          return payload.leagues[0] ? `${payload.leagues[0].league_id}` : ''
        })
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.provider, apiFootballSelectedCountry])

  useEffect(() => {
    if (state.provider !== 'API-Football') {
      return
    }

    const selectedLeague = apiFootballLeagues.find(
      (league) => `${league.league_id}` === apiFootballSelectedLeagueId,
    )
    const nextSeasonOptions = selectedLeague?.seasons ?? []

    setApiFootballSelectedSeason((currentSeason) => {
      if (currentSeason && nextSeasonOptions.includes(Number(currentSeason))) {
        return currentSeason
      }
      if (selectedLeague?.current_season) {
        return `${selectedLeague.current_season}`
      }
      return nextSeasonOptions[0] ? `${nextSeasonOptions[0]}` : ''
    })
  }, [apiFootballLeagues, apiFootballSelectedLeagueId, state.provider])

  useEffect(() => {
    void processedHistoryTask.run(() => fetchProcessedHistory()).then((payload) => {
      if (!payload) {
        return
      }
      setProcessedHistoryEntries(payload)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (activeTab === 'pdf') {
      scheduleModulePrefetch(loadMetricsSection)
      scheduleModulePrefetch(loadPdfTacticalBoard)
    }
  }, [activeTab])

  useEffect(() => {
    if (!state.result) {
      return
    }

    scheduleModulePrefetch(loadMetricsSection)
    scheduleModulePrefetch(loadVisualizationGrid)

    if (state.selectedPlayer !== 'Todos') {
      scheduleModulePrefetch(loadPlayerSpotlight)
    }
  }, [state.result, state.selectedPlayer])

  useEffect(() => {
    if (state.provider !== 'StatsBomb Open Data') {
      return
    }
    if (!state.selectedCompetition) return
    void matchesTask
      .run(() =>
        fetchMatches(
          state.provider,
          state.selectedCompetition!.competition_id,
          state.selectedCompetition!.season_id,
        ),
      )
      .then((payload) => {
        if (!payload) return
        dispatch({ type: 'setMatches', matches: payload.matches })
        setFallbackWarning(
          payload.status.source === 'mock' || payload.status.source === 'error'
            ? payload.status.message
            : undefined,
        )

        const routeMatchId = params.matchId ? Number(params.matchId) : undefined
        const firstMatch = routeMatchId
          ? payload.matches.find((match) => match.match_id === routeMatchId)
          : payload.matches[0]
        dispatch({ type: 'setSelectedMatch', match: firstMatch })
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.provider, state.selectedCompetition, params.matchId, dispatch])

  const selectedApiFootballLeague = useMemo(
    () => apiFootballLeagues.find((league) => `${league.league_id}` === apiFootballSelectedLeagueId),
    [apiFootballLeagues, apiFootballSelectedLeagueId],
  )
  const selectedMatchKey = state.selectedMatch?.match_id ? `${state.selectedMatch.match_id}` : undefined

  const currentResultMatchesSelection =
    !!state.result &&
    !!state.selectedMatch &&
    `${state.result.match_id}` === `${state.selectedMatch.match_id}`

  const teamOptions = useMemo(
    () => {
      if (currentResultMatchesSelection) {
        return ['Todos', ...extractTeams(state.result?.canonical_events ?? [])]
      }

      const matchTeams = inferTeamsFromMatch(state.selectedMatch)

      return ['Todos', ...new Set(matchTeams)]
    },
    [currentResultMatchesSelection, state.result?.canonical_events, state.selectedMatch],
  )

  const playerOptions = useMemo(
    () =>
      currentResultMatchesSelection
        ? ['Todos', ...extractPlayers(state.result?.canonical_events ?? [], state.selectedTeam)]
        : ['Todos'],
    [currentResultMatchesSelection, state.result?.canonical_events, state.selectedTeam],
  )

  const filteredEvents = useMemo(
    () =>
      filterEventsBySelection(state.result?.canonical_events ?? [], {
        team: state.selectedTeam,
        player: state.selectedPlayer,
      }),
    [state.result?.canonical_events, state.selectedPlayer, state.selectedTeam],
  )

  const selectionMetrics = useMemo(() => {
    if (!state.result) {
      return undefined
    }

    return calculateOpenEventMetrics(state.result.canonical_events, {
      team: state.selectedTeam,
      player: state.selectedPlayer,
    })
  }, [state.result, state.selectedPlayer, state.selectedTeam])

  const selectionInsights = useMemo(() => {
    if (!selectionMetrics) {
      return []
    }

    return generateOpenEventInsights(selectionMetrics, {
      team: state.selectedTeam,
      player: state.selectedPlayer,
    })
  }, [selectionMetrics, state.selectedPlayer, state.selectedTeam])

  const primaryMetrics = useMemo(
    () => (selectionMetrics ? buildPrimaryMetrics(selectionMetrics) : []),
    [selectionMetrics],
  )

  const proprietaryMetrics = useMemo(() => {
    if (!selectionMetrics) {
      return []
    }

    return buildProprietaryMetrics(selectionMetrics).filter((metric) =>
      metric.title === 'Influencia jugador' ? state.selectedPlayer !== 'Todos' : true,
    )
  }, [selectionMetrics, state.selectedPlayer])

  const sourceLabel = useMemo(() => {
    if (!state.result) {
      return 'API'
    }

    const currentResult = state.result
    const isCurrentApiResult = apiTask.data?.match_id === currentResult.match_id
    const isCurrentBackendHistory = historyEntryTask.data?.match_id === currentResult.match_id
    const existsInHistory = historyEntries.some((entry) => entry.result.match_id === currentResult.match_id)

    if (!isCurrentApiResult && existsInHistory) {
      return 'Historial local'
    }

    if (isCurrentBackendHistory) {
      return 'Historial backend'
    }

    return 'API'
  }, [apiTask.data?.match_id, historyEntries, historyEntryTask.data?.match_id, state.result])

  const matchingLocalHistoryEntry = useMemo(() => {
    if (!selectedMatchKey) {
      return undefined
    }
    return historyEntries.find(
      (entry) => entry.provider === state.provider && entry.result.match_id === selectedMatchKey,
    )
  }, [historyEntries, selectedMatchKey, state.provider])

  const matchingBackendHistoryEntry = useMemo(() => {
    if (!selectedMatchKey) {
      return undefined
    }
    return processedHistoryEntries.find(
      (entry) => entry.provider === state.provider && entry.match_id === selectedMatchKey,
    )
  }, [processedHistoryEntries, selectedMatchKey, state.provider])

  useEffect(() => {
    if (!params.matchId) {
      return
    }

    if (state.selectedMatch && `${state.selectedMatch.match_id}` !== params.matchId) {
      return
    }

    const historyEntry = historyEntries.find((entry) => entry.result.match_id === params.matchId)
    if (!historyEntry || state.result?.match_id === historyEntry.result.match_id) {
      return
    }

    dispatch({ type: 'setResult', result: historyEntry.result })
    dispatch({ type: 'setTeam', team: historyEntry.selection.team })
    dispatch({ type: 'setPlayer', player: historyEntry.selection.player })
  }, [dispatch, historyEntries, params.matchId, state.result?.match_id, state.selectedMatch])

  function resetApiSelectionState() {
    dispatch({ type: 'setMatches', matches: [] })
    dispatch({ type: 'setSelectedMatch', match: undefined })
  }

  async function handleSearchApiFootballFixtures() {
    if (!selectedApiFootballLeague || !apiFootballSelectedSeason) {
      return
    }

    const payload = await apiFootballFixturesTask.run(() =>
      fetchApiFootballFixtures({
        leagueId: selectedApiFootballLeague.league_id,
        season: Number(apiFootballSelectedSeason),
      }),
    )
    if (!payload) {
      return
    }

    dispatch({ type: 'setMatches', matches: payload.matches })
    setFallbackWarning(payload.status.source === 'api' ? undefined : payload.status.message)

    const routeMatchId = params.matchId ? Number(params.matchId) : undefined
    const firstMatch = routeMatchId
      ? payload.matches.find((match) => match.match_id === routeMatchId)
      : payload.matches[0]
    dispatch({ type: 'setSelectedMatch', match: firstMatch })
  }

  async function handleLoadData() {
    if (!state.selectedMatch) return
    const result = await apiTask.run(() =>
      loadEventData({
        provider: state.provider,
        matchId: state.selectedMatch!.match_id,
        team: state.selectedTeam === 'Todos' ? undefined : state.selectedTeam,
        player: state.selectedPlayer === 'Todos' ? undefined : state.selectedPlayer,
        competitionName:
          state.provider === 'API-Football'
            ? selectedApiFootballLeague?.league_name
            : state.selectedCompetition?.competition_name,
        seasonName:
          state.provider === 'API-Football' ? apiFootballSelectedSeason : state.selectedCompetition?.season_name,
        matchLabel: state.selectedMatch?.display_name,
        homeTeam: state.selectedMatch?.home_team,
        awayTeam: state.selectedMatch?.away_team,
        matchDate: state.selectedMatch?.match_date,
      }),
    )
    if (!result) return
    dispatch({ type: 'setResult', result })
    saveEventHistoryEntry({
      provider: state.provider,
      result,
      selection: {
        team: state.selectedTeam,
        player: state.selectedPlayer,
      },
    })
    setHistoryEntries(listEventHistoryEntries())
    navigate(`/vertical2/match/${result.match_id}`)
  }

  async function handleSubmitPdf(file: File) {
    setPdfError(undefined)
    const result = await pdfTask.run(() => uploadPdfReport(file))
    if (!result) {
      setPdfError('No se pudo procesar el PDF.')
      return
    }
    setPdfResult(result)
  }

  function handleLoadHistory(entry: EventHistoryEntry) {
    dispatch({ type: 'setResult', result: entry.result })
    dispatch({ type: 'setTeam', team: entry.selection.team })
    dispatch({ type: 'setPlayer', player: entry.selection.player })
    setActiveTab('api')
    navigate(`/vertical2/match/${entry.result.match_id}`)
  }

  async function handleLoadProcessedHistory(provider: ProviderOption, matchId: string) {
    const result = await historyEntryTask.run(() => loadProcessedHistoryEntry(provider, matchId))
    if (!result) {
      return
    }

    dispatch({ type: 'setResult', result })
    dispatch({ type: 'setTeam', team: 'Todos' })
    dispatch({ type: 'setPlayer', player: 'Todos' })
    setActiveTab('api')
    navigate(`/vertical2/match/${result.match_id}`)
  }

  const pdfMatchInfo =
    pdfResult?.normalized_payload && typeof pdfResult.normalized_payload.match_info === 'object'
      ? (pdfResult.normalized_payload.match_info as Record<string, unknown>)
      : undefined
  const pdfIngestion = pdfResult?.ingestion

  return (
    <section className="space-y-6">
      <header className="relative overflow-hidden rounded-[28px] border border-emerald-500/20 bg-slate-950/70 p-6 shadow-2xl md:p-8">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(29,215,144,0.12),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(79,140,255,0.12),_transparent_30%)]" />
        <div className="relative">
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-emerald-300">
            Tactical Intelligence Platform
          </p>
          <h1 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-100 md:text-4xl">
            Vertical 2 · Data Analytics
          </h1>
          <p className="mt-3 max-w-3xl text-sm text-slate-300 md:text-base">
            Resumen táctico desde reportes PDF y datos de proveedores API, con foco en lectura rápida,
            métricas propietarias y una experiencia más robusta que la versión original.
          </p>
        </div>
      </header>

      <Tabs
        onChange={(value: string) => setActiveTab(value as ActiveTab)}
        options={[
          { value: 'pdf', label: 'Subir PDF' },
          { value: 'api', label: 'Datos por API' },
        ]}
        value={activeTab}
      />

      {activeTab === 'pdf' ? (
        <div className="space-y-4">
          <PdfUploadForm loading={pdfTask.loading} onSubmit={handleSubmitPdf} />
          {pdfTask.loading ? <LoadingState label="Procesando reporte PDF..." /> : null}
          {pdfError ? <ErrorState message={pdfError} /> : null}
          {pdfResult ? (
            <>
              <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
                <h3 className="text-lg font-semibold text-slate-100">Estado de procesamiento</h3>
                <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                  <StatusItem
                    label="Archivo"
                    value={String(pdfMatchInfo?.file_name ?? 'Reporte sin nombre')}
                  />
                  <StatusItem
                    label="Competición"
                    value={String(pdfMatchInfo?.competition ?? 'No detectada')}
                  />
                  <StatusItem
                    label="Fecha"
                    value={String(pdfMatchInfo?.date ?? 'No detectada')}
                  />
                  <StatusItem
                    label="Páginas"
                    value={String(pdfMatchInfo?.source_pages ?? '0')}
                  />
                  <StatusItem
                    label="Parser"
                    value={String(pdfIngestion?.parser ?? 'No detectado')}
                  />
                  <StatusItem
                    label="Extracción"
                    value={String(pdfIngestion?.status ?? 'Sin estado')}
                  />
                </div>
                {pdfIngestion?.messages?.length ? (
                  <p className="mt-4 text-sm text-slate-300">{pdfIngestion.messages.join(' ')}</p>
                ) : null}
              </section>

              <Suspense fallback={<LoadingState label="Cargando KPIs del partido..." />}>
                <MetricsSection
                  columns="grid-cols-1 md:grid-cols-2 xl:grid-cols-4"
                  items={buildProprietaryMetrics(pdfResult.metrics).filter(
                    (metric) => metric.title !== 'Altura de recuperación' && metric.title !== 'Influencia jugador',
                  )}
                  title="KPIs del partido"
                />
              </Suspense>

              <Suspense fallback={<LoadingState label="Cargando visual táctico..." />}>
                <PdfTacticalBoard payload={pdfResult.normalized_payload} />
              </Suspense>
              <section className="space-y-2">
                <h3 className="text-lg font-semibold text-slate-100">Insights del partido</h3>
                <InsightsList insights={pdfResult.insights} />
              </section>
              <details className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
                <summary className="cursor-pointer text-sm font-semibold text-slate-200">
                  Preview del schema normalizado
                </summary>
                <pre className="mt-3 max-h-64 overflow-auto rounded bg-slate-950 p-3 text-xs text-emerald-200">
                  {JSON.stringify(pdfResult.normalized_payload, null, 2)}
                </pre>
              </details>
            </>
          ) : null}
        </div>
      ) : (
        <div className="space-y-4">
          <ApiFiltersPanel
            apiFootballCountries={apiFootballCountries}
            apiFootballLeagues={apiFootballLeagues}
            apiFootballLoadingFixtures={apiFootballFixturesTask.loading}
            apiFootballSelectedCountry={apiFootballSelectedCountry}
            apiFootballSelectedLeagueId={apiFootballSelectedLeagueId}
            apiFootballSelectedSeason={apiFootballSelectedSeason}
            competitions={state.competitions}
            loading={
              apiTask.loading ||
              competitionsTask.loading ||
              matchesTask.loading ||
              apiFootballCountriesTask.loading ||
              apiFootballLeaguesTask.loading
            }
            matches={state.matches}
            onApiFootballCountryChange={(country) => {
              setApiFootballSelectedCountry(country)
              setApiFootballLeagues([])
              setApiFootballSelectedLeagueId('')
              setApiFootballSelectedSeason('')
              resetApiSelectionState()
            }}
            onApiFootballLeagueChange={(leagueId) => {
              setApiFootballSelectedLeagueId(leagueId)
              resetApiSelectionState()
            }}
            onApiFootballSearch={handleSearchApiFootballFixtures}
            onApiFootballSeasonChange={(season) => {
              setApiFootballSelectedSeason(season)
              resetApiSelectionState()
            }}
            onCompetitionChange={(competitionId) => {
              const selected = state.competitions.find(
                (competition) => competition.competition_id === Number(competitionId),
              )
              dispatch({ type: 'setSelectedCompetition', competition: selected })
            }}
            onMatchChange={(matchId) => {
              const selected = state.matches.find((match) => match.match_id === Number(matchId))
              dispatch({ type: 'setSelectedMatch', match: selected })
            }}
            onPlayerChange={(player) => dispatch({ type: 'setPlayer', player })}
            onProviderChange={(provider) =>
              dispatch({ type: 'setProvider', provider: provider as ProviderOption })
            }
            onSubmit={handleLoadData}
            onTeamChange={(team) => dispatch({ type: 'setTeam', team })}
            playerOptions={playerOptions}
            provider={state.provider}
            selectedCompetition={state.selectedCompetition}
            selectedMatch={state.selectedMatch}
            selectedPlayer={state.selectedPlayer}
            selectedTeam={state.selectedTeam}
            teamOptions={teamOptions}
            warning={fallbackWarning}
          />

          {state.selectedMatch &&
          !currentResultMatchesSelection &&
          (matchingLocalHistoryEntry || matchingBackendHistoryEntry) ? (
            <HistoryMatchHint
              backendEntry={matchingBackendHistoryEntry}
              localEntry={matchingLocalHistoryEntry}
              onLoadBackend={handleLoadProcessedHistory}
              onLoadLocal={handleLoadHistory}
            />
          ) : null}

          <EventHistoryPanel
            activeMatchId={state.result?.match_id}
            entries={historyEntries}
            onLoad={handleLoadHistory}
          />

          <ProcessedHistoryPanel
            activeMatchId={state.result?.match_id}
            entries={processedHistoryEntries}
            loading={processedHistoryTask.loading}
            onLoad={handleLoadProcessedHistory}
          />

          {apiTask.loading ? <LoadingState label="Cargando y normalizando eventos..." /> : null}
          {apiTask.error ? <ErrorState message={apiTask.error} onRetry={handleLoadData} /> : null}
          {historyEntryTask.error ? (
            <ErrorState message={historyEntryTask.error} onRetry={() => void processedHistoryTask.run(() => fetchProcessedHistory())} />
          ) : null}

          {state.result ? (
            <>
              {state.result.used_fallback_events ? (
                <ErrorState
                  message={`Mostrando datos de fallback para eventos${
                    state.result.events_status_message
                      ? ` (${state.result.events_status_message})`
                      : ''
                  }`}
                />
              ) : null}

              <MatchSummary
                competitionName={state.result.competition_name}
                matchLabel={state.result.match_label}
                player={state.selectedPlayer}
                sourceLabel={sourceLabel}
                team={state.selectedTeam}
              />

              <Suspense fallback={<LoadingState label="Cargando resumen del partido..." />}>
                <MetricsSection items={primaryMetrics} title="Resumen del partido" />
                <MetricsSection
                  columns="grid-cols-1 md:grid-cols-2 xl:grid-cols-3"
                  items={proprietaryMetrics}
                  title="Métricas propietarias"
                />
              </Suspense>

              <section className="space-y-2">
                <h3 className="text-lg font-semibold text-slate-100">Visualizaciones tácticas</h3>
                <Suspense fallback={<LoadingState label="Cargando visualizaciones tácticas..." />}>
                  <VisualizationGrid events={filteredEvents} />
                </Suspense>
              </section>

              {selectionMetrics ? (
                <Suspense fallback={<LoadingState label="Cargando perfil del jugador..." />}>
                  <PlayerSpotlight metrics={selectionMetrics} player={state.selectedPlayer} />
                </Suspense>
              ) : null}

              <section className="space-y-2">
                <h3 className="text-lg font-semibold text-slate-100">Insights iniciales</h3>
                <InsightsList insights={selectionInsights} />
              </section>

              <AiCoachPanel
                key={`${state.result.provider}-${state.result.match_id}-${state.selectedTeam}-${state.selectedPlayer}`}
                result={state.result}
                selectedPlayer={state.selectedPlayer}
                selectedTeam={state.selectedTeam}
              />

              <ProviderDebugPanel
                history={{
                  localEntry: matchingLocalHistoryEntry,
                  backendEntry: matchingBackendHistoryEntry,
                  onLoadLocal: handleLoadHistory,
                  onLoadBackend: handleLoadProcessedHistory,
                }}
                result={state.result}
              />

              <CanonicalPreview events={state.result.canonical_events} />
            </>
          ) : null}
        </div>
      )}
    </section>
  )
}

function StatusItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm font-semibold text-slate-100">{value}</p>
    </div>
  )
}
