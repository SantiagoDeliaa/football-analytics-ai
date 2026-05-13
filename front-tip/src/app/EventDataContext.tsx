/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useMemo, useReducer, type ReactNode } from 'react'
import type {
  Competition,
  EventDataResult,
  Match,
  ProviderOption,
} from '../types/eventData'

type EventDataState = {
  provider: ProviderOption
  competitions: Competition[]
  matches: Match[]
  selectedCompetition?: Competition
  selectedMatch?: Match
  selectedTeam: string
  selectedPlayer: string
  result?: EventDataResult
}

type Action =
  | { type: 'setProvider'; provider: ProviderOption }
  | { type: 'setCompetitions'; competitions: Competition[] }
  | { type: 'setSelectedCompetition'; competition?: Competition }
  | { type: 'setMatches'; matches: Match[] }
  | { type: 'setSelectedMatch'; match?: Match }
  | { type: 'setTeam'; team: string }
  | { type: 'setPlayer'; player: string }
  | { type: 'setResult'; result?: EventDataResult }

export const initialEventDataState: EventDataState = {
  provider: 'StatsBomb Open Data',
  competitions: [],
  matches: [],
  selectedTeam: 'Todos',
  selectedPlayer: 'Todos',
}

export function eventDataReducer(state: EventDataState, action: Action): EventDataState {
  switch (action.type) {
    case 'setProvider':
      return {
        ...state,
        provider: action.provider,
        competitions: [],
        matches: [],
        selectedCompetition: undefined,
        selectedMatch: undefined,
        selectedTeam: 'Todos',
        selectedPlayer: 'Todos',
        result: undefined,
      }
    case 'setCompetitions':
      return { ...state, competitions: action.competitions }
    case 'setSelectedCompetition':
      return {
        ...state,
        selectedCompetition: action.competition,
        selectedMatch: undefined,
        selectedTeam: 'Todos',
        selectedPlayer: 'Todos',
        result: undefined,
      }
    case 'setMatches':
      return { ...state, matches: action.matches }
    case 'setSelectedMatch': {
      const nextMatchId =
        action.match && action.match.match_id !== undefined && action.match.match_id !== null
          ? `${action.match.match_id}`
          : undefined
      const currentResultId = state.result?.match_id ? `${state.result.match_id}` : undefined
      const shouldPreserveCurrentResult = nextMatchId && currentResultId && nextMatchId === currentResultId

      if (shouldPreserveCurrentResult) {
        return { ...state, selectedMatch: action.match }
      }

      return {
        ...state,
        selectedMatch: action.match,
        selectedTeam: 'Todos',
        selectedPlayer: 'Todos',
        result: undefined,
      }
    }
    case 'setTeam':
      return { ...state, selectedTeam: action.team, selectedPlayer: 'Todos' }
    case 'setPlayer':
      return { ...state, selectedPlayer: action.player }
    case 'setResult':
      return { ...state, result: action.result }
    default:
      return state
  }
}

type EventDataContextValue = {
  state: EventDataState
  dispatch: React.Dispatch<Action>
}

const EventDataContext = createContext<EventDataContextValue | null>(null)

export function EventDataProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(eventDataReducer, initialEventDataState)
  const value = useMemo(() => ({ state, dispatch }), [state])
  return <EventDataContext.Provider value={value}>{children}</EventDataContext.Provider>
}

export function useEventDataContext() {
  const ctx = useContext(EventDataContext)
  if (!ctx) {
    throw new Error('useEventDataContext debe usarse dentro de EventDataProvider.')
  }
  return ctx
}
