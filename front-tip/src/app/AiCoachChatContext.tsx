/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  askCoachQuestion,
  fetchCoachStatus,
  generateCoachDiagnosis,
} from '../services/eventDataApi'
import type {
  CoachConfigStatus,
  CoachConversationMessage,
  EventDataResult,
  ProviderOption,
} from '../types/eventData'

const STORAGE_KEY = 'tip:ai-coach-chat-session'

const DEFAULT_SUGGESTED_QUESTIONS = [
  '¿Cómo estuvo el equipo en términos generales?',
  '¿Dónde generó más peligro?',
  '¿Qué debería corregir el cuerpo técnico?',
]

interface AiCoachRuntimeContext {
  id: string
  sourceLabel: string
  summary: string
  provider: ProviderOption
  matchId: string
  team?: string
  player?: string
  competitionName?: string
  seasonName?: string
  matchLabel?: string
  homeTeam?: string
  awayTeam?: string
  matchDate?: string
}

interface AiCoachChatMessage extends CoachConversationMessage {
  id: string
  createdAt: string
  contextId: string
  contextLabel: string
}

interface PersistedAiCoachState {
  activeContext?: AiCoachRuntimeContext
  conversation: AiCoachChatMessage[]
  diagnosisByContext: Record<string, string>
  isOpen: boolean
  suggestedQuestionsByContext: Record<string, string[]>
}

interface RegisterEventDataContextInput {
  result: EventDataResult
  selectedTeam: string
  selectedPlayer: string
}

interface AiCoachChatContextValue {
  activeContext?: AiCoachRuntimeContext
  conversation: AiCoachChatMessage[]
  diagnosis: string
  diagnosisLoading: boolean
  draft: string
  hasActiveContext: boolean
  isConfigured: boolean
  isOpen: boolean
  pendingQuestion?: string
  questionLoading: boolean
  registerEventDataContext: (input: RegisterEventDataContextInput) => void
  requestQuestion: (question: string) => void
  responseError?: string
  setDraft: (value: string) => void
  setIsOpen: (value: boolean) => void
  status?: CoachConfigStatus
  statusLoading: boolean
  submitQuestion: (question: string) => Promise<boolean>
  suggestedQuestions: string[]
  toggleOpen: () => void
  triggerDiagnosis: () => Promise<boolean>
  clearPendingQuestion: () => void
}

function readPersistedState(): PersistedAiCoachState {
  if (typeof window === 'undefined') {
    return {
      conversation: [],
      diagnosisByContext: {},
      isOpen: false,
      suggestedQuestionsByContext: {},
    }
  }

  try {
    const stored = window.sessionStorage.getItem(STORAGE_KEY)
    if (!stored) {
      return {
        conversation: [],
        diagnosisByContext: {},
        isOpen: false,
        suggestedQuestionsByContext: {},
      }
    }

    const parsed = JSON.parse(stored) as Partial<PersistedAiCoachState>
    return {
      activeContext: parsed.activeContext,
      conversation: Array.isArray(parsed.conversation) ? parsed.conversation : [],
      diagnosisByContext:
        parsed.diagnosisByContext && typeof parsed.diagnosisByContext === 'object'
          ? parsed.diagnosisByContext
          : {},
      isOpen: Boolean(parsed.isOpen),
      suggestedQuestionsByContext:
        parsed.suggestedQuestionsByContext && typeof parsed.suggestedQuestionsByContext === 'object'
          ? parsed.suggestedQuestionsByContext
          : {},
    }
  } catch {
    return {
      conversation: [],
      diagnosisByContext: {},
      isOpen: false,
      suggestedQuestionsByContext: {},
    }
  }
}

function buildContextSummary(
  result: EventDataResult,
  selectedTeam: string,
  selectedPlayer: string,
): string {
  const selectionParts = [
    selectedTeam !== 'Todos' ? `equipo ${selectedTeam}` : undefined,
    selectedPlayer !== 'Todos' ? `jugador ${selectedPlayer}` : undefined,
  ].filter(Boolean)

  const selectionLabel =
    selectionParts.length > 0 ? ` | ${selectionParts.join(' | ')}` : ''

  return `${result.match_label} | ${result.provider}${selectionLabel}`
}

function buildEventDataContext({
  result,
  selectedPlayer,
  selectedTeam,
}: RegisterEventDataContextInput): AiCoachRuntimeContext {
  const scopedTeam = selectedTeam !== 'Todos' ? selectedTeam : undefined
  const scopedPlayer = selectedPlayer !== 'Todos' ? selectedPlayer : undefined

  return {
    id: [
      result.provider,
      result.match_id,
      scopedTeam ?? 'Todos',
      scopedPlayer ?? 'Todos',
    ].join('::'),
    sourceLabel: 'Data Analytics',
    summary: buildContextSummary(result, selectedTeam, selectedPlayer),
    provider: result.provider,
    matchId: result.match_id,
    team: scopedTeam,
    player: scopedPlayer,
    competitionName: result.competition_name,
    seasonName: result.season_name,
    matchLabel: result.match_label,
    homeTeam: result.home_team,
    awayTeam: result.away_team,
    matchDate: result.match_date,
  }
}

const AiCoachChatContext = createContext<AiCoachChatContextValue | null>(null)

export function AiCoachChatProvider({ children }: { children: ReactNode }) {
  const persistedState = useMemo(readPersistedState, [])
  const [activeContext, setActiveContext] = useState<AiCoachRuntimeContext | undefined>(
    persistedState.activeContext,
  )
  const [conversation, setConversation] = useState<AiCoachChatMessage[]>(persistedState.conversation)
  const [diagnosisByContext, setDiagnosisByContext] = useState<Record<string, string>>(
    persistedState.diagnosisByContext,
  )
  const [draft, setDraft] = useState('')
  const [isOpen, setIsOpen] = useState(persistedState.isOpen)
  const [pendingQuestion, setPendingQuestion] = useState<string>()
  const [responseError, setResponseError] = useState<string>()
  const [status, setStatus] = useState<CoachConfigStatus>()
  const [statusLoading, setStatusLoading] = useState(true)
  const [diagnosisLoading, setDiagnosisLoading] = useState(false)
  const [questionLoading, setQuestionLoading] = useState(false)
  const [suggestedQuestionsByContext, setSuggestedQuestionsByContext] = useState<
    Record<string, string[]>
  >(persistedState.suggestedQuestionsByContext)

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const payload: PersistedAiCoachState = {
      activeContext,
      conversation,
      diagnosisByContext,
      isOpen,
      suggestedQuestionsByContext,
    }
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  }, [activeContext, conversation, diagnosisByContext, isOpen, suggestedQuestionsByContext])

  useEffect(() => {
    let cancelled = false

    async function loadStatus() {
      setStatusLoading(true)

      try {
        const nextStatus = await fetchCoachStatus()
        if (cancelled) {
          return
        }
        setStatus(nextStatus)
        setResponseError(nextStatus.configured ? undefined : nextStatus.message)
      } catch (error) {
        if (cancelled) {
          return
        }
        setResponseError(
          error instanceof Error ? error.message : 'No se pudo verificar la configuración del AI Coach.',
        )
      } finally {
        if (!cancelled) {
          setStatusLoading(false)
        }
      }
    }

    void loadStatus()

    return () => {
      cancelled = true
    }
  }, [])

  const activeConversationHistory = useMemo<CoachConversationMessage[]>(
    () =>
      activeContext
        ? conversation
            .filter((message) => message.contextId === activeContext.id)
            .map((message) => ({ role: message.role, content: message.content }))
        : [],
    [activeContext, conversation],
  )

  const diagnosis = activeContext ? diagnosisByContext[activeContext.id] ?? '' : ''
  const suggestedQuestions = activeContext
    ? suggestedQuestionsByContext[activeContext.id] ?? DEFAULT_SUGGESTED_QUESTIONS
    : DEFAULT_SUGGESTED_QUESTIONS
  const isConfigured = status?.configured ?? false
  const hasActiveContext = Boolean(activeContext)

  const registerEventDataContext = useCallback((input: RegisterEventDataContextInput) => {
    setActiveContext(buildEventDataContext(input))
  }, [])

  const triggerDiagnosis = useCallback(async () => {
    if (!activeContext) {
      setResponseError('Cargá un partido en Data Analytics para activar el contexto del AI Coach.')
      return false
    }

    if (!isConfigured) {
      setResponseError(status?.message || 'El AI Coach no está configurado todavía.')
      return false
    }

    setResponseError(undefined)
    setDiagnosisLoading(true)

    try {
      const response = await generateCoachDiagnosis({
        provider: activeContext.provider,
        matchId: activeContext.matchId,
        team: activeContext.team,
        player: activeContext.player,
        competitionName: activeContext.competitionName,
        seasonName: activeContext.seasonName,
        matchLabel: activeContext.matchLabel,
        homeTeam: activeContext.homeTeam,
        awayTeam: activeContext.awayTeam,
        matchDate: activeContext.matchDate,
      })

      if (!response.ok) {
        setResponseError(response.error || 'No se pudo generar el diagnóstico táctico.')
        return false
      }

      setDiagnosisByContext((current) => ({
        ...current,
        [activeContext.id]: response.diagnosis,
      }))

      if (response.suggested_questions.length > 0) {
        setSuggestedQuestionsByContext((current) => ({
          ...current,
          [activeContext.id]: response.suggested_questions,
        }))
      }

      return true
    } catch (error) {
      setResponseError(
        error instanceof Error ? error.message : 'No se pudo generar el diagnóstico táctico.',
      )
      return false
    } finally {
      setDiagnosisLoading(false)
    }
  }, [activeContext, isConfigured, status?.message])

  const submitQuestion = useCallback(
    async (question: string) => {
      const normalizedQuestion = question.trim()

      if (!normalizedQuestion) {
        setResponseError('La pregunta no puede estar vacía.')
        return false
      }

      if (!activeContext) {
        setResponseError('Cargá un partido en Data Analytics para activar el contexto del AI Coach.')
        return false
      }

      if (!isConfigured) {
        setResponseError(status?.message || 'El AI Coach no está configurado todavía.')
        return false
      }

      setResponseError(undefined)
      setQuestionLoading(true)

      try {
        const response = await askCoachQuestion({
          provider: activeContext.provider,
          matchId: activeContext.matchId,
          team: activeContext.team,
          player: activeContext.player,
          competitionName: activeContext.competitionName,
          seasonName: activeContext.seasonName,
          matchLabel: activeContext.matchLabel,
          homeTeam: activeContext.homeTeam,
          awayTeam: activeContext.awayTeam,
          matchDate: activeContext.matchDate,
          question: normalizedQuestion,
          conversationHistory: activeConversationHistory,
        })

        if (!response.ok) {
          setResponseError(response.error || 'No se pudo responder la pregunta.')
          return false
        }

        const createdAt = new Date().toISOString()
        const contextLabel = activeContext.summary

        setConversation((current) => [
          ...current,
          {
            id: `${createdAt}-user`,
            role: 'user',
            content: normalizedQuestion,
            createdAt,
            contextId: activeContext.id,
            contextLabel,
          },
          {
            id: `${createdAt}-assistant`,
            role: 'assistant',
            content: response.answer,
            createdAt,
            contextId: activeContext.id,
            contextLabel,
          },
        ])
        setDraft('')

        if (response.suggested_questions.length > 0) {
          setSuggestedQuestionsByContext((current) => ({
            ...current,
            [activeContext.id]: response.suggested_questions,
          }))
        }

        return true
      } catch (error) {
        setResponseError(error instanceof Error ? error.message : 'No se pudo responder la pregunta.')
        return false
      } finally {
        setQuestionLoading(false)
      }
    },
    [activeContext, activeConversationHistory, isConfigured, status?.message],
  )

  const requestQuestion = useCallback((question: string) => {
    setDraft(question)
    setPendingQuestion(question)
    setIsOpen(true)
  }, [])

  const clearPendingQuestion = useCallback(() => {
    setPendingQuestion(undefined)
  }, [])

  const toggleOpen = useCallback(() => {
    setIsOpen((current) => !current)
  }, [])

  const value = useMemo(
    () => ({
      activeContext,
      clearPendingQuestion,
      conversation,
      diagnosis,
      diagnosisLoading,
      draft,
      hasActiveContext,
      isConfigured,
      isOpen,
      pendingQuestion,
      questionLoading,
      registerEventDataContext,
      requestQuestion,
      responseError,
      setDraft,
      setIsOpen,
      status,
      statusLoading,
      submitQuestion,
      suggestedQuestions,
      toggleOpen,
      triggerDiagnosis,
    }),
    [
      activeContext,
      clearPendingQuestion,
      conversation,
      diagnosis,
      diagnosisLoading,
      draft,
      hasActiveContext,
      isConfigured,
      isOpen,
      pendingQuestion,
      questionLoading,
      registerEventDataContext,
      requestQuestion,
      responseError,
      status,
      statusLoading,
      submitQuestion,
      suggestedQuestions,
      toggleOpen,
      triggerDiagnosis,
    ],
  )

  return <AiCoachChatContext.Provider value={value}>{children}</AiCoachChatContext.Provider>
}

export function useAiCoachChat() {
  const context = useContext(AiCoachChatContext)
  if (!context) {
    throw new Error('useAiCoachChat debe usarse dentro de AiCoachChatProvider.')
  }
  return context
}
