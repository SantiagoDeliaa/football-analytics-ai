import { useEffect, useMemo, useState } from 'react'
import { ErrorState } from '../common/ErrorState'
import { LoadingState } from '../common/LoadingState'
import { useAsync } from '../../hooks/useAsync'
import {
  askCoachQuestion,
  fetchCoachStatus,
  generateCoachDiagnosis,
} from '../../services/eventDataApi'
import type {
  CoachAnswerResult,
  CoachConfigStatus,
  CoachConversationMessage,
  CoachDiagnosisResult,
  EventDataResult,
} from '../../types/eventData'

const DEFAULT_SUGGESTED_QUESTIONS = [
  '¿Cómo estuvo el equipo en términos generales?',
  '¿Dónde generó más peligro?',
  '¿Qué debería corregir el cuerpo técnico?',
]

interface AiCoachPanelProps {
  result: EventDataResult
  selectedTeam: string
  selectedPlayer: string
}

export function AiCoachPanel({ result, selectedTeam, selectedPlayer }: AiCoachPanelProps) {
  const [diagnosis, setDiagnosis] = useState('')
  const [question, setQuestion] = useState('')
  const [conversation, setConversation] = useState<CoachConversationMessage[]>([])
  const [responseError, setResponseError] = useState<string>()
  const [suggestedQuestions, setSuggestedQuestions] = useState(DEFAULT_SUGGESTED_QUESTIONS)
  const statusTask = useAsync<CoachConfigStatus>()
  const diagnosisTask = useAsync<CoachDiagnosisResult>()
  const questionTask = useAsync<CoachAnswerResult>()

  const coachRequest = useMemo(
    () => ({
      provider: result.provider,
      matchId: result.match_id,
      team: selectedTeam === 'Todos' ? undefined : selectedTeam,
      player: selectedPlayer === 'Todos' ? undefined : selectedPlayer,
      competitionName: result.competition_name,
      seasonName: result.season_name,
      matchLabel: result.match_label,
      homeTeam: result.home_team,
      awayTeam: result.away_team,
      matchDate: result.match_date,
    }),
    [result, selectedPlayer, selectedTeam],
  )

  useEffect(() => {
    setDiagnosis('')
    setQuestion('')
    setConversation([])
    setResponseError(undefined)
    setSuggestedQuestions(DEFAULT_SUGGESTED_QUESTIONS)

    void statusTask.run(fetchCoachStatus).then((status) => {
      if (!status) {
        return
      }
      setResponseError(status.configured ? undefined : status.message)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coachRequest.matchId, coachRequest.player, coachRequest.provider, coachRequest.team])

  async function handleGenerateDiagnosis() {
    setResponseError(undefined)
    const response = await diagnosisTask.run(() => generateCoachDiagnosis(coachRequest))
    if (!response) {
      return
    }

    if (!response.ok) {
      setResponseError(response.error || 'No se pudo generar el diagnóstico táctico.')
      return
    }

    setDiagnosis(response.diagnosis)
    if (response.suggested_questions.length > 0) {
      setSuggestedQuestions(response.suggested_questions)
    }
  }

  async function submitQuestion(nextQuestion: string) {
    const normalizedQuestion = nextQuestion.trim()
    if (!normalizedQuestion) {
      setResponseError('La pregunta no puede estar vacía.')
      return
    }

    setResponseError(undefined)
    const response = await questionTask.run(() =>
      askCoachQuestion({
        ...coachRequest,
        question: normalizedQuestion,
        conversationHistory: conversation,
      }),
    )
    if (!response) {
      return
    }

    if (!response.ok) {
      setResponseError(response.error || 'No se pudo responder la pregunta.')
      return
    }

    setConversation((current) => [
      ...current,
      { role: 'user', content: normalizedQuestion },
      { role: 'assistant', content: response.answer },
    ])
    setQuestion('')
    if (response.suggested_questions.length > 0) {
      setSuggestedQuestions(response.suggested_questions)
    }
  }

  const isConfigured = statusTask.data?.configured ?? false
  const actionDisabled = !isConfigured || diagnosisTask.loading || questionTask.loading
  const visibleError = responseError || diagnosisTask.error || questionTask.error || statusTask.error

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="space-y-1">
          <h3 className="text-lg font-semibold text-slate-100">AI Tactical Coach</h3>
          <p className="text-sm text-slate-300">
            Consultá el partido con contexto táctico estructurado desde el backend actual.
          </p>
        </div>
        <div className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">
          {statusTask.loading ? 'Verificando configuración...' : statusTask.data?.message || 'Sin estado'}
        </div>
      </div>

      {statusTask.loading ? <LoadingState label="Verificando configuración del AI Coach..." /> : null}
      {visibleError ? <ErrorState message={visibleError} /> : null}

      <div className="flex flex-wrap gap-2">
        <button
          className="rounded-md border border-emerald-500/50 bg-emerald-500/10 px-4 py-2 text-sm font-medium text-emerald-200 hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={actionDisabled}
          onClick={() => void handleGenerateDiagnosis()}
          type="button"
        >
          Generar diagnóstico táctico
        </button>
      </div>

      {diagnosisTask.loading ? <LoadingState label="Generando diagnóstico táctico..." /> : null}
      {diagnosis ? (
        <article className="rounded-xl border border-slate-700 bg-slate-950/60 p-4">
          <h4 className="text-sm font-semibold uppercase tracking-[0.18em] text-emerald-300">
            Diagnóstico táctico
          </h4>
          <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-200">{diagnosis}</p>
        </article>
      ) : null}

      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {suggestedQuestions.map((suggestedQuestion) => (
            <button
              className="rounded-full border border-slate-700 px-3 py-1.5 text-xs text-slate-200 hover:border-emerald-400 hover:text-emerald-200 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={actionDisabled}
              key={suggestedQuestion}
              onClick={() => void submitQuestion(suggestedQuestion)}
              type="button"
            >
              {suggestedQuestion}
            </button>
          ))}
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-slate-200" htmlFor="ai-coach-question">
            Preguntale algo al AI Coach
          </label>
          <textarea
            className="min-h-24 w-full rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-emerald-400"
            disabled={!isConfigured || questionTask.loading}
            id="ai-coach-question"
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ej.: ¿Dónde estuvo la mayor amenaza ofensiva y qué debería ajustar el cuerpo técnico?"
            value={question}
          />
          <div className="flex justify-end">
            <button
              className="rounded-md border border-sky-500/50 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-200 hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={actionDisabled || !question.trim()}
              onClick={() => void submitQuestion(question)}
              type="button"
            >
              Preguntar
            </button>
          </div>
        </div>
      </div>

      {questionTask.loading ? <LoadingState label="Consultando al AI Coach..." /> : null}
      {conversation.length > 0 ? (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">
            Conversación
          </h4>
          {conversation.map((message, index) => (
            <article
              className={`rounded-xl border p-3 text-sm ${
                message.role === 'assistant'
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-slate-100'
                  : 'border-slate-700 bg-slate-950/70 text-slate-200'
              }`}
              key={`${message.role}-${index}`}
            >
              <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                {message.role === 'assistant' ? 'AI Coach' : 'Pregunta'}
              </p>
              <p className="whitespace-pre-wrap leading-6">{message.content}</p>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}
