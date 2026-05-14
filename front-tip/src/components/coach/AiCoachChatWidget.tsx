import { useEffect, useMemo, useRef } from 'react'
import { useLocation } from 'react-router-dom'
import { ErrorState } from '../common/ErrorState'
import { LoadingState } from '../common/LoadingState'
import { useAiCoachChat } from '../../app/AiCoachChatContext'

function formatLocationLabel(pathname: string) {
  if (pathname.startsWith('/vertical1')) {
    return 'Computer Vision'
  }

  if (pathname.startsWith('/vertical2')) {
    return 'Data Analytics'
  }

  return 'Inicio'
}

export function AiCoachChatWidget() {
  const location = useLocation()
  const conversationEndRef = useRef<HTMLDivElement | null>(null)
  const {
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
  } = useAiCoachChat()

  const locationLabel = useMemo(() => formatLocationLabel(location.pathname), [location.pathname])
  const actionDisabled = !hasActiveContext || !isConfigured || diagnosisLoading || questionLoading

  useEffect(() => {
    if (!pendingQuestion || questionLoading || statusLoading) {
      return
    }

    void submitQuestion(pendingQuestion).finally(() => {
      clearPendingQuestion()
    })
  }, [clearPendingQuestion, pendingQuestion, questionLoading, statusLoading, submitQuestion])

  useEffect(() => {
    if (!isOpen) {
      return
    }

    if (typeof conversationEndRef.current?.scrollIntoView === 'function') {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [conversation.length, diagnosis, isOpen])

  async function handleSubmitCurrentDraft() {
    const submitted = await submitQuestion(draft)
    if (submitted) {
      return
    }
  }

  return (
    <div className="pointer-events-none fixed inset-x-3 bottom-3 z-30 sm:inset-x-auto sm:right-4">
      <div
        className={`pointer-events-auto transition-all duration-300 ease-out ${
          isOpen
            ? 'translate-y-0 opacity-100'
            : 'pointer-events-none translate-y-4 opacity-0 sm:translate-y-6'
        }`}
      >
        <section className="flex h-[min(78vh,42rem)] w-full flex-col overflow-hidden rounded-[26px] border border-emerald-500/20 bg-slate-950/95 shadow-[0_24px_70px_rgba(15,23,42,0.55)] backdrop-blur-xl sm:w-[25rem] md:w-[29rem]">
          <header className="border-b border-slate-800 bg-slate-950/90 px-4 py-4">
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
                  <h2 className="text-sm font-semibold uppercase tracking-[0.22em] text-emerald-200">
                    AI Coach
                  </h2>
                </div>
                <p className="text-sm text-slate-300">
                  Disponible en toda la navegación. Vista actual: {locationLabel}.
                </p>
                <p className="text-xs text-slate-400">
                  {statusLoading ? 'Verificando configuración...' : status?.message || 'Sin estado'}
                </p>
              </div>
              <button
                aria-label="Minimizar AI Coach"
                className="rounded-full border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-200 transition hover:border-slate-500 hover:bg-slate-900"
                onClick={() => setIsOpen(false)}
                type="button"
              >
                Minimizar
              </button>
            </div>
            <div className="mt-3 rounded-2xl border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-300">
              {activeContext ? (
                <>
                  <p className="font-semibold text-slate-100">Contexto activo</p>
                  <p className="mt-1">{activeContext.summary}</p>
                  {locationLabel !== activeContext.sourceLabel ? (
                    <p className="mt-1 text-slate-400">
                      Seguís conversando con el contexto cargado en {activeContext.sourceLabel}.
                    </p>
                  ) : null}
                </>
              ) : (
                <>
                  <p className="font-semibold text-slate-100">Sin contexto táctico activo</p>
                  <p className="mt-1">
                    Cargá un partido en Data Analytics para habilitar preguntas contextualizadas. El chat
                    queda disponible durante toda la sesión.
                  </p>
                </>
              )}
            </div>
          </header>

          <div className="flex-1 overflow-y-auto px-4 py-4">
            {statusLoading ? <LoadingState label="Verificando configuración del AI Coach..." /> : null}
            {responseError ? <ErrorState message={responseError} /> : null}

            {hasActiveContext ? (
              <div className="mb-4 flex flex-wrap gap-2">
                <button
                  className="rounded-md border border-emerald-500/50 bg-emerald-500/10 px-4 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={actionDisabled}
                  onClick={() => void triggerDiagnosis()}
                  type="button"
                >
                  Generar diagnóstico táctico
                </button>
                {suggestedQuestions.slice(0, 3).map((suggestedQuestion) => (
                  <button
                    className="rounded-full border border-slate-700 px-3 py-1.5 text-xs text-slate-200 transition hover:border-emerald-400 hover:text-emerald-200 disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={actionDisabled}
                    key={suggestedQuestion}
                    onClick={() => requestQuestion(suggestedQuestion)}
                    type="button"
                  >
                    {suggestedQuestion}
                  </button>
                ))}
              </div>
            ) : null}

            {diagnosisLoading ? <LoadingState label="Generando diagnóstico táctico..." /> : null}
            {diagnosis ? (
              <article className="mb-4 rounded-2xl border border-emerald-500/25 bg-emerald-500/10 p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-300">
                  Diagnóstico táctico
                </p>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-100">{diagnosis}</p>
              </article>
            ) : null}

            {conversation.length > 0 ? (
              <div className="space-y-3">
                {conversation.map((message) => (
                  <article
                    className={`rounded-2xl border p-3 text-sm ${
                      message.role === 'assistant'
                        ? 'border-emerald-500/30 bg-emerald-500/10 text-slate-100'
                        : 'border-slate-700 bg-slate-900/80 text-slate-200'
                    }`}
                    key={message.id}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                        {message.role === 'assistant' ? 'AI Coach' : 'Pregunta'}
                      </p>
                      <span className="rounded-full border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400">
                        {message.contextLabel}
                      </span>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap leading-6">{message.content}</p>
                  </article>
                ))}
                <div ref={conversationEndRef} />
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/60 p-4 text-sm leading-6 text-slate-300">
                El historial del chat se conserva durante toda la sesión. Abrí una conversación desde
                cualquier vertical y retomala cuando quieras.
              </div>
            )}
          </div>

          <footer className="border-t border-slate-800 bg-slate-950/90 px-4 py-4">
            <label className="block text-sm font-medium text-slate-200" htmlFor="ai-coach-floating-input">
              Preguntale algo al AI Coach
            </label>
            <textarea
              className="mt-2 min-h-24 w-full rounded-2xl border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-emerald-400"
              disabled={!hasActiveContext || !isConfigured || questionLoading}
              id="ai-coach-floating-input"
              onChange={(event) => setDraft(event.target.value)}
              placeholder={
                hasActiveContext
                  ? 'Ej.: ¿Dónde estuvo la mayor amenaza ofensiva y qué debería ajustar el cuerpo técnico?'
                  : 'Cargá primero un partido en Data Analytics para habilitar el contexto.'
              }
              value={draft}
            />
            <div className="mt-3 flex items-center justify-between gap-3">
              <p className="text-xs text-slate-400">
                {hasActiveContext
                  ? 'La conversación persiste entre rutas y mantiene el contexto activo.'
                  : 'Todavía no hay un partido cargado para contextualizar al coach.'}
              </p>
              <button
                className="rounded-md border border-sky-500/50 bg-sky-500/10 px-4 py-2 text-sm font-medium text-sky-200 transition hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={actionDisabled || !draft.trim()}
                onClick={() => void handleSubmitCurrentDraft()}
                type="button"
              >
                Preguntar
              </button>
            </div>
          </footer>
        </section>
      </div>

      <button
        aria-label={isOpen ? 'Ocultar AI Coach' : 'Abrir AI Coach'}
        className={`pointer-events-auto ml-auto flex items-center gap-3 rounded-full border px-4 py-3 text-sm font-semibold shadow-lg transition-all duration-300 ${
          isOpen
            ? 'translate-y-3 border-emerald-500/0 bg-transparent text-transparent opacity-0'
            : 'translate-y-0 border-emerald-400/40 bg-slate-950/95 text-emerald-100 opacity-100 hover:border-emerald-300 hover:bg-slate-900'
        }`}
        onClick={toggleOpen}
        type="button"
      >
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/15 text-lg text-emerald-300">
          AI
        </span>
        <span className="hidden text-left sm:block">
          Chat persistente
          <span className="block text-xs font-normal text-slate-400">
            {conversation.length > 0
              ? `${Math.floor(conversation.length / 2)} intercambio${conversation.length > 2 ? 's' : ''} guardado${conversation.length > 2 ? 's' : ''}`
              : 'Listo para acompañarte'}
          </span>
        </span>
      </button>
    </div>
  )
}
