interface ErrorStateProps {
  message: string
  onRetry?: () => void
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="rounded-xl border border-rose-700/60 bg-rose-900/30 p-4">
      <p className="text-sm text-rose-100">{message}</p>
      {onRetry ? (
        <button
          className="mt-3 rounded-md border border-rose-500 px-3 py-1.5 text-sm text-rose-50 hover:bg-rose-700/40"
          onClick={onRetry}
          type="button"
        >
          Reintentar
        </button>
      ) : null}
    </div>
  )
}
