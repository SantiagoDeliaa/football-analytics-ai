export function LoadingState({ label = 'Cargando datos...' }: { label?: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 text-sm text-slate-200">
      <div className="flex items-center gap-3">
        <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-emerald-400 border-t-transparent" />
        <span>{label}</span>
      </div>
    </div>
  )
}
