interface InsightsPanelProps {
  insights?: string[]
}

export function InsightsPanel({ insights }: InsightsPanelProps) {
  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="text-lg font-semibold text-slate-50">Insights automáticos</h2>
      {!insights?.length ? (
        <p className="mt-3 text-sm text-slate-300">
          No hay insights automáticos disponibles para este partido.
        </p>
      ) : (
        <ul className="mt-4 space-y-2 text-sm text-slate-200">
          {insights.map((insight, index) => (
            <li className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2" key={`${index}-${insight}`}>
              {insight}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
