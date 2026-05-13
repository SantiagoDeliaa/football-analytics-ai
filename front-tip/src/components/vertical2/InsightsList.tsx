export function InsightsList({ insights }: { insights: string[] }) {
  if (!insights.length) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 text-sm text-slate-300">
        No hay insights disponibles para esta selección.
      </div>
    )
  }

  return (
    <ul className="space-y-2 rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      {insights.map((insight) => (
        <li className="text-sm text-slate-200" key={insight}>
          - {insight}
        </li>
      ))}
    </ul>
  )
}
