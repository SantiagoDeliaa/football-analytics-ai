interface MetricCardProps {
  title: string
  value: string
  subtitle?: string
}

export function MetricCard({ title, value, subtitle }: MetricCardProps) {
  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <p className="text-xs uppercase tracking-wider text-slate-400">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-50">{value}</p>
      {subtitle ? <p className="mt-2 text-xs text-slate-300">{subtitle}</p> : null}
    </article>
  )
}
