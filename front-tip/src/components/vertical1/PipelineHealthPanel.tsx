import type { ComputerVisionResult } from '../../types/computerVision'
import {
  formatFramesWithShare,
  formatRatioAsPercentage,
  formatValue,
  getHomographyOverview,
} from '../../utils/computerVisionInsights'

export function PipelineHealthPanel({ result }: { result: ComputerVisionResult }) {
  const homographyOverview = getHomographyOverview(result.health_summary)
  const cards = [
    {
      label: 'Frames válidos',
      value: formatFramesWithShare(result.health_summary.valid_frames, result.health_summary.total_frames ?? result.total_frames),
      hint: 'homografía usable',
    },
    {
      label: 'Fallback',
      value: formatRatioAsPercentage(result.health_summary.fallback_ratio),
      hint: 'frames con fallback',
    },
    {
      label: 'Warnings',
      value: formatRatioAsPercentage(result.health_summary.warn_ratio),
      hint: 'frames con alertas',
    },
    {
      label: 'Tracking',
      value: formatRatioAsPercentage(result.health_summary.p95_churn_ratio, 0),
      hint: 'churn ratio',
    },
    {
      label: 'Velocidad',
      value: formatValue(result.health_summary.p95_max_speed_mps, 'm/s'),
      hint: 'p95 speed',
    },
    {
      label: 'Posesión',
      value: formatRatioAsPercentage(result.health_summary.ball_detected_ratio, 0),
      hint: 'ball detected',
    },
  ]

  const toneClasses = {
    emerald: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100',
    amber: 'border-amber-500/30 bg-amber-500/10 text-amber-100',
    rose: 'border-rose-500/30 bg-rose-500/10 text-rose-100',
  } as const

  return (
    <section className="space-y-3">
      <header className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Salud del pipeline</h3>
        <p className="mt-2 text-sm text-slate-300">
          La lectura sigue el orden operativo del proyecto: homografía, tracking, velocidad, posesión y formación.
        </p>
      </header>

      <article className={`rounded-2xl border p-4 ${toneClasses[homographyOverview.tone]}`}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="inline-flex rounded-full border border-current/30 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]">
              {homographyOverview.badge}
            </div>
            <h4 className="mt-3 text-base font-semibold">{homographyOverview.title}</h4>
            <p className="mt-2 max-w-3xl text-sm text-current/90">{homographyOverview.description}</p>
          </div>
          <div className="rounded-xl border border-current/20 bg-slate-950/30 px-4 py-3 text-right">
            <p className="text-xs uppercase tracking-[0.18em] text-current/70">Reproj error p95</p>
            <p className="mt-2 text-2xl font-semibold">
              {formatValue(result.health_summary.p95_reproj_error_m, 'm')}
            </p>
          </div>
        </div>
      </article>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        {cards.map((card) => (
          <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4" key={card.label}>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{card.label}</p>
            <p className="mt-2 text-2xl font-semibold text-slate-100">{card.value}</p>
            <p className="mt-2 text-xs text-slate-300">{card.hint}</p>
          </article>
        ))}
      </div>

      {result.warnings.length ? (
        <div className="space-y-2 rounded-2xl border border-amber-700/40 bg-amber-900/20 p-4">
          <h4 className="text-sm font-semibold text-amber-100">Alertas operativas</h4>
          <ul className="space-y-2">
            {result.warnings.map((warning) => (
              <li className="text-sm text-amber-50" key={warning}>
                - {warning}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  )
}
