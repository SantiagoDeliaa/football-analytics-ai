import { Link } from 'react-router-dom'
import {
  loadVertical1Page,
  loadVertical2Page,
  scheduleModulePrefetch,
} from '../app/modulePreload'

function HomeCard({
  title,
  description,
  to,
  disabled,
  onPrefetch,
}: {
  title: string
  description: string
  to: string
  disabled?: boolean
  onPrefetch?: () => void
}) {
  const baseClass =
    'group relative overflow-hidden rounded-[28px] border p-8 text-left shadow-2xl transition hover:-translate-y-1'
  const enabledClass =
    'border-slate-700 bg-[linear-gradient(145deg,rgba(21,19,40,0.95),rgba(13,42,39,0.9))] hover:border-emerald-400/60 hover:shadow-emerald-900/20'
  const disabledClass =
    'border-slate-700 bg-[linear-gradient(140deg,rgba(7,17,32,0.95),rgba(10,38,55,0.88))] opacity-70'

  if (disabled) {
    return (
      <article className={`${baseClass} ${disabledClass}`}>
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_14%_16%,rgba(255,255,255,0.12),transparent_32%),radial-gradient(circle_at_86%_84%,rgba(255,255,255,0.08),transparent_38%)]" />
        <h2 className="text-2xl font-bold text-slate-100">{title}</h2>
        <p className="relative mt-3 max-w-sm text-sm leading-6 text-slate-300">{description}</p>
        <p className="relative mt-6 inline-flex rounded-full border border-slate-600 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
          Próximamente
        </p>
      </article>
    )
  }

  return (
    <Link
      className={`${baseClass} ${enabledClass}`}
      onFocus={onPrefetch}
      onMouseEnter={onPrefetch}
      to={to}
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_14%_16%,rgba(255,255,255,0.12),transparent_32%),radial-gradient(circle_at_86%_84%,rgba(255,255,255,0.08),transparent_38%)]" />
      <h2 className="text-2xl font-bold text-slate-100">{title}</h2>
      <p className="relative mt-3 max-w-sm text-sm leading-6 text-slate-300">{description}</p>
      <p className="relative mt-6 inline-flex rounded-full border border-emerald-400/40 bg-slate-950/40 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300">
        Open Module
      </p>
    </Link>
  )
}

export function HomePage() {
  return (
    <section className="space-y-8">
      <header className="relative overflow-hidden rounded-[32px] border border-emerald-500/15 bg-slate-950/60 px-6 py-10 text-center md:px-10 md:py-14">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_18%_15%,rgba(36,88,62,0.16),transparent_36%),radial-gradient(circle_at_82%_88%,rgba(24,79,58,0.14),transparent_35%),linear-gradient(180deg,#0a1016_0%,#0b131b_45%,#0a1119_100%)]" />
        <div className="pointer-events-none absolute inset-[18px] rounded-[28px] border border-emerald-300/10 bg-[radial-gradient(circle_at_50%_50%,transparent_0_62px,rgba(126,227,187,0.08)_62px_64px,transparent_64px),linear-gradient(90deg,transparent_5%,rgba(126,227,187,0.04)_6%,transparent_7%),linear-gradient(0deg,transparent_34%,rgba(126,227,187,0.04)_35%,transparent_36%)]" />
        <div className="relative">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-emerald-300">TIP Platform</p>
          <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-slate-100 md:text-6xl">
            <span className="text-emerald-400">T</span>actical <span className="text-emerald-400">I</span>
            ntelligence <span className="text-emerald-400">P</span>latform
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-sm leading-6 text-slate-300 md:text-base">
            Plataforma de inteligencia táctica para análisis de fútbol con una UX moderna, escalable y
            alineada al look & feel premium del proyecto original.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <HomeCard
          description="Tracking y métricas tácticas desde video broadcast."
          onPrefetch={() => scheduleModulePrefetch(loadVertical1Page)}
          title="Computer Vision"
          to="/vertical1"
        />
        <HomeCard
          description="Insights tácticos y métricas propietarias desde event data y reportes."
          onPrefetch={() => scheduleModulePrefetch(loadVertical2Page)}
          title="Data Analytics"
          to="/vertical2"
        />
      </div>
    </section>
  )
}
