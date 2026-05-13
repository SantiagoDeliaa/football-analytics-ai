import type { ComputerVisionResult } from '../../types/computerVision'

export function ScoutingPanels({ result }: { result: ComputerVisionResult }) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <h3 className="text-lg font-semibold text-slate-100">Scouting</h3>
      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <ScoutingTeamCard
          bullets={result.scouting.bullets.team1}
          confidence={result.scouting.confidence.team1}
          team="Team 1"
        />
        <ScoutingTeamCard
          bullets={result.scouting.bullets.team2}
          confidence={result.scouting.confidence.team2}
          team="Team 2"
        />
      </div>
    </section>
  )
}

export function PossessionPanel({ result }: { result: ComputerVisionResult }) {
  if (!result.possession) {
    return (
      <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Posesión</h3>
        <p className="mt-2 text-sm text-slate-300">
          La posesión no está disponible para este resultado o fue desactivada en la corrida.
        </p>
      </section>
    )
  }

  const bars = [
    { label: 'Team 1', value: result.possession.team1_pct, color: 'bg-emerald-400' },
    { label: 'Team 2', value: result.possession.team2_pct, color: 'bg-sky-400' },
    { label: 'Contestado', value: result.possession.contested_pct, color: 'bg-amber-400' },
  ]

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div>
        <h3 className="text-lg font-semibold text-slate-100">Posesión</h3>
        <p className="mt-2 text-sm text-slate-300">
          El reparto sólo es confiable si la señal del balón se mantiene estable en el clip.
        </p>
      </div>

      <div className="space-y-3">
        {bars.map((bar) => (
          <div key={bar.label}>
            <div className="mb-1 flex items-center justify-between text-sm text-slate-200">
              <span>{bar.label}</span>
              <span>{bar.value}%</span>
            </div>
            <div className="h-3 rounded-full bg-slate-950/80">
              <div className={`h-3 rounded-full ${bar.color}`} style={{ width: `${bar.value}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export function InterpretationPanel({ result }: { result: ComputerVisionResult }) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <h3 className="text-lg font-semibold text-slate-100">Interpretación</h3>
      <ul className="mt-4 space-y-2">
        {result.interpretation.map((item) => (
          <li className="text-sm text-slate-200" key={item}>
            - {item}
          </li>
        ))}
      </ul>
    </section>
  )
}

function ScoutingTeamCard({
  team,
  confidence,
  bullets,
}: {
  team: string
  confidence: { label: string; score: number }
  bullets: string[]
}) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{team}</p>
      <p className="mt-2 text-sm font-semibold text-slate-100">
        Confianza {confidence.label} ({confidence.score}/100)
      </p>
      <ul className="mt-3 space-y-2">
        {bullets.map((item) => (
          <li className="text-sm text-slate-200" key={item}>
            - {item}
          </li>
        ))}
      </ul>
    </article>
  )
}
