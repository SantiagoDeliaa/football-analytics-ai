interface MatchSummaryProps {
  competitionName: string
  matchLabel: string
  team: string
  player: string
  sourceLabel: string
}

export function MatchSummary({
  competitionName,
  matchLabel,
  team,
  player,
  sourceLabel,
}: MatchSummaryProps) {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <SummaryItem label="Competición" value={competitionName} />
        <SummaryItem label="Partido" value={matchLabel} />
        <SummaryItem label="Equipo" value={team} />
        <SummaryItem label="Jugador" value={player} />
        <SummaryItem label="Origen" value={sourceLabel} />
      </div>
    </section>
  )
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm font-semibold text-slate-100">{value}</p>
    </div>
  )
}
