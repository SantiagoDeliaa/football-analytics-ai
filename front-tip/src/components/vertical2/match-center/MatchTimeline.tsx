import type { MatchTimelineEvent } from '../../../types/matchCenter'
import { formatNullableValue } from '../../../utils/matchCenter'

interface MatchTimelineProps {
  timeline?: MatchTimelineEvent[]
}

function formatMinute(event: MatchTimelineEvent) {
  if (event.minute === null || event.minute === undefined) {
    return '-'
  }
  if (event.extra_minute === null || event.extra_minute === undefined || event.extra_minute === 0) {
    return `${event.minute}`
  }
  return `${event.minute}+${event.extra_minute}`
}

export function MatchTimeline({ timeline }: MatchTimelineProps) {
  if (!timeline?.length) {
    return (
      <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
        <h2 className="text-lg font-semibold text-slate-50">Timeline de eventos</h2>
        <p className="mt-3 text-sm text-slate-300">
          No hay eventos principales disponibles para este partido.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="text-lg font-semibold text-slate-50">Timeline de eventos</h2>
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-slate-300">
              <th className="px-3 py-2">Minuto</th>
              <th className="px-3 py-2">Evento</th>
              <th className="px-3 py-2">Equipo</th>
              <th className="px-3 py-2">Jugador</th>
              <th className="px-3 py-2">Relacionado</th>
              <th className="px-3 py-2">Resultado</th>
              <th className="px-3 py-2">Descripción</th>
            </tr>
          </thead>
          <tbody>
            {timeline.map((event, index) => (
              <tr className="border-b border-slate-800/80 align-top" key={`${event.minute}-${event.player_name}-${index}`}>
                <td className="px-3 py-2 text-slate-100">{formatMinute(event)}</td>
                <td className="px-3 py-2 text-slate-100">{formatNullableValue(event.event_label)}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(event.team_name)}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(event.player_name)}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(event.related_player_name, '-')}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(event.result, '-')}</td>
                <td className="px-3 py-2 text-slate-300">{formatNullableValue(event.description, '-')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
