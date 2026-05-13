import type { EventDataResult, EventHistoryEntry, ProcessedHistoryMatch, ProviderOption } from '../../types/eventData'
import { formatDateTime } from '../../utils/formatters'

interface HistoryAvailability {
  localEntry?: EventHistoryEntry
  backendEntry?: ProcessedHistoryMatch
  onLoadLocal?: (entry: EventHistoryEntry) => void
  onLoadBackend?: (provider: ProviderOption, matchId: string) => void
}

interface ProviderDebugPanelProps {
  result: EventDataResult
  history?: HistoryAvailability
}

const APP_ENV = (import.meta.env.VITE_APP_ENV ?? 'development').toLowerCase()
const SHOW_DEMO_OPS = APP_ENV === 'development'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function getNestedName(value: unknown, fallback = '') {
  if (typeof value === 'string') {
    return value
  }
  if (isRecord(value) && typeof value.name === 'string') {
    return value.name
  }
  return fallback
}

function buildArrayPreview(items: unknown[], limit: number) {
  return items.slice(0, limit)
}

function formatCoordinate(value: number | null | undefined) {
  return typeof value === 'number' ? value.toFixed(1) : 'N/D'
}

function summarizeApiFootballEvents(rawEvents: unknown[]) {
  const summary = { goals: 0, cards: 0, substitutions: 0 }
  rawEvents.forEach((event) => {
    const raw = isRecord(event) ? event : {}
    const eventType = String(raw.type ?? '').toLowerCase()
    const detail = String(raw.detail ?? '').toLowerCase()
    if (eventType.includes('goal') || detail.includes('goal')) {
      summary.goals += 1
    }
    if (eventType.includes('card') || detail.includes('card')) {
      summary.cards += 1
    }
    if (eventType.includes('subst') || eventType.includes('substitution') || detail.includes('subst')) {
      summary.substitutions += 1
    }
  })
  return summary
}

function summarizeStatsBomb(rawEvents: unknown[]) {
  const types = new Set<string>()
  const teams = new Set<string>()
  let eventsWithMinute = 0
  let eventsWithTeam = 0

  rawEvents.forEach((event) => {
    const raw = isRecord(event) ? event : {}
    const typeName = getNestedName(raw.type, String(raw.type ?? ''))
    const teamName = getNestedName(raw.team, String(raw.team ?? ''))

    if (typeName) {
      types.add(typeName)
    }
    if (teamName) {
      teams.add(teamName)
      eventsWithTeam += 1
    }
    if (raw.minute !== undefined && raw.minute !== null) {
      eventsWithMinute += 1
    }
  })

  return {
    uniqueTypes: types.size,
    uniqueTeams: teams.size,
    eventsWithMinute,
    eventsWithTeam,
  }
}

function buildCanonicalPreviewRows(events: EventDataResult['canonical_events']) {
  return events.slice(0, 12).map((event) => [
    `${event.minute}:${`${event.second}`.padStart(2, '0')}`,
    event.team_name || 'Sin equipo',
    event.player_name || 'Jugador desconocido',
    event.event_type || 'Evento',
    formatCoordinate(event.x),
    formatCoordinate(event.y),
  ])
}

function buildApiFootballTimelineRows(rawEvents: unknown[]) {
  return rawEvents.slice(0, 12).map((event) => {
    const raw = isRecord(event) ? event : {}
    const time = isRecord(raw.time) ? raw.time : {}
    return [
      `${time.elapsed ?? 'N/D'}`,
      getNestedName(raw.team, 'Sin equipo'),
      getNestedName(raw.player, 'Jugador desconocido'),
      `${raw.type ?? 'Evento'}`,
      `${raw.detail ?? raw.comments ?? 'N/D'}`,
    ]
  })
}

function buildApiFootballFormationRows(rawLineups: unknown[]) {
  return rawLineups.slice(0, 4).map((lineup) => {
    const raw = isRecord(lineup) ? lineup : {}
    const starters = asArray(raw.startXI)
      .slice(0, 5)
      .map((entry) => getNestedName(isRecord(entry) ? entry.player : undefined))
      .filter(Boolean)
    return [
      getNestedName(raw.team, 'Equipo'),
      `${raw.formation ?? 'Sin formación reportada'}`,
      `${asArray(raw.startXI).length}`,
      `${asArray(raw.substitutes).length}`,
      starters.length ? starters.join(', ') : 'Sin titulares reportados',
    ]
  })
}

function buildApiFootballStatisticsRows(rawStatistics: unknown[]) {
  const rows: string[][] = []
  rawStatistics.slice(0, 2).forEach((teamBlock) => {
    const rawTeam = isRecord(teamBlock) ? teamBlock : {}
    const teamName = getNestedName(rawTeam.team, 'Equipo')
    const stats = asArray(rawTeam.statistics).slice(0, 8)
    stats.forEach((stat, index) => {
      const rawStat = isRecord(stat) ? stat : {}
      rows.push([
        teamName,
        `${rawStat.type ?? `Métrica ${index + 1}`}`,
        `${rawStat.value ?? 'N/D'}`,
      ])
    })
  })
  return rows
}

function buildApiFootballPlayersRows(rawPlayers: unknown[]) {
  const rows: string[][] = []
  rawPlayers.slice(0, 2).forEach((teamBlock) => {
    const rawTeam = isRecord(teamBlock) ? teamBlock : {}
    const teamName = getNestedName(rawTeam.team, 'Equipo')
    const players = asArray(rawTeam.players).slice(0, 8)
    players.forEach((playerBlock) => {
      const rawPlayerBlock = isRecord(playerBlock) ? playerBlock : {}
      const player = isRecord(rawPlayerBlock.player) ? rawPlayerBlock.player : {}
      rows.push([
        teamName,
        `${player.name ?? 'Jugador desconocido'}`,
        `${player.age ?? 'N/D'}`,
        `${player.pos ?? 'N/D'}`,
        `${player.number ?? 'N/D'}`,
      ])
    })
  })
  return rows
}

function buildStatsBombTimelineRows(rawEvents: unknown[]) {
  return rawEvents.slice(0, 12).map((event) => {
    const raw = isRecord(event) ? event : {}
    return [
      `${raw.minute ?? 'N/D'}`,
      getNestedName(raw.team, 'Sin equipo'),
      getNestedName(raw.player, 'Jugador desconocido'),
      getNestedName(raw.type, String(raw.type ?? 'Evento')),
      getNestedName(raw.outcome, String(raw.outcome ?? 'N/D')),
    ]
  })
}

export function ProviderDebugPanel({ result, history }: ProviderDebugPanelProps) {
  if (!SHOW_DEMO_OPS) {
    return null
  }

  const rawPayload = result.raw_payload
  const localHistoryEntry = history?.localEntry
  const backendHistoryEntry = history?.backendEntry
  const rawEvents = Array.isArray(rawPayload)
    ? rawPayload
    : isRecord(rawPayload)
      ? asArray(rawPayload.events)
      : []
  const rawLineups = isRecord(rawPayload) ? asArray(rawPayload.lineups) : []
  const rawStatistics = isRecord(rawPayload) ? asArray(rawPayload.statistics) : []
  const rawPlayers = isRecord(rawPayload) ? asArray(rawPayload.players) : []
  const apiFootballSummary = result.provider === 'API-Football' ? summarizeApiFootballEvents(rawEvents) : undefined
  const statsBombSummary = result.provider === 'StatsBomb Open Data' ? summarizeStatsBomb(rawEvents) : undefined
  const canonicalPreviewRows = buildCanonicalPreviewRows(result.canonical_events)

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-300">Demo Ops</p>
          <h3 className="mt-2 text-lg font-semibold text-slate-100">Panel técnico del provider</h3>
          <p className="mt-1 text-sm text-slate-300">
            Vista de soporte para validar cobertura del provider, payloads crudos y puntos de reuso
            durante la demo.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge label={result.provider} tone="sky" />
          <Badge label={`${result.raw_events_count} eventos`} tone="emerald" />
          {history?.localEntry ? <Badge label="Disponible en historial local" tone="amber" /> : null}
          {history?.backendEntry ? <Badge label="Disponible en historial backend" tone="violet" /> : null}
        </div>
      </div>

      {(history?.localEntry || history?.backendEntry) && (history?.onLoadLocal || history?.onLoadBackend) ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {localHistoryEntry && history.onLoadLocal ? (
            <article className="rounded-xl border border-amber-700/40 bg-amber-900/20 p-4">
              <p className="text-sm font-semibold text-amber-50">Historial local disponible</p>
              <p className="mt-1 text-sm text-amber-100">
                Guardado: {formatDateTime(localHistoryEntry.saved_at)}
              </p>
              <button
                className="mt-3 rounded-md border border-amber-300/40 px-3 py-2 text-sm font-semibold text-amber-50 hover:bg-amber-800/30"
                onClick={() => history.onLoadLocal?.(localHistoryEntry)}
                type="button"
              >
                Cargar historial local
              </button>
            </article>
          ) : null}

          {backendHistoryEntry && history.onLoadBackend ? (
            <article className="rounded-xl border border-violet-700/40 bg-violet-900/20 p-4">
              <p className="text-sm font-semibold text-violet-50">Historial backend disponible</p>
              <p className="mt-1 text-sm text-violet-100">
                Actualizado: {formatDateTime(backendHistoryEntry.updated_at)}
              </p>
              <button
                className="mt-3 rounded-md border border-violet-300/40 px-3 py-2 text-sm font-semibold text-violet-50 hover:bg-violet-800/30"
                onClick={() => history.onLoadBackend?.(backendHistoryEntry.provider, backendHistoryEntry.match_id)}
                type="button"
              >
                Cargar historial backend
              </button>
            </article>
          ) : null}
        </div>
      ) : null}

      <section className="space-y-3">
        <SectionTitle
          description="Chequeo rápido para confirmar qué llegó del provider antes de profundizar en tablas o JSON."
          title="Cobertura técnica"
        />
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Eventos crudos" value={`${rawEvents.length}`} />
          <MetricCard label="Eventos canónicos" value={`${result.canonical_events.length}`} />
          <MetricCard label="Lineups" value={`${rawLineups.length}`} />
          <MetricCard label="Bloques stats" value={`${rawStatistics.length}`} />
          {result.provider === 'API-Football' && apiFootballSummary ? (
            <>
              <MetricCard label="Goles detectados" value={`${apiFootballSummary.goals}`} />
              <MetricCard label="Tarjetas detectadas" value={`${apiFootballSummary.cards}`} />
              <MetricCard label="Sustituciones" value={`${apiFootballSummary.substitutions}`} />
              <MetricCard label="Bloques jugadores" value={`${rawPlayers.length}`} />
            </>
          ) : null}
          {result.provider === 'StatsBomb Open Data' && statsBombSummary ? (
            <>
              <MetricCard label="Tipos detectados" value={`${statsBombSummary.uniqueTypes}`} />
              <MetricCard label="Equipos en raw" value={`${statsBombSummary.uniqueTeams}`} />
              <MetricCard label="Con minuto" value={`${statsBombSummary.eventsWithMinute}`} />
              <MetricCard label="Con equipo" value={`${statsBombSummary.eventsWithTeam}`} />
            </>
          ) : null}
        </div>
      </section>

      {result.provider === 'API-Football' ? (
        <>
          <section className="space-y-3">
            <SectionTitle
              description="Lectura literal del provider para ver timeline, formaciones, estadísticas y jugadores."
              title="Tablas técnicas"
            />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <DataTable
                columns={['Min', 'Equipo', 'Jugador', 'Tipo', 'Detalle']}
                rows={buildApiFootballTimelineRows(rawEvents)}
                title="Timeline de eventos"
              />
              <DataTable
                columns={['Equipo', 'Formación', 'Titulares', 'Suplentes', 'Primeros nombres']}
                rows={buildApiFootballFormationRows(rawLineups)}
                title="Resumen de formaciones"
              />
              <DataTable
                columns={['Equipo', 'Métrica', 'Valor']}
                rows={buildApiFootballStatisticsRows(rawStatistics)}
                title="Tabla técnica de estadísticas"
              />
              <DataTable
                columns={['Equipo', 'Jugador', 'Edad', 'Posición', 'Número']}
                rows={buildApiFootballPlayersRows(rawPlayers)}
                title="Detalle técnico de jugadores"
              />
            </div>
          </section>

          <section className="space-y-3">
            <SectionTitle
              description="Recortes rápidos del JSON por bloque para validar el shape sin abrir el payload completo."
              title="Vistas raw"
            />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <JsonPreview payload={buildArrayPreview(rawEvents, 12)} title="Eventos crudos" />
              <JsonPreview payload={buildArrayPreview(rawLineups, 4)} title="Lineups" />
              <JsonPreview payload={buildArrayPreview(rawStatistics, 4)} title="Estadísticas por equipo" />
              <JsonPreview payload={buildArrayPreview(rawPlayers, 4)} title="Jugadores por equipo" />
            </div>
          </section>
        </>
      ) : (
        <>
          <section className="space-y-3">
            <SectionTitle
              description="Chequeo operativo de los eventos raw para contrastar el timeline con el modelo canónico."
              title="Timeline y cobertura"
            />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <DataTable
                columns={['Min', 'Equipo', 'Jugador', 'Tipo', 'Outcome']}
                rows={buildStatsBombTimelineRows(rawEvents)}
                title="Timeline raw de StatsBomb"
              />
              <DataTable
                columns={['Min', 'Equipo', 'Jugador', 'Tipo', 'x', 'y']}
                rows={canonicalPreviewRows}
                title="Preview canónico operativo"
              />
            </div>
          </section>

          <section className="space-y-3">
            <SectionTitle
              description="Recortes de los eventos crudos y del modelo canónico para validar shape y consistencia."
              title="Vistas raw"
            />
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <JsonPreview payload={buildArrayPreview(rawEvents, 12)} title="Eventos crudos" />
              <JsonPreview payload={buildArrayPreview(result.canonical_events, 12)} title="Eventos canónicos" />
            </div>
          </section>
        </>
      )}

      <JsonPreview payload={rawPayload ?? null} title="Payload completo del provider" />
    </section>
  )
}

function Badge({ label, tone }: { label: string; tone: 'sky' | 'emerald' | 'amber' | 'violet' }) {
  const palette = {
    sky: 'border-sky-500/30 bg-sky-500/10 text-sky-200',
    emerald: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200',
    amber: 'border-amber-500/30 bg-amber-500/10 text-amber-200',
    violet: 'border-violet-500/30 bg-violet-500/10 text-violet-200',
  }

  return (
    <span className={`rounded-full border px-3 py-1 text-xs font-semibold tracking-wide ${palette[tone]}`}>
      {label}
    </span>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-lg font-semibold text-slate-100">{value}</p>
    </article>
  )
}

function SectionTitle({ title, description }: { title: string; description: string }) {
  return (
    <div>
      <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
      <p className="mt-1 text-sm text-slate-400">{description}</p>
    </div>
  )
}

function DataTable({
  title,
  columns,
  rows,
}: {
  title: string
  columns: string[]
  rows: string[][]
}) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
      <h4 className="text-sm font-semibold text-slate-100">{title}</h4>
      {!rows.length ? (
        <p className="mt-3 text-sm text-slate-400">No hay datos técnicos disponibles para esta vista.</p>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                {columns.map((column) => (
                  <th className="px-3 py-2 font-semibold" key={column}>
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr className="border-b border-slate-900/80 align-top" key={`${title}-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td className="px-3 py-2 text-slate-200" key={`${title}-${rowIndex}-${cellIndex}`}>
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function JsonPreview({ title, payload }: { title: string; payload: unknown }) {
  return (
    <details className="rounded-xl border border-slate-700 bg-slate-950/60 p-4">
      <summary className="cursor-pointer text-sm font-semibold text-slate-200">{title}</summary>
      <pre className="mt-3 max-h-72 overflow-auto rounded bg-slate-950 p-3 text-xs text-sky-100">
        {JSON.stringify(payload, null, 2)}
      </pre>
    </details>
  )
}
