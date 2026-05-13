import { Tabs } from '../common/Tabs'
import { useMemo, useState } from 'react'
import { PlotlyChart } from '../common/PlotlyChart'
import {
  buildPdfPitchFigure,
  buildPdfPitchViews,
  buildPdfRadarFigure,
  buildPdfRadarMetrics,
} from '../../utils/pdfTacticalViews'

type PdfTab = 'attack' | 'defense' | 'transitions'

export function PdfTacticalBoard({ payload }: { payload: Record<string, unknown> }) {
  const [activeTab, setActiveTab] = useState<PdfTab>('attack')
  const radarFigure = useMemo(() => buildPdfRadarFigure(payload), [payload])
  const radarMetrics = useMemo(() => buildPdfRadarMetrics(payload), [payload])
  const views = useMemo(() => buildPdfPitchViews(payload), [payload])
  const currentView = views.find((view) => view.key === activeTab) ?? views[0]
  const pitchFigure = useMemo(() => buildPdfPitchFigure(currentView), [currentView])
  const maxZoneValue = Math.max(...currentView.zones.map((zone) => zone.value), 0)

  return (
    <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.05fr_1.35fr]">
      <article className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Radar táctico</h3>
        <p className="mt-1 text-sm text-slate-300">
          Comparativa rápida de cinco dimensiones clave del comportamiento colectivo.
        </p>
        <div className="mt-4 h-[460px] overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60 p-2">
          <PlotlyChart
            className="h-full"
            config={radarFigure.config}
            data={radarFigure.data}
            layout={radarFigure.layout}
          />
        </div>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {radarMetrics.map((metric) => (
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3" key={metric.label}>
              <p className="text-xs uppercase tracking-wide text-slate-400">{metric.label}</p>
              <div className="mt-3 flex items-end justify-between gap-3">
                <p className="text-2xl font-semibold text-slate-50">{metric.value}</p>
                <span className="text-xs text-slate-400">/100</span>
              </div>
              <div className="mt-3 h-2 rounded-full bg-slate-800">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-sky-400"
                  style={{ width: `${metric.value}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </article>

      <article className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <div className="flex flex-col gap-3">
          <div>
            <h3 className="text-lg font-semibold text-slate-100">Cancha principal</h3>
            <p className="mt-1 text-sm text-slate-300">{currentView.subtitle}</p>
          </div>

          <Tabs
            onChange={setActiveTab}
            options={[
              { value: 'attack', label: 'Attack' },
              { value: 'defense', label: 'Defense' },
              { value: 'transitions', label: 'Transitions' },
            ]}
            value={activeTab}
          />
        </div>

        <div className="mt-4 overflow-hidden rounded-2xl border border-emerald-500/20 bg-[#0f1720] p-3">
          <div className="h-[420px] w-full rounded-xl bg-[#14532d]">
            <PlotlyChart
              config={pitchFigure.config}
              data={pitchFigure.data}
              layout={pitchFigure.layout}
            />
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
          {currentView.zones.map((zone) => (
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3" key={zone.label}>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-100">{zone.label}</p>
                  <p className="mt-1 text-xs text-slate-400">{currentView.title}</p>
                </div>
                <span className="text-lg font-semibold text-slate-50">{Math.round(zone.value)}</span>
              </div>
              <div className="mt-3 h-2 rounded-full bg-slate-800">
                <div
                  className="h-2 rounded-full"
                  style={{
                    width: `${maxZoneValue > 0 ? Math.max(10, (zone.value / maxZoneValue) * 100) : 0}%`,
                    backgroundColor: zone.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </article>
    </section>
  )
}
