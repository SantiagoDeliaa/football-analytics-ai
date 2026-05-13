import createPlotlyComponentModule from 'react-plotly.js/factory'
import Plotly from 'plotly.js/lib/core'
import scatter from 'plotly.js/lib/scatter'
import scatterpolar from 'plotly.js/lib/scatterpolar'
import type { CSSProperties, ComponentType } from 'react'
import type { Config, Data, Layout } from 'plotly.js'

const createPlotlyComponent =
  (
    createPlotlyComponentModule as unknown as {
      default?: (plotly: unknown) => unknown
    } & ((plotly: unknown) => unknown)
  ).default ?? createPlotlyComponentModule
const plotlyCore = Plotly as { register: (modules: unknown[]) => void }
plotlyCore.register([scatter, scatterpolar])
const Plot = createPlotlyComponent(plotlyCore as never) as ComponentType<Record<string, unknown>>

export interface PlotlyChartProps {
  className?: string
  data: Data[]
  layout?: Partial<Layout>
  config?: Partial<Config>
  frames?: unknown[]
  style?: CSSProperties
  useResizeHandler?: boolean
  onError?: (error: unknown) => void
  onInitialized?: (figure: unknown, graphDiv: unknown) => void
  onPurge?: (figure: unknown, graphDiv: unknown) => void
  onUpdate?: (figure: unknown, graphDiv: unknown) => void
  divId?: string
  debug?: boolean
}

export function PlotlyChart({
  className,
  config,
  layout,
  style,
  useResizeHandler,
  ...props
}: PlotlyChartProps) {
  const resolvedConfig =
    config && typeof config === 'object'
      ? {
          displayModeBar: false,
          responsive: true,
          ...config,
        }
      : {
          displayModeBar: false,
          responsive: true,
        }

  const resolvedStyle =
    style && typeof style === 'object'
      ? { width: '100%', height: '100%', ...style }
      : { width: '100%', height: '100%' }

  return (
    <div className={className}>
      <Plot
        config={resolvedConfig}
        data={props.data}
        debug={props.debug}
        divId={props.divId}
        frames={props.frames}
        layout={layout}
        onError={props.onError}
        onInitialized={props.onInitialized}
        onPurge={props.onPurge}
        onUpdate={props.onUpdate}
        style={resolvedStyle}
        useResizeHandler={useResizeHandler ?? true}
      />
    </div>
  )
}
