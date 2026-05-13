declare module 'react-plotly.js/factory' {
  import type { ComponentType } from 'react'

  export default function createPlotlyComponent(plotly: unknown): ComponentType<Record<string, unknown>>
}

declare module 'plotly.js/lib/core' {
  const Plotly: unknown
  export default Plotly
}

declare module 'plotly.js/lib/scatter' {
  const scatter: unknown
  export default scatter
}

declare module 'plotly.js/lib/scatterpolar' {
  const scatterpolar: unknown
  export default scatterpolar
}
