import type { MatchCenterDataQuality } from '../../../types/matchCenter'
import { getDataQualityLabel } from '../../../utils/matchCenter'

interface DataQualityBannerProps {
  dataQuality?: MatchCenterDataQuality | null
}

export function DataQualityBanner({ dataQuality }: DataQualityBannerProps) {
  return (
    <section className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
      <p className="text-sm font-semibold text-amber-100">
        Cobertura de datos: {getDataQualityLabel(dataQuality?.level)}
      </p>
      <p className="mt-2 text-sm text-amber-50">
        {dataQuality?.message ||
          'No hay información detallada de cobertura para este partido.'}
      </p>
    </section>
  )
}
