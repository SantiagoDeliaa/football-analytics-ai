type AsyncModuleLoader = () => Promise<unknown>

export function loadHomePage() {
  return import('../pages/HomePage')
}

export function loadNotFoundPage() {
  return import('../pages/NotFoundPage')
}

export function loadVertical1Page() {
  return import('../pages/Vertical1Page')
}

export function loadVertical2Page() {
  return import('../pages/Vertical2Page')
}

export function loadTimelineCharts() {
  return import('../components/vertical1/TimelineCharts')
}

export function loadMetricsSection() {
  return import('../components/vertical2/MetricsSection')
}

export function loadPlayerSpotlight() {
  return import('../components/vertical2/PlayerSpotlight')
}

export function loadVisualizationGrid() {
  return import('../components/vertical2/VisualizationGrid')
}

export function loadPdfTacticalBoard() {
  return import('../components/vertical2/PdfTacticalBoard')
}

export function scheduleModulePrefetch(loader: AsyncModuleLoader) {
  const run = () => {
    void loader().catch(() => undefined)
  }

  if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
    window.requestIdleCallback(() => {
      run()
    })
    return
  }

  globalThis.setTimeout(run, 0)
}
