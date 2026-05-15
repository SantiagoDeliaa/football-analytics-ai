import '@testing-library/jest-dom'
import { vi } from 'vitest'

Object.defineProperty(window.HTMLElement.prototype, 'scrollIntoView', {
  value: vi.fn(),
  writable: true,
})

vi.mock('react-chartjs-2', () => ({
  Bar: () => null,
  Doughnut: () => null,
  Line: () => null,
  Radar: () => null,
  Scatter: () => null,
}))

vi.mock('react-plotly.js/factory', () => ({
  default: () => () => null,
}))

vi.mock('plotly.js/lib/core', () => ({
  default: {},
}))

vi.mock('plotly.js/lib/scatter', () => ({
  default: {},
}))

vi.mock('plotly.js/lib/scatterpolar', () => ({
  default: {},
}))
