import { Suspense, lazy } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import {
  loadHomePage,
  loadNotFoundPage,
  loadVertical1Page,
  loadVertical2Page,
} from './modulePreload'
import { LoadingState } from '../components/common/LoadingState'
import { RouteErrorBoundary } from '../components/common/RouteErrorBoundary'
import { AppLayout } from '../components/layout/AppLayout'
import { EventDataProvider } from './EventDataContext'

const HomePage = lazy(async () => {
  const module = await loadHomePage()
  return { default: module.HomePage }
})

const NotFoundPage = lazy(async () => {
  const module = await loadNotFoundPage()
  return { default: module.NotFoundPage }
})

const Vertical1Page = lazy(async () => {
  const module = await loadVertical1Page()
  return { default: module.Vertical1Page }
})

const Vertical2Page = lazy(async () => {
  const module = await loadVertical2Page()
  return { default: module.Vertical2Page }
})

function withRouteLoader(node: React.ReactNode) {
  return <Suspense fallback={<LoadingState label="Cargando módulo..." />}>{node}</Suspense>
}

export const router = createBrowserRouter([
  {
    path: '/',
    errorElement: <RouteErrorBoundary />,
    element: (
      <EventDataProvider>
        <AppLayout />
      </EventDataProvider>
    ),
    children: [
      { index: true, element: withRouteLoader(<HomePage />) },
      { path: 'vertical1', element: withRouteLoader(<Vertical1Page />) },
      { path: 'vertical2', element: withRouteLoader(<Vertical2Page />) },
      { path: 'vertical2/match/:matchId', element: withRouteLoader(<Vertical2Page />) },
      { path: '*', element: withRouteLoader(<NotFoundPage />) },
    ],
  },
])
