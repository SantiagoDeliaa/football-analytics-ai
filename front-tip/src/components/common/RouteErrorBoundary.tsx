import { isRouteErrorResponse, useRouteError } from 'react-router-dom'
import { ErrorState } from './ErrorState'

export function RouteErrorBoundary() {
  const error = useRouteError()

  if (isRouteErrorResponse(error)) {
    return <ErrorState message={error.data?.message ?? error.statusText ?? 'Ocurrió un error inesperado.'} />
  }

  if (error instanceof Error) {
    return <ErrorState message={error.message} />
  }

  return <ErrorState message="Ocurrió un error inesperado en la navegación." />
}
