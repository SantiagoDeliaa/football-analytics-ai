import { useCallback, useState } from 'react'

type AsyncState<T> = {
  data?: T
  loading: boolean
  error?: string
}

export function useAsync<T>() {
  const [state, setState] = useState<AsyncState<T>>({ loading: false })

  const run = useCallback(async (task: () => Promise<T>) => {
    setState({ loading: true, error: undefined })
    try {
      const data = await task()
      setState({ loading: false, data })
      return data
    } catch (error) {
      setState({
        loading: false,
        error: error instanceof Error ? error.message : 'Error inesperado.',
      })
      return undefined
    }
  }, [])

  return { ...state, run }
}
