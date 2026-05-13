const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() ||
  import.meta.env.VITE_STREAMLIT_BACKEND_URL?.trim() ||
  'http://localhost:8000'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    const rawMessage = await response.text()

    try {
      const parsed = JSON.parse(rawMessage) as { detail?: string }
      throw new ApiError(parsed.detail || rawMessage || 'Error de red', response.status)
    } catch {
      throw new ApiError(rawMessage || 'Error de red', response.status)
    }
  }

  return (await response.json()) as T
}

export { API_BASE_URL }
