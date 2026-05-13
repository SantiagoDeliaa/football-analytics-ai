import { useState, type FormEvent } from 'react'
import { z } from 'zod'

const pdfFileSchema = z.custom<File>(
  (file) => file instanceof File && file.type === 'application/pdf',
  'Debés seleccionar un archivo PDF.',
)

interface PdfUploadFormProps {
  loading: boolean
  onSubmit: (file: File) => void
}

export function PdfUploadForm({ loading, onSubmit }: PdfUploadFormProps) {
  const [error, setError] = useState<string>()

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const input = event.currentTarget.elements.namedItem('report') as HTMLInputElement | null
    const file = input?.files?.[0]
    const parsed = pdfFileSchema.safeParse(file)

    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? 'Archivo inválido.')
      return
    }

    setError(undefined)
    onSubmit(parsed.data)
  }

  return (
    <form
      className="rounded-xl border border-slate-700 bg-slate-900/70 p-4"
      onSubmit={handleSubmit}
      noValidate
    >
      <h3 className="text-base font-semibold text-slate-100">Subir PDF</h3>
      <p className="mt-1 text-sm text-slate-300">
        Cargá un reporte Wyscout en PDF para obtener KPIs e insights tácticos.
      </p>

      {error ? (
        <p className="mt-3 rounded-lg border border-rose-700/50 bg-rose-900/30 p-2 text-sm text-rose-100">
          {error}
        </p>
      ) : null}

      <label className="mt-4 block text-sm text-slate-200">
        Archivo PDF
        <input
          accept=".pdf,application/pdf"
          className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
          name="report"
          type="file"
        />
      </label>

      <button
        className="mt-4 rounded-md border border-sky-500 bg-sky-700 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-600 disabled:cursor-not-allowed disabled:opacity-50"
        disabled={loading}
        type="submit"
      >
        {loading ? 'Procesando...' : 'Procesar PDF'}
      </button>
    </form>
  )
}
