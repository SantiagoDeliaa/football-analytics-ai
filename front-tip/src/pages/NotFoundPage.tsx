import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-8 text-center">
      <h1 className="text-2xl font-bold text-slate-100">Página no encontrada</h1>
      <p className="mt-2 text-slate-300">La ruta que buscás no existe en esta versión del frontend.</p>
      <Link
        className="mt-5 inline-flex rounded-md border border-slate-600 px-4 py-2 text-sm text-slate-100 hover:bg-slate-800"
        to="/"
      >
        Volver al inicio
      </Link>
    </section>
  )
}
