import { Link, NavLink, Outlet } from 'react-router-dom'
import {
  loadHomePage,
  loadVertical1Page,
  loadVertical2Page,
  scheduleModulePrefetch,
} from '../../app/modulePreload'

const navItems = [
  { to: '/', label: 'Inicio', preload: loadHomePage },
  { to: '/vertical1', label: 'Computer Vision', preload: loadVertical1Page },
  { to: '/vertical2', label: 'Data Analytics', preload: loadVertical2Page },
]

export function AppLayout() {
  return (
    <div className="min-h-screen text-slate-100">
      <header className="sticky top-0 z-10 border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-4">
          <Link className="flex flex-col" to="/">
            <span className="text-[11px] font-semibold uppercase tracking-[0.28em] text-emerald-300">
              TIP
            </span>
            <span className="text-sm font-semibold tracking-wide text-slate-100">
              Tactical Intelligence Platform
            </span>
          </Link>
          <nav className="flex items-center gap-2">
            {navItems.map((item) => (
              <NavLink
                className={({ isActive }) =>
                  `rounded-full border px-3 py-1.5 text-sm transition ${
                    isActive
                      ? 'border-emerald-400/40 bg-emerald-500/10 text-emerald-100'
                      : 'border-transparent text-slate-300 hover:border-slate-700 hover:bg-slate-900 hover:text-slate-50'
                  }`
                }
                onFocus={() => scheduleModulePrefetch(item.preload)}
                onMouseEnter={() => scheduleModulePrefetch(item.preload)}
                key={item.to}
                to={item.to}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl px-4 py-6 md:py-8">
        <Outlet />
      </main>
    </div>
  )
}
