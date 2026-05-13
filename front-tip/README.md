# Frontend TIP (React + Tailwind)

Frontend moderno para `Tactical Intelligence Platform` migrado desde Streamlit, con foco en:

- UI responsive y look & feel oscuro/premium del proyecto original.
- Vertical 1 y Vertical 2 navegables desde una home unificada.
- Vertical 2 funcional (`Subir PDF` y `Datos por API`).
- Componentes reutilizables, rutas dinámicas y manejo robusto de errores/loading.
- Historial local para reabrir partidos procesados en modo demo.
- Tests unitarios para piezas críticas.

## Stack

- React 19 + TypeScript
- Vite 8
- Tailwind CSS 4
- React Router
- Vitest + Testing Library
- Zod (validaciones)

## Estructura

```text
src/
  app/                 # router y context global
  components/
    common/            # UI base reutilizable
    layout/            # layout principal
    vertical1/         # componentes dominio Vertical 1
    vertical2/         # componentes dominio Vertical 2
  hooks/               # hooks reutilizables
  pages/               # Home, Vertical1, Vertical2, NotFound
  services/            # cliente API + servicios de integración
  types/               # contratos TS
  utils/               # selectores y helpers puros
  test/                # setup de pruebas
```

## Variables de entorno

Crear `.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

Si no está definida, se usa `http://localhost:8000` por defecto. `VITE_STREAMLIT_BACKEND_URL` se mantiene sólo por compatibilidad temporal.

## Contrato de integración backend

El frontend espera endpoints HTTP (pueden implementarse en FastAPI/Nest o un gateway delante de Streamlit):

- `POST /api/v1/computer-vision/analyze` (compatibilidad)
- `POST /api/v1/computer-vision/jobs` (multipart/form-data)
- `GET /api/v1/computer-vision/jobs/{job_id}`
- `GET /api/v1/event-data/competitions`
- `GET /api/v1/event-data/matches?competition_id=<id>&season_id=<id>`
- `POST /api/v1/event-data/analyze`
- `POST /api/v1/event-data/pdf` (multipart/form-data)
- `GET /api/v1/event-data/history`
- `GET /api/v1/event-data/history/{provider}/{match_id}`

Si fallan competiciones/partidos o el análisis de computer vision, el frontend usa fallbacks locales (modo demo) para no romper UX.

## Scripts

```bash
npm install
npm run dev
npm run build
npm run lint
npm run test
```

## Docker Compose

Desde la raíz del workspace:

```bash
docker compose up --build
```

Nota importante:

- cuando el frontend corre en el navegador del host, `VITE_API_BASE_URL` debe apuntar a `http://localhost:8000`
- no conviene usar `http://backend:8000` en variables de Vite, porque ese hostname existe dentro de la red Docker, pero no en el navegador del usuario

## Decisiones de diseño

- Se preserva el idioma español en toda la UI.
- `Vertical 1` crea jobs de backend y hace polling hasta obtener el resultado; si el pipeline real falla, el backend devuelve una salida mock controlada para no romper UX.
- Se mantiene el patrón de Vertical 2: `Provider -> Canonical Model -> Metrics -> Insights -> UI`.
- La orquestación de `Vertical2Page` se apoya en helpers/componentes de dominio para reducir acoplamiento.
- El historial de partidos procesados se persiste tanto en `localStorage` como en el backend para reutilización entre sesiones.
