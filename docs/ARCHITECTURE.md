# Arquitectura Actual

## Estado general
La arquitectura actual es **hibrida**:

- `front-tip/` concentra la experiencia visual moderna en React.
- `api/` expone FastAPI como capa HTTP para el frontend nuevo.
- `src/` conserva el dominio analitico reutilizado desde la aplicacion original.
- `app.py` y `src/verticals/` siguen operativos como legado Streamlit y referencia funcional.

Esto mantiene estable la demo mientras desacopla la UI nueva del stack legacy.

## Entrypoints principales

- `front-tip/src/main.tsx`: entrada del frontend React.
- `api/main.py`: entrada del backend HTTP.
- `app.py`: entrada legacy para Streamlit.

## Responsabilidades por capa

- `front-tip/`
  - navegacion, layout y UX moderna
  - renderizado de Vertical 1 y Vertical 2
  - formularios, estados async y visualizaciones cliente
- `api/`
  - contratos HTTP y serializacion
  - orquestacion de llamadas hacia servicios del dominio
  - compatibilidad con frontend nuevo
- `src/controllers/`
  - pipeline de Computer Vision y metricas de video
- `src/services/`
  - ingestion de providers
  - normalizacion a modelo canonico
  - metricas, insights, PDF ingestion y AI Coach
  - persistencia local SQLite + JSON
- `src/utils/ui/`
  - helpers visuales del legado Streamlit
- `data/`
  - base SQLite y payloads persistidos de demo

## Arquitectura conceptual

```text
React Frontend
├── Home
├── Vertical 1
│   └── FastAPI
│       └── Computer Vision domain
└── Vertical 2
    └── FastAPI
        └── Event Data domain
            ├── Provider ingestion
            ├── Canonical Event Model
            ├── Metrics engine
            ├── Insights engine
            ├── Tactical visualizations
            ├── AI Coach context builder
            └── Local persistence

Legacy Streamlit
└── Usa el mismo dominio en `src/`
```

## Vertical 1

- El frontend nuevo consume `/api/v1/computer-vision/*`.
- El backend puede crear jobs o devolver fallback controlado segun disponibilidad del pipeline real.
- El dominio principal vive en `src/controllers/` y `src/utils/`.

## Vertical 2

- El frontend nuevo consume `/api/v1/event-data/*`.
- Soporta `StatsBomb Open Data`, `API-Football` y flujo PDF.
- Todo provider debe pasar por:
  `Provider -> Ingestion/Adapter -> Canonical Model -> Metrics -> Insights -> UI`

## AI Tactical Coach

- Vive en `src/services/ai_coach/`.
- Consume contexto estructurado y agnostico al provider.
- No debe depender de raw provider data completo por defecto.

## Nota de alcance

- No es una arquitectura enterprise final.
- Es una integracion pragmatica orientada a demo estable.
- La prioridad actual es mantener compatibilidad entre backend existente y frontend renovado sin reescribir el dominio analitico.
