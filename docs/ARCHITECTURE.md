# Arquitectura Actual

## Decisión principal

La arquitectura oficial del producto es:

- `front-tip/`: frontend principal en `React + TypeScript + Vite`.
- `api/`: backend principal en `FastAPI + Uvicorn`.
- `src/`: core analitico reutilizable en Python.

`Streamlit` ya no participa del flujo principal. Queda archivado en `legacy/streamlit/` y como capa de compatibilidad temporal en algunos modulos de `src/verticals/*` y `src/utils/ui/*`.

## Flujo obligatorio

```text
React frontend
    ->
FastAPI endpoints
    ->
Python services / adapters / canonical models
    ->
Metrics / insights / persistence
```

## Reglas de arquitectura

- Toda nueva UI debe implementarse en `front-tip/`.
- React debe consumir `FastAPI`, no servicios Python directos ni providers externos.
- FastAPI debe orquestar `src/services/`, `src/controllers/` y repositories.
- Los providers nunca deben conectarse directo al frontend.
- Los modelos canonicos son el contrato interno entre ingestion, analitica y presentacion.
- La logica de negocio no debe quedar atrapada en componentes React ni en UI legacy de Streamlit.

## Responsabilidades por capa

- `front-tip/`
  - routing, layout, UX, estados async, tabs y visualizaciones cliente
  - integracion con `api/*`
- `api/`
  - contratos HTTP
  - serializacion y validacion de requests/responses
  - orquestacion de servicios del dominio
- `src/controllers/`
  - pipeline de Computer Vision y analitica asociada
- `src/services/`
  - ingestion de providers
  - adapters y normalizacion a modelos canonicos
  - metricas, insights, PDF ingestion, AI Coach y persistencia
  - presentation helpers reutilizables para tablas, labels y formateo
- `src/utils/ui/`
  - helpers visuales de Streamlit legacy que siguen solo por compatibilidad
- `legacy/streamlit/`
  - launcher y referencia temporal de la UI historica

## Entrypoints

- `front-tip/src/main.tsx`: entrada del frontend React
- `api/main.py`: entrada del backend FastAPI
- `legacy/streamlit/app.py`: entrada opcional del legacy Streamlit
- `app.py`: launcher de compatibilidad que delega a `legacy/streamlit/app.py`

## Vertical 1

- El frontend nuevo consume `/api/v1/computer-vision/*`.
- El dominio principal vive en `src/controllers/` y `src/utils/`.
- La persistencia moderna usa `SQLite + JSON sidecar`.
- No agregar nueva UI de Computer Vision en Streamlit salvo instruccion explicita.

## Vertical 2

- El frontend nuevo consume `/api/v1/event-data/*`.
- Soporta `StatsBomb Open Data`, `API-Football` y flujo PDF.
- Todo provider debe seguir:
  `Provider -> Ingestion/Adapter -> Canonical Model -> Metrics -> Insights -> UI`
- `StatsBomb` sigue siendo el provider tactico para mapas y visualizaciones espaciales.
- `Sportmonks` se usa para contexto, timeline, lineups, stats y expected metrics.
- `Sportmonks` no debe habilitar mapas de cancha ni metricas espaciales si no hay coordenadas confirmadas.

## AI Tactical Coach

- Vive en `src/services/ai_coach/`.
- Debe consumir contexto estructurado y agnostico al provider.
- No debe depender de raw provider data completo por defecto.

## Legacy Streamlit

- No es superficie principal de producto.
- No implementar nuevas features en Streamlit.
- Mantener solo lo minimo para compatibilidad temporal y referencia funcional.
- El retiro definitivo debe ocurrir cuando React + FastAPI cubran todos los flujos necesarios.
