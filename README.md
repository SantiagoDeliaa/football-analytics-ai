---
title: Soccer Analytics AI
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚽ Soccer Analytics AI - Proyecto Final

**Plataforma de analisis tactico para futbol con frontend React, backend FastAPI y core analitico en Python**

---

## Resumen Ejecutivo

Este repositorio concentra el estado integrado del proyecto:

- `front-tip/` provee la experiencia visual moderna en `React + TypeScript + Vite` para `Vertical 1` y `Vertical 2`.
- `api/` expone `FastAPI` como backend HTTP principal para el frontend nuevo.
- `src/` conserva el dominio analitico reutilizable: servicios, adapters, canonical models, metricas, insights y persistencia local.
- `legacy/streamlit/` conserva la UI historica de Streamlit solo como referencia temporal y compatibilidad limitada.

La prioridad actual es sostener una demo sólida con visual moderna, buena UX y compatibilidad con la lógica existente.

En este estado del repo, el despliegue objetivo para publicar la UI moderna quedó alineado con **Hugging Face Docker Spaces**:

- `README.md` raíz configurado con `sdk: docker`;
- `Dockerfile` raíz multi-stage para compilar `front-tip` y levantar FastAPI en `7860`;
- FastAPI sirviendo el build React y resolviendo rutas SPA;
- frontend consumiendo `/api/*` en misma origin en producción.

## Funcionalidades Principales

- `Vertical 1`:
  - tracking de jugadores con IDs persistentes
  - radar 2D y homografía
  - formaciones tácticas
  - métricas colectivas, posesión y scouting
- `Vertical 2`:
  - carga de reportes PDF
  - ingesta desde `StatsBomb Open Data`
  - ingesta desde `API-Football`
  - modelo canónico de eventos
  - métricas propietarias, visualizaciones e insights
  - AI Tactical Coach con contexto estructurado
- `Infraestructura`:
  - backend FastAPI para React
  - persistencia local en SQLite + JSON
  - persistencia remota opcional en Neon PostgreSQL + Cloudflare R2
  - Docker Compose para entorno integrado
  - suite de tests backend/frontend/regresión

## Arquitectura Principal

```text
React frontend (`front-tip/`)
    ->
FastAPI endpoints (`api/`)
    ->
Python services / adapters / canonical models (`src/`)
    ->
Metrics / insights / persistence

Legacy / soporte:
- Streamlit archivado en `legacy/streamlit/`
- UI legacy residual en `src/verticals/*` y `src/utils/ui/*`
- Docs vivas en `docs/`
- Reglas para agentes en `.trae/`
```

## Cómo Levantar el Proyecto

### Opción 1: Docker Compose

Desde la raíz del repo:

```bash
docker compose up --build
```

Servicios expuestos:

- Frontend React: `http://localhost:5173`
- Backend FastAPI: `http://localhost:8000`

Nota importante:

- `VITE_API_BASE_URL` debe resolver a `http://localhost:8000` desde el navegador.
- El hostname `backend` sirve sólo para la red interna de Docker, no para una app Vite ejecutada en el browser del host.

### Opcion 2: Desarrollo local sin Docker

Backend:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Configuración de persistencia por entorno:

```bash
# modo local
PERSISTENCE_BACKEND=local
STORAGE_BACKEND=local
SQLITE_DB_PATH=data/tip_event_data.sqlite
LOCAL_STORAGE_ROOT=data/storage

# modo remoto
PERSISTENCE_BACKEND=postgres
STORAGE_BACKEND=r2
DATABASE_URL=
POSTGRES_HOST=
POSTGRES_PORT=5432
POSTGRES_DATABASE=
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_SSLMODE=require
POSTGRES_CHANNEL_BINDING=require
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_ENDPOINT_URL=
R2_PUBLIC_BASE_URL=
```

Aplicar schema base sobre PostgreSQL/Neon:

```bash
python scripts/apply_postgres_schema.py --database-url "postgresql://..."
```

Notas:

- Si `DATABASE_URL` está vacío, la app intenta construirlo desde `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_SSLMODE` y `POSTGRES_CHANNEL_BINDING`.
- `.env` local está ignorado por Git y `.env.example` queda como referencia segura sin secrets reales.
- Si `STORAGE_BACKEND=r2`, el bucket debe existir y la app requiere `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` y `R2_ENDPOINT_URL`.
- Para ejecutar una prueba real opt-in contra Cloudflare R2: `RUN_LIVE_R2_TESTS=1 python -m pytest tests/test_r2_live_integration.py`.

Frontend:

```bash
cd front-tip
npm install
npm run dev
```

Requisito de Node para `front-tip`:

- `Node >= 20.19.0`
- recomendado: `Node 22.12+`

Legacy Streamlit opcional:

```bash
python -m pip install -r requirements-legacy.txt
streamlit run legacy/streamlit/app.py
```

### Opción 3: Hugging Face Spaces

El repo ya quedó adaptado para publicar la UI moderna del proyecto en Hugging Face usando:

- usar `Docker Space`;
- construir `front-tip` en producción;
- servir el build React desde FastAPI;
- exponer todo en el puerto `7860`.

Validación local realizada sobre esta configuración:

- `npm ci`
- `npm run build`
- `python -m compileall api src`
- `docker build -t sport-analytics-hf-recovery -f Dockerfile .`
- `docker run --rm -p 7860:7860 sport-analytics-hf-recovery`
- verificación de `GET /api/health`
- verificación de `GET /`

La guía paso a paso y los requisitos operativos quedaron documentados en `docs/HUGGINGFACE_DEPLOY.md`.

## Testing

### Backend / Python

Instalación de dependencias:

```bash
python -m pip install -r requirements.txt
```

Comandos oficiales:

```bash
python -m pytest tests/test_api_computer_vision.py
python -m pytest tests/test_api_event_data.py
python -m pytest tests/test_computer_vision_repository.py
python -m pytest tests/test_event_data_repository.py
python -m pytest tests/test_frontend_regression.py -k vertical1
python -m pytest tests/test_frontend_regression.py -k vertical2
python -m pytest
```

Notas:

- Los tests `test_frontend_regression.py` son regresiones Python sobre la capa legacy/compatibilidad, no Vitest del frontend React.
- `requirements.txt` ya no instala `streamlit`; si se necesita la UI legacy hay que usar `requirements-legacy.txt`.
- Si falla un import como `ModuleNotFoundError: fastapi`, el problema es del entorno Python local y no del código del test; volver a instalar `requirements.txt`.
- Si `PERSISTENCE_BACKEND=postgres`, el backend usa Neon/PostgreSQL para metadata y el storage configurado para payloads grandes.
- Si `STORAGE_BACKEND=r2`, los payloads grandes pasan a Cloudflare R2 vía cliente S3-compatible.

### Frontend

Instalación de dependencias:

```bash
cd front-tip
npm install
```

Versión mínima:

- `Node >= 20.19.0`
- recomendado: `Node 22.12+`

Comandos oficiales:

```bash
cd front-tip
npm run test
npm run lint
npm run build
```

Notas:

- Con `Node 18.20.5`, `npm install` puede resolver paquetes, pero `Vitest` falla al arrancar por incompatibilidad real del stack `Vite/Vitest/rolldown`.
- El `typecheck` con `npx tsc -b` puede seguir funcionando incluso cuando Vitest no arranca.
- En Docker ya queda alineado porque el repo usa Node 22 en [Dockerfile](file:///c:/football-analytics-ai/Dockerfile) y [front-tip/Dockerfile](file:///c:/football-analytics-ai/front-tip/Dockerfile).

## Endpoints principales

- `GET /api/v1/event-data/competitions`
- `GET /api/v1/event-data/matches`
- `POST /api/v1/event-data/analyze`
- `POST /api/v1/event-data/pdf`
- `GET /api/v1/event-data/history`
- `GET /api/v1/event-data/history/{provider}/{match_id}`
- `POST /api/v1/computer-vision/analyze`
- `POST /api/v1/computer-vision/jobs`
- `GET /api/v1/computer-vision/jobs/{job_id}`
- `GET /api/v1/computer-vision/history`
- `GET /api/v1/computer-vision/history/{processing_id}`
- `DELETE /api/v1/computer-vision/history/{processing_id}`

## Estructura del Proyecto

```text
football-analytics-ai-recovery/
├── api/                 # FastAPI y contratos HTTP
├── data/                # fixtures y persistencia local de demo
├── docs/                # documentación viva del producto y arquitectura
├── front-tip/           # frontend React + Vite + Tailwind
├── models/              # referencias/modelos pesados versionados selectivamente
├── legacy/              # superficies archivadas y compatibilidad temporal
├── src/                 # dominio analitico y servicios compartidos
├── tests/               # pruebas backend, frontend-compat y soporte legacy
├── .trae/               # contexto para agentes, skills y reglas
├── Dockerfile           # runtime final para Hugging Face Docker Space
├── Dockerfile.api
├── docker-compose.yml
├── app.py               # launcher de compatibilidad hacia legacy/streamlit
├── requirements.txt
├── requirements-legacy.txt
└── README.md
```

## Contexto para agentes y desarrollo

Antes de tocar código, leer:

1. `docs/PRODUCT_CONTEXT.md`
2. `docs/ARCHITECTURE.md`
3. `docs/EVENT_DATA_VERTICAL.md`
4. `docs/LOCAL_PERSISTENCE.md`
5. `docs/AGENT_WORKFLOW.md`
6. `.trae/project_rules.md`

Eso asegura que cualquier agente o dev tenga contexto suficiente sobre:

- visión de producto
- límites de la demo
- arquitectura principal React + FastAPI + src
- pipeline canónico de Vertical 2
- reglas de integración entre frontend nuevo y backend existente

## Persistencia y fixtures locales

Se restauraron fixtures minimos de persistencia local en el workspace para preservar contexto operativo y pruebas manuales:

- `data/tip_event_data.sqlite`
- `data/event_data/raw/statsbomb_open_data/3895302.json`
- `data/event_data/canonical/statsbomb_open_data/3895302.json`
- `data/event_data/metrics/statsbomb_open_data/3895302.json`

Esos archivos quedan como contexto local y `data/` sigue ignorado por Git para no subir persistencia ni payloads al repositorio.

## Persistencia remota

La persistencia remota actual de `Vertical 2` queda soportada por:

- `Neon PostgreSQL` para metadata, historial, datasets, jobs y referencias.
- `Cloudflare R2` para payloads grandes y artefactos pesados.
- `Repository Layer` para desacoplar el dominio de la implementación concreta.
- `Storage Service` para desacoplar el backend del storage físico.

Regla operativa:

- `DB`: metadata, relaciones, estados, versiones, referencias y resúmenes.
- `Object Storage`: raw payloads, canonical events completos, metrics grandes, tracking, reportes y exports.

Convención de object keys:

```text
organizations/{organization_id}/matches/{match_id}/{category}/{provider?}/{subcategory?}/{version}/{file_name}
```

Ejemplos:

```text
organizations/local_demo/matches/3895302/raw/statsbomb_open_data/v1/events.json
organizations/local_demo/matches/3895302/canonical/v1/events.json
organizations/local_demo/matches/3895302/metrics/event_data_metrics/v1/metrics.json
```

Documentación operativa ampliada:

- `docs/REMOTE_PERSISTENCE.md`

Migración segura desde persistencia local:

```bash
# auditoría sin escribir en Neon/R2
python scripts/migrate_event_data_to_remote.py

# migración real
python scripts/migrate_event_data_to_remote.py --execute

# reporte estructurado
python scripts/migrate_event_data_to_remote.py --report-file logs/migration-report.json
```

## Verificación recomendada

```bash
python -m pytest
cd front-tip
npx tsc -b
npm run test -- --run
npm run build
```

Para una validación guiada del estado integrado, ver `docs/INTEGRATION_VALIDATION.md`.

Para validar específicamente el runtime final de Hugging Face:

```bash
docker build -t sport-analytics-hf-recovery -f Dockerfile .
docker run --rm -p 7860:7860 sport-analytics-hf-recovery
```

## Créditos

- Desarrollo: Matías
- Modelo Soccana: [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint)
- Detección de jugadores: Ultralytics YOLO
- Visualización: Plotly, React, Tailwind

## Licencia

Proyecto de código abierto con fines educativos y de demo.
