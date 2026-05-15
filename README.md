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

**Plataforma de análisis táctico para fútbol con frontend React, backend FastAPI y legado Streamlit**

---

## Resumen Ejecutivo

Este repositorio concentra el estado integrado del proyecto:

- `front-tip/` provee la experiencia visual moderna en React + Vite para `Vertical 1` y `Vertical 2`.
- `api/` expone endpoints FastAPI para que el frontend nuevo consuma el motor actual sin depender de la UI de Streamlit.
- `src/` conserva el dominio analítico existente, incluyendo Computer Vision, Event Data, AI Tactical Coach y persistencia local.
- `app.py` y `src/streamlit_app.py` siguen disponibles como interfaz legacy y referencia funcional.

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
  - Docker Compose para entorno integrado
  - suite de tests backend/frontend/regresión

## Arquitectura Actual

```text
Home React
├── Vertical 1 (Computer Vision)
│   └── FastAPI -> src/controllers + src/utils
└── Vertical 2 (Event Data)
    └── FastAPI -> src/services
        ├── Provider ingestion
        ├── Canonical Event Model
        ├── Metrics + insights
        ├── Tactical visualizations
        ├── AI Coach context builder
        └── Local persistence

Legacy / soporte:
- Streamlit (`app.py`, `src/verticals/*`)
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

### Opción 2: Desarrollo local sin Docker

Backend:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Frontend:

```bash
cd front-tip
npm install
npm run dev
```

Requisito de Node para `front-tip`:

- `Node >= 20.19.0`
- recomendado: `Node 22.12+`

Legacy Streamlit:

```bash
streamlit run app.py
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

### Backend

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
- Si falla un import como `ModuleNotFoundError: fastapi`, el problema es del entorno Python local y no del código del test; volver a instalar `requirements.txt`.

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
npm run test -- --run
npm run test -- --run src/pages/Vertical1Page.test.tsx
npm run test -- --run src/pages/Vertical2Page.test.tsx
npx tsc -b
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
├── src/                 # dominio analítico legado y servicios compartidos
├── tests/               # pruebas backend, regresión y soporte Streamlit
├── .trae/               # contexto para agentes, skills y reglas
├── Dockerfile           # runtime final para Hugging Face Docker Space
├── Dockerfile.api
├── docker-compose.yml
├── app.py
├── requirements.txt
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
- arquitectura híbrida actual
- pipeline canónico de Vertical 2
- reglas de integración entre frontend nuevo y backend existente

## Persistencia y fixtures locales

Se restauraron fixtures minimos de persistencia local en el workspace para preservar contexto operativo y pruebas manuales:

- `data/tip_event_data.sqlite`
- `data/event_data/raw/statsbomb_open_data/3895302.json`
- `data/event_data/canonical/statsbomb_open_data/3895302.json`
- `data/event_data/metrics/statsbomb_open_data/3895302.json`

Esos archivos quedan como contexto local y `data/` sigue ignorado por Git para no subir persistencia ni payloads al repositorio.

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
