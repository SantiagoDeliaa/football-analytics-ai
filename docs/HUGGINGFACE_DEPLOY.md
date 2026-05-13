# Deploy En Hugging Face Spaces

## Estado Actual

Este repositorio ya quedó adaptado para desplegarse en Hugging Face Spaces usando **Docker Space** con la UI moderna del proyecto:

- frontend React + Vite en `front-tip/`
- backend FastAPI en `api/`
- lógica analítica y persistencia local en `src/`
- soporte Streamlit legacy mantenido sólo como referencia en `app.py`

La implementación actual usa:

- `README.md` raíz con `sdk: docker`
- `Dockerfile` raíz multi-stage
- build de `front-tip` integrado en la imagen
- FastAPI sirviendo `front-tip/dist`
- frontend consumiendo `/api/*` en misma origin en producción
- puerto público `7860`

## Arquitectura Implementada

```text
Usuario
  -> Hugging Face Space (Docker, puerto 7860)
      -> FastAPI
          -> /api/v1/*
          -> /api/static/computer-vision/*
          -> / + rutas SPA -> index.html
          -> assets compilados desde front-tip/dist
      -> React SPA
          -> consume /api/* en el mismo host
```

## Archivos Clave

- `README.md`: metadata del Space con `sdk: docker`
- `Dockerfile`: imagen final para Hugging Face
- `.dockerignore`: reduce el contexto del build
- `api/main.py`: sirve API, assets y fallback SPA
- `front-tip/src/services/apiClient.ts`: resuelve same-origin en producción

## Cambios Aplicados

## 1. Metadata del Space

La raíz del repositorio quedó configurada así:

```yaml
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
```

## 2. Runtime Docker final

El `Dockerfile` raíz ahora:

1. usa un stage Node para compilar `front-tip`;
2. usa un stage Python para FastAPI;
3. instala `curl`, `ffmpeg` y `git`;
4. instala `requirements.txt`;
5. copia `api/`, `src/`, `app.py` y `front-tip/`;
6. copia `front-tip/dist` desde el builder;
7. crea `outputs/api` y `data` con permisos para el usuario `1000`;
8. levanta `uvicorn api.main:app --host 0.0.0.0 --port 7860`.

## 3. Serving del frontend desde FastAPI

`api/main.py` quedó preparado para:

- mantener `/api/health`;
- mantener `/api/static/computer-vision`;
- devolver `index.html` en `/`;
- servir archivos reales del build cuando existen;
- devolver `index.html` para rutas SPA que no empiecen con `/api/`.

Esto resuelve correctamente rutas como:

- `/vertical1`
- `/vertical2`
- `/vertical2/match/:matchId`

## 4. Base URL del frontend

`front-tip/src/services/apiClient.ts` quedó ajustado así conceptualmente:

- si existe `VITE_API_BASE_URL`, la usa;
- si existe `VITE_STREAMLIT_BACKEND_URL`, la usa;
- en desarrollo usa `http://localhost:8000`;
- en producción usa `''`, o sea misma origin.

Eso evita acoplar el build al dominio final del Space.

## Variables Y Secretos

En Hugging Face Spaces, cargá en **Settings -> Variables and secrets**:

### Recomendadas

- `API_FOOTBALL_KEY`
- `AI_COACH_API_KEY`
- `AI_COACH_MODEL`
- `AI_COACH_BASE_URL`

### Opcionales de frontend

- `VITE_API_BASE_URL`
- `VITE_APP_ENV=production`

En la implementación actual, `VITE_API_BASE_URL` no es obligatoria para producción si querés usar misma origin.

## Persistencia

El proyecto sigue usando:

- SQLite en `data/tip_event_data.sqlite`
- JSON en `data/event_data/...`
- outputs locales en `outputs/...`

En Hugging Face esto debe considerarse **efímero**:

- puede perderse entre reinicios o rebuilds;
- sirve para demo, no como persistencia productiva.

## Limitaciones Reales

## Vertical 2

Es la mejor candidata para una demo pública en Hugging Face porque:

- consume menos recursos;
- trabaja bien con StatsBomb Open Data;
- tiene mejor comportamiento de fallback.

## Vertical 1

Puede exponerse en la UI, pero el procesamiento real de video sigue teniendo riesgos:

- dependencias pesadas;
- uso intensivo de CPU;
- `ffmpeg`;
- posibles artefactos/modelos no completos en todos los entornos.

La recomendación sigue siendo tratar `Vertical 1` como demo con fallback controlado si el pipeline real falla.

## Validación Realizada

Sobre esta implementación se validó localmente:

```bash
cd front-tip
npm ci
npm run build
```

```bash
python -m compileall api src
docker build -t sport-analytics-hf-recovery -f Dockerfile .
docker run --rm -p 7860:7860 sport-analytics-hf-recovery
```

Resultado comprobado:

- `GET /api/health` -> `{"status":"ok"}`
- `GET /` -> `200`

## Pasos Para Publicarlo

## Paso 1. Crear el Space

En Hugging Face:

1. crear un Space nuevo;
2. elegir `Docker` como SDK;
3. conectar o subir este repositorio;
4. cargar variables y secretos;
5. esperar el build inicial.

## Paso 2. Verificar el arranque

Validar en el Space:

- home `/`
- navegación a `Vertical 1`
- navegación a `Vertical 2`
- `GET /api/health`
- carga de rutas profundas como `/vertical2/match/3895302`

## Paso 3. Probar flujo funcional mínimo

Recomendación para demo:

1. probar `Vertical 2` con StatsBomb Open Data;
2. probar carga de PDF;
3. validar que la UI responda correctamente;
4. probar `Vertical 1` sabiendo que puede caer en fallback.

## Checklist Final

Antes de pushear a Hugging Face, dejá confirmado:

- `README.md` con `sdk: docker`
- `Dockerfile` raíz presente
- `.dockerignore` presente
- build local del frontend funcionando
- build local de Docker funcionando
- `/api/health` respondiendo
- `/` respondiendo con la SPA
- variables cargadas en Settings del Space

## Resumen Ejecutivo

Este proyecto **ya quedó preparado** para Hugging Face bajo esta estrategia:

- **Docker Space**
- **FastAPI + build de React en el mismo contenedor**
- **frontend consumiendo `/api/*` en misma origin**

No se recomienda volver a `streamlit` si el objetivo es mostrar la UI moderna, ni pasar a `static` mientras backend y frontend sigan formando una única demo integrada.
