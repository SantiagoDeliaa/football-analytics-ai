# Requerimientos para Escalar

## Objetivo de este documento
Separar claramente:
- lo necesario para la demo actual;
- lo necesario para llevar la plataforma a una operación más robusta y escalable.

La premisa es simple: **hoy la prioridad es la migración visual**, pero esta es la base de lo que hará falta para escalar después.

## Estado actual
- Frontend nuevo en React (`front-tip`).
- Backend Python reutilizando la lógica existente.
- Persistencia liviana en `Vertical 2` con SQLite + JSON.
- Jobs de `Vertical 1` actualmente resueltos de forma simple para no frenar la migración visual.

## Qué problema aparece al escalar
Cuando el sistema pase de demo a uso sostenido, aparecen necesidades que hoy no son críticas:
- múltiples usuarios concurrentes;
- jobs largos de video;
- reintentos y recuperación tras reinicios;
- trazabilidad de errores;
- cancelación de procesos;
- serving estable del frontend;
- despliegue repetible entre ambientes.

## Recomendación de arquitectura objetivo

### 1. Frontend productivo
- Servir el build de React desde **Nginx**.
- Usar FastAPI sólo para API/backend.
- Evitar `Vite dev server` en producción.

### 2. Persistencia de negocio y jobs
- Migrar metadata persistente a **PostgreSQL**.
- Mantener archivos grandes fuera de la base:
  - videos
  - outputs procesados
  - payloads grandes JSON
- Guardar en DB:
  - estado de jobs
  - progreso
  - errores
  - referencias a archivos
  - historial consultable

### 3. Background processing real para Vertical 1
- Separar la ejecución pesada del proceso web.
- Usar un **worker** dedicado.
- Incorporar una cola o broker, preferentemente **Redis**.
- Evitar jobs en memoria del proceso principal para escenarios productivos.

### 4. Observabilidad mínima
- Logs estructurados.
- Correlation ID por request/job.
- Registro de errores de pipeline.
- Métricas mínimas de:
  - tiempo de procesamiento
  - fallas
  - porcentaje de jobs completados

## Decisión recomendada sobre base de datos

### Qué usar ahora
- **No es obligatorio migrar todo ya** antes de la demo.
- El esquema actual con SQLite + JSON sirve para demo, validación funcional y trabajo local.

### Qué usar al escalar
- La mejor decisión es **PostgreSQL** para metadata y estado de jobs.

### Por qué PostgreSQL
- mejor concurrencia que SQLite;
- mejor resiliencia para varios usuarios;
- mejor manejo de queries de historial;
- más apto para integrarse con futuros workers y dashboards operativos;
- migración natural desde el patrón repository actual.

### Qué no conviene guardar en PostgreSQL
- blobs pesados de video;
- archivos procesados grandes;
- dumps enormes de eventos si sólo necesitás consultarlos ocasionalmente.

### Estrategia recomendada
- DB para metadata y estado.
- Filesystem local o storage externo para artefactos pesados.

## Modelo mínimo recomendado para `Vertical 1`

### Tabla `cv_jobs`
- `id`
- `job_type`
- `status`
- `progress_pct`
- `progress_stage`
- `source_type`
- `source_path`
- `result_path`
- `error_message`
- `created_at`
- `started_at`
- `finished_at`
- `cancel_requested_at`

### Tabla `cv_job_events`
- historial de transición de estados;
- mensajes de progreso;
- eventos de error;
- eventos de cancelación.

## Modelo recomendado para `Vertical 2`
- Mantener la idea actual de `processed_matches`.
- Migrar metadata a PostgreSQL.
- Mantener JSON o storage externo para payloads grandes.
- Exponer historial consultable por API.

## Progreso real por job
Para un flujo productivo de `Vertical 1`, el sistema debería reportar:
- `queued`
- `preparing-video`
- `loading-models`
- `detecting-players`
- `tracking`
- `homography`
- `metrics`
- `exporting`
- `completed`
- `failed`

Además:
- `progress_pct` debería actualizarse durante el pipeline;
- el frontend debería consultar o suscribirse al estado del job;
- el backend debería persistir ese progreso.

## Cancelación
Para soportar cancelación real:
- el frontend solicita cancelación;
- el backend marca el job como `cancel_requested`;
- el worker chequea esa señal entre etapas seguras;
- el job termina como `cancelled`.

## Orden recomendado de evolución

### Fase 1 — Después de la demo
- estabilizar contrato React + backend;
- mantener arquitectura actual;
- corregir bugs de integración reales.

### Fase 2 — Persistencia robusta
- migrar SQLite a PostgreSQL para metadata;
- agregar tabla de jobs;
- mantener filesystem para outputs.

### Fase 3 — Background jobs productivos
- introducir Redis + worker;
- sacar jobs pesados del proceso web;
- persistir progreso y cancelación.

### Fase 4 — Serving productivo
- servir React con Nginx;
- usar reverse proxy hacia FastAPI;
- definir configuración por ambiente.

## Conclusión ejecutiva
Para la demo:
- conviene priorizar la migración visual y la estabilidad.

Para escalar:
- la mejor decisión técnica es evolucionar hacia:
  - **React build servido por Nginx**
  - **FastAPI como backend**
  - **PostgreSQL para metadata y jobs**
  - **Redis + worker para procesamiento pesado**
  - **filesystem o storage externo para artefactos grandes**

Esto permite crecer sin tirar lo ya construido, porque el proyecto ya tiene una separación razonable entre UI, servicios y capa de persistencia.
