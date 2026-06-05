# Análisis Consolidado de Base de Datos y Persistencia

## Objetivo

Este documento actualiza y consolida el feedback del equipo y con el estado real de implementación alcanzado en el repo.

Su propósito es dejar claro:

- la dirección de arquitectura aprobada;
- el modelo de datos objetivo;
- qué partes ya están implementadas;
- qué partes quedaron preparadas en schema pero todavía no conectadas al runtime;
- qué sigue pendiente para fases posteriores.

## Contexto

TIP evoluciona desde un MVP local hacia una arquitectura SaaS progresiva para múltiples organizaciones, múltiples fuentes de datos y persistencia remota compartida.

Las dos verticales principales siguen siendo:

- `Vertical 1`: Computer Vision / Tracking Intelligence
- `Vertical 2`: Event Data / Football Intelligence Engine

La arquitectura de producto se mantiene así:

```text
Provider / Video / Dataset
    ->
Adapter / Ingestion Layer
    ->
Canonical Models
    ->
Metrics Engine
    ->
Insights / AI Coach
    ->
Visualizations / Reports
    ->
Persistence
```

La regla estratégica sigue siendo:

- TIP no debe depender de un único provider.
- TIP no debe comportarse como un data vendor.
- TIP debe transformar datos externos y outputs procesados en inteligencia táctica accionable.

## Decisión Técnica Consolidada

La decisión aprobada para la siguiente fase quedó consolidada como:

- `Neon PostgreSQL` para metadata, relaciones, historial, trazabilidad y resúmenes.
- `Cloudflare R2` para payloads grandes y artefactos pesados.
- `Repository Layer` para desacoplar a la app de la base concreta.
- `Storage Service` para desacoplar a la app del backend de archivos.
- `SQLite + LocalStorage` como fallback de desarrollo y compatibilidad.

Regla central:

- `DB = metadata, relaciones, estado, permisos, referencias, versiones, métricas resumidas y trazabilidad`
- `Object Storage = raw payloads, canonical events completos, tracking, frame metrics, reportes, exports y videos`

## Principios de Diseño Aprobados

### Partido canónico separado del provider

`matches` representa el partido canónico interno de TIP.

`match_provider_links` representa equivalencias por provider:

- `provider`
- `provider_match_id`
- metadata resumida del provider
- versionado del adapter

Esto evita contaminar la entidad canónica con IDs externos.

### Semántica del asset separada del objeto físico

`match_assets` representa el significado funcional del artefacto:

- `raw_provider_events`
- `canonical_events`
- `metrics_summary`
- `annotated_video`
- `quality_summary`
- `report_pdf`

`storage_objects` representa la ubicación física:

- `storage_provider`
- `bucket`
- `object_key`
- `mime_type`
- `checksum`
- `size_bytes`
- `asset_version`

### Resumen consultable separado del payload pesado

Las tablas analíticas pueden guardar:

- `summary_json` en `JSONB` para consultas rápidas y payloads cortos
- `storage_object_id` para enlazar el contenido completo en `R2`

Esto aplica sobre todo a:

- `metric_sets`
- `quality_summaries`
- `event_datasets`
- contextos grandes del AI Coach

### Trazabilidad operativa

`processing_jobs` formaliza la trazabilidad por pipeline:

- qué corrió
- quién lo pidió
- con qué config
- qué versión usó
- qué outputs produjo
- en qué estado terminó

### Multi-tenant desde el inicio

El modelo incorpora tenancy desde la base:

- `organizations`
- `organization_users`
- `organization_matches`
- `organization_clubs`
- `organization_settings`

El tenant bootstrap actual para la primera fase es:

- `local_demo`

## Modelo Objetivo Aprobado

El modelo base aprobado con el equipo es:

- `organizations`
- `users`
- `organization_users`
- `clubs`
- `organization_clubs`
- `teams`
- `team_seasons`
- `competitions`
- `seasons`
- `matches`
- `organization_matches`
- `match_participants`
- `match_provider_links`
- `provider_snapshots`
- `storage_objects`
- `match_assets`
- `processing_jobs`
- `event_datasets`
- `metric_sets`
- `quality_summaries`
- `ai_coach_sessions`
- `ai_coach_messages`
- `organization_settings`

## Estado Real de Implementación

## Resumen ejecutivo

El estado actual no es solo teórico.

Ya quedó implementado:

- schema base en `Neon`
- storage real en `R2`
- repository remoto funcional
- storage service local y remoto
- historial remoto operativo para `Vertical 2`
- carga remota de payloads
- migración segura desde `SQLite + JSON`

Pero no todas las tablas del modelo final están conectadas todavía al flujo funcional actual.

La mejor forma de describir el estado es:

- `modelo aprobado`: sí
- `schema implementado`: sí
- `runtime fase 1`: sí
- `modelo completo explotado en todas las entidades`: todavía no

## Matriz de estado

### Implementado en schema y usado en runtime

- `organizations`
- `matches`
- `organization_matches`
- `match_provider_links`
- `storage_objects`
- `match_assets`
- `event_datasets`
- `metric_sets`
- `organization_settings` como base de modelo
- `Repository Layer`
- `Storage Service`

### Implementado en schema, preparado pero todavía no conectado plenamente al runtime

- `users`
- `organization_users`
- `clubs`
- `organization_clubs`
- `teams`
- `team_seasons`
- `competitions`
- `seasons`
- `match_participants`
- `provider_snapshots`
- `processing_jobs`
- `quality_summaries`
- `ai_coach_sessions`
- `ai_coach_messages`

### Implementado parcialmente en runtime

- `processing_jobs`
  - modelado correctamente en schema
  - todavía no se crean jobs reales por cada persistencia de Vertical 2
- `produced_by_job_id`
  - existe en schema
  - todavía no se puebla porque falta integración operativa de `processing_jobs`
- `match_participants`
  - existe en schema
  - aún no se puebla con `team_id` canónicos desde providers
- `provider_snapshots`
  - existe
  - aún no se usa como snapshot operativo persistido
- `quality_summaries`
  - existe
  - todavía no está integrada en el flujo actual
- `AI Coach persistence`
  - schema listo
  - integración funcional pendiente

## Correspondencia Con El Feedback Del Equipo

### Comentarios ya reflejados

- `organizations` con `slug` único y `name` único: sí
- `users.email` único: sí
- `organization_users (organization_id, user_id)` único: sí
- `organization_clubs`: sí
- `team_seasons`, `competitions`, `seasons`: sí
- `matches` canónico: sí
- `match_participants`: sí en schema
- `match_provider_links`: sí
- `provider_snapshots`: sí en schema
- `storage_objects` separado de `match_assets`: sí
- `processing_jobs` con `config_json`, `organization_id`, `requested_by`: sí
- `event_datasets` con `provider_link_id`, `storage_object_id`, `produced_by_job_id`, `is_current`: sí en schema
- `metric_sets` con `summary_json` y `storage_object_id`: sí
- `quality_summaries` con `summary_json`, `storage_object_id`, `produced_by_job_id`: sí en schema
- `organization_settings` en `JSONB`: sí
- implementación incremental por fases: sí

### Comentarios aún pendientes de cierre funcional

- poblar entidades canónicas de fútbol (`clubs`, `teams`, `competitions`, `seasons`) desde fuentes reales
- registrar `processing_jobs` reales por corrida
- persistir `provider_snapshots`
- usar `quality_summaries` en pipelines activos
- persistir sesiones y mensajes de `AI Coach`

## Qué Quedó Validado

Validado en runtime:

- schema aplicado en `Neon`
- uploads y lecturas reales en `Cloudflare R2`
- `POST /api/v1/event-data/analyze`
- `GET /api/v1/event-data/history`
- `GET /api/v1/event-data/history/{provider}/{match_id}`
- `DELETE /api/v1/event-data/history/{provider}/{match_id}`
- migración real desde fixture local controlado con `SQLite + JSON -> Neon + R2`

## Artefactos Implementados

Referencias principales:

- [001_tip_postgres_foundation.sql](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/db/migrations/001_tip_postgres_foundation.sql)
- [settings.py](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/src/services/storage/settings.py)
- [storage_service.py](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/src/services/storage/storage_service.py)
- [postgres_event_data_repository.py](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/src/services/storage/postgres_event_data_repository.py)
- [event_data_repository.py](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/src/services/storage/event_data_repository.py)
- [apply_postgres_schema.py](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/scripts/apply_postgres_schema.py)
- [migrate_event_data_to_remote.py](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/scripts/migrate_event_data_to_remote.py)
- [REMOTE_PERSISTENCE.md](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/docs/REMOTE_PERSISTENCE.md)
- [PERSISTENCE_HANDOFF.md](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/docs/PERSISTENCE_HANDOFF.md)

## Conclusión

La respuesta a la pregunta original es:

- sí, el trabajo realizado contempla la estrategia del documento inicial;
- sí, el feedback del equipo quedó incorporado en el modelo y en el schema;
- sí, la primera fase operativa ya quedó funcionando con `Neon + R2`;
- no, todavía no está conectada toda la arquitectura final en runtime.

La situación correcta para comunicar internamente es:

- `dirección aprobada`: implementada
- `base técnica`: implementada
- `fase 1 funcional`: validada
- `fases avanzadas`: pendientes

## Próximas fases recomendadas

1. poblar catálogos canónicos de fútbol en runtime
2. integrar `processing_jobs` reales con `produced_by_job_id`
3. persistir `provider_snapshots`
4. conectar `quality_summaries`
5. conectar persistencia real de `AI Coach`
6. migrar un subconjunto histórico real cuando exista fuente local poblada
