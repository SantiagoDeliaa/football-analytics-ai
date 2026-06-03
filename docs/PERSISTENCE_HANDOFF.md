# Handoff Técnico de Persistencia

## Objetivo

Dejar documentado el estado final de la primera fase de persistencia remota para `Vertical 2`.

## Alcance implementado

- schema base en `Neon PostgreSQL`
- storage real en `Cloudflare R2`
- selección de backend por entorno
- `Repository Layer` compatible con flujo actual
- `Storage Service` local y R2
- historial remoto
- carga remota de payloads
- borrado remoto de metadata y objetos
- migrador seguro `SQLite + JSON -> Neon + R2`

## Decisiones técnicas consolidadas

- `matches` representa el partido canónico
- `match_provider_links` representa equivalencias por provider
- `organization_matches` habilita multi-tenant sin duplicar partidos
- `storage_objects` separa ubicación física del objeto
- `match_assets` expresa semántica funcional del asset
- `event_datasets`, `metric_sets` y `quality_summaries` trazan outputs analíticos
- payloads grandes van a `R2`
- metadata y resúmenes van a `PostgreSQL`

## Estado validado

Validado en runtime:

- guardado en `Neon`
- guardado en `R2`
- `GET /api/v1/event-data/history`
- `GET /api/v1/event-data/history/{provider}/{match_id}`
- `DELETE /api/v1/event-data/history/{provider}/{match_id}`
- migración real desde fixture local controlado

## Archivos clave

- `db/migrations/001_tip_postgres_foundation.sql`
- `src/services/storage/settings.py`
- `src/services/storage/storage_service.py`
- `src/services/storage/postgres_event_data_repository.py`
- `src/services/storage/event_data_repository.py`
- `scripts/apply_postgres_schema.py`
- `scripts/migrate_event_data_to_remote.py`
- `docs/REMOTE_PERSISTENCE.md`

## Operación

### Backend local

```bash
PERSISTENCE_BACKEND=local
STORAGE_BACKEND=local
```

### Backend remoto

```bash
PERSISTENCE_BACKEND=postgres
STORAGE_BACKEND=r2
```

## Migración

Auditoría sin cambios:

```bash
python scripts/migrate_event_data_to_remote.py
```

Migración real:

```bash
python scripts/migrate_event_data_to_remote.py --execute
```

Reporte JSON:

```bash
python scripts/migrate_event_data_to_remote.py --report-file logs/migration-report.json
python scripts/migrate_event_data_to_remote.py --execute --report-file logs/migration-report.json
```

## Riesgos abiertos

- falta integrar migración histórica real desde una fuente local poblada
- falta endurecer rollback compensatorio si falla escritura parcial entre DB y storage
- falta rotar credenciales expuestas durante setup manual
- Computer Vision todavía no usa esta infraestructura remota

## Próximos pasos recomendados

1. migrar un subconjunto histórico real
2. agregar reportes de migración a CI/manual QA
3. definir estrategia de rotación de secretos
4. evaluar fase 2 para `processing_jobs`, `quality_summaries` y AI Coach persistente
