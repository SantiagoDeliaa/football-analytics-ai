# Persistencia Remota

## Objetivo

Formalizar la persistencia productiva incremental de `Vertical 2` usando:

- `Neon PostgreSQL` para metadata, relaciones, historial y trazabilidad.
- `Cloudflare R2` para payloads grandes y artefactos pesados.

La transición mantiene compatibilidad con el modo local actual:

- `PERSISTENCE_BACKEND=local`
- `STORAGE_BACKEND=local`

y habilita el modo remoto:

- `PERSISTENCE_BACKEND=postgres`
- `STORAGE_BACKEND=r2`

## Documentos Relacionados

- [DATABASE_ANALYSIS.md](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/docs/DATABASE_ANALYSIS.md): estrategia consolidada, modelo objetivo, feedback del equipo y estado real de implementación.
- [PERSISTENCE_HANDOFF.md](file:///c:/Users/matu_/OneDrive/Escritorio/sport-analytics/football-analytics-ai-recovery/docs/PERSISTENCE_HANDOFF.md): handoff corto para onboarding técnico y operación.

## Regla central

- `DB = metadata, relaciones, estado, permisos, índices, versiones y resúmenes`.
- `Object Storage = archivos grandes, JSON completos, videos, reportes y outputs`.

No guardar en PostgreSQL:

- videos
- tracking frame-by-frame
- raw provider payloads completos
- canonical events completos si son grandes
- métricas densas por frame
- PDFs y exports pesados

Sí guardar en PostgreSQL:

- organizaciones
- usuarios y memberships
- partidos canónicos
- equivalencias multi-provider
- datasets y metric sets
- quality summaries resumidos
- object keys
- checksums
- tamaños
- versiones
- jobs y estados de procesamiento

## Componentes

### PostgreSQL / Neon

Schema base actual:

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

### Cloudflare R2

Se usa como backend S3-compatible para:

- `raw_provider_data`
- `canonical_events`
- `metrics` grandes
- `tracking_json`
- `frame_metrics`
- `quality_summaries` completos
- `ai_context`
- `reports`
- videos y outputs renderizados

## Configuración por entorno

Variables mínimas para modo remoto:

```bash
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

Notas:

- Si `DATABASE_URL` está vacío, la app lo arma desde las variables `POSTGRES_*`.
- `R2_PUBLIC_BASE_URL` es opcional.
- `.env` local no debe versionarse en Git.

## Aplicación del schema

Crear el schema base en Neon:

```bash
python scripts/apply_postgres_schema.py --database-url "postgresql://..."
```

Si el entorno ya tiene `.env` remoto completo:

```bash
python scripts/apply_postgres_schema.py
```

## Convención de object keys

Patrón general:

```text
organizations/{organization_id}/matches/{match_id}/{category}/{provider?}/{subcategory?}/{version}/{file_name}
```

Ejemplos:

```text
organizations/local_demo/matches/3895302/raw/statsbomb_open_data/v1/events.json
organizations/local_demo/matches/3895302/canonical/v1/events.json
organizations/local_demo/matches/3895302/metrics/event_data_metrics/v1/metrics.json
organizations/local_demo/matches/3895302/quality/event_data/v1/summary.json
organizations/local_demo/matches/3895302/ai_context/v1/context.json
```

## Flujo operativo actual

```text
FastAPI / Vertical 2
    ->
Repository Layer
    ->
PostgresEventDataRepository
    ->
Neon PostgreSQL

FastAPI / Vertical 2
    ->
Storage Service
    ->
R2StorageService
    ->
Cloudflare R2
```

## Validación recomendada

### Repository y settings

```bash
python -m pytest tests/test_storage_settings.py
python -m pytest tests/test_event_data_repository_initialization.py
python -m pytest tests/test_postgres_event_data_repository.py
python -m pytest tests/test_event_data_repository_dispatch.py
```

### Storage local y R2

```bash
python -m pytest tests/test_storage_service.py
python -m pytest tests/test_r2_storage_service.py
```

### R2 real opt-in

```bash
RUN_LIVE_R2_TESTS=1 python -m pytest tests/test_r2_live_integration.py
```

### API / historial remoto

```bash
python -m pytest tests/test_api_main.py
python -m pytest tests/test_api_event_data.py -k "history"
```

## Troubleshooting

### Falta `DATABASE_URL`

Verificar:

- `PERSISTENCE_BACKEND=postgres`
- `DATABASE_URL` o variables `POSTGRES_*`

### Falta configuración de R2

Verificar:

- `STORAGE_BACKEND=r2`
- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET_NAME`
- `R2_ENDPOINT_URL`

### El bucket existe pero falla upload

Verificar:

- permisos de la access key
- bucket correcto
- endpoint correcto
- rotación de credenciales si fueron expuestas

### La API lista historial pero no carga payloads

Verificar:

- registros en `storage_objects`
- `object_key` correcto
- existencia real del objeto en R2
- `storage_backend` alineado con el entorno

## Seguridad

- No versionar `.env`.
- Usar access keys con permisos mínimos sobre el bucket.
- Rotar credenciales si fueron compartidas en texto claro.
- Evitar usar `R2_PUBLIC_BASE_URL` salvo necesidad real de exposición pública.

## Estado actual

La base remota ya quedó validada en runtime con:

- guardado en `Neon`
- guardado en `R2`
- listado de historial remoto
- carga remota por `provider + match_id`
- borrado remoto de metadata y artefactos

## Siguiente paso recomendado

- crear script de migración `SQLite + JSON -> Neon + R2`
- validar subconjunto histórico
- documentar rollback y estrategia de corte

## Migración segura desde SQLite

El repo incluye un migrador incremental:

```bash
python scripts/migrate_event_data_to_remote.py
```

Comportamiento por defecto:

- corre en `dry-run`
- no borra nada local
- no sube archivos
- no modifica Neon ni R2
- informa partidos detectados, omitidos y planificados

Ejecutar migración real:

```bash
python scripts/migrate_event_data_to_remote.py --execute
```

Generar reporte estructurado:

```bash
python scripts/migrate_event_data_to_remote.py --report-file logs/migration-report.json
python scripts/migrate_event_data_to_remote.py --execute --report-file logs/migration-report.json
```

Filtrar o acotar:

```bash
python scripts/migrate_event_data_to_remote.py --provider "StatsBomb Open Data"
python scripts/migrate_event_data_to_remote.py --match-id 3895302
python scripts/migrate_event_data_to_remote.py --limit 10
```

Reglas de seguridad:

- por defecto omite partidos que ya existen remoto
- valida que existan los tres archivos locales: `raw`, `canonical`, `metrics`
- valida carga remota posterior al guardado
- no elimina `SQLite` ni `JSON` locales

Sobrescritura controlada:

```bash
python scripts/migrate_event_data_to_remote.py --execute --include-existing
```

Usar `--include-existing` con cuidado, porque puede generar nuevas versiones remotas del mismo asset.
