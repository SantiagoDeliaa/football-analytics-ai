# Persistencia Local (MVP)

## Objetivo
Permitir reutilización de partidos procesados sin redescarga completa.

## Componentes
- SQLite:
  - `data/tip_event_data.sqlite`
- JSON locales:
  - `data/event_data/raw/statsbomb_open_data/`
  - `data/event_data/canonical/statsbomb_open_data/`
  - `data/event_data/metrics/statsbomb_open_data/`
  - `data/event_data/raw/api_football/`
  - `data/event_data/canonical/api_football/`
  - `data/event_data/metrics/api_football/`

## Fixtures locales de referencia
- `data/tip_event_data.sqlite`
- `data/event_data/raw/statsbomb_open_data/3895302.json`
- `data/event_data/canonical/statsbomb_open_data/3895302.json`
- `data/event_data/metrics/statsbomb_open_data/3895302.json`

Estos archivos sirven como contexto minimo reproducible para agentes, pruebas manuales y validacion de persistencia local.
Se mantienen locales en el workspace y `data/` continua ignorado por Git para no versionar persistencia de demo.

## Estrategia
- SQLite guarda metadata de partidos procesados.
- JSON guarda payloads grandes:
  - raw events
  - canonical events
  - metrics
- En API-Football, el raw payload puede ser combinado:
  - `events`
  - `lineups`
  - `statistics`
  - `players`

## Tabla principal
`processed_matches`
- `id`
- `provider`
- `match_id`
- `competition_name`
- `season_name`
- `home_team`
- `away_team`
- `match_date`
- `raw_path`
- `canonical_path`
- `metrics_path`
- `created_at`
- `updated_at`
- `UNIQUE(provider, match_id)`

## Funciones disponibles
- `save_processed_match()`
- `get_processed_matches()`
- `get_processed_match()`
- `load_processed_match_payloads()`
- `has_processed_match()`

## Consideraciones
- Persistencia liviana orientada a demo/MVP.
- Sin backend externo ni locking avanzado.
- Manejar errores de IO/DB sin romper UI.

## Evolución futura
- SQLite puede migrar a PostgreSQL manteniendo la capa repository.
- Evitar acceso directo a SQLite desde la UI fuera del repository.
