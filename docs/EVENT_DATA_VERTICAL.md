# Vertical 2 — Event Data

## Propósito
Transformar eventos de partido (PDF o API) en lectura táctica accionable: métricas, insights y visualizaciones sobre cancha.

## Tabs actuales
- `Subir PDF`
- `API Event Data`

## Flujo Subir PDF
- Se mantiene como flujo de demo/fallback.
- Es crítico y no debe romperse.

## Flujo Datos por API
- Provider activo: StatsBomb Open Data.
- Provider activo: API-Football / API-SPORTS.
- API-Football requiere `API_FOOTBALL_KEY` por variable de entorno.
- API-Football puede traer cobertura amplia y partidos actuales, pero no siempre ofrece coordenadas event-level.

## Pipeline operativo
```text
Provider
→ Ingestion
→ Canonical Event Model
→ Metrics
→ Insights
→ Visualizations
→ Local Persistence
```

## Archivos principales
- `src/verticals/vertical2.py`
- `src/verticals/vertical2_api_event.py`
- `src/services/open_event_data_ingestion.py`
- `src/services/open_event_normalizer.py`
- `src/services/api_football_ingestion.py`
- `src/services/api_football_normalizer.py`
- `src/services/open_event_metrics.py`
- `src/services/open_event_insights.py`
- `src/services/open_event_visualizations.py`
- `src/services/storage/database.py`
- `src/services/storage/event_data_repository.py`

## Reglas de diseño
- No acoplar UI directamente al provider.
- Todo provider debe pasar por ingestion/adapter.
- Toda métrica debe calcularse sobre Canonical Event Model.
- Mantener textos visibles en español.
- No romper flujo PDF de Vertical 2.
- Si un provider no entrega coordenadas, no se deben forzar mapas de cancha.
