# Integracion de Sportmonks

## Rol de Sportmonks dentro de TIP

`Sportmonks` sera el primer provider comercial y orientado a `current-data` dentro de Tactical Intelligence Platform.

Su uso inicial dentro de TIP apunta a cubrir:

- partidos actuales;
- contexto de partido;
- fixtures;
- lineups;
- estadisticas de equipo;
- estadisticas de jugador;
- timeline de eventos principales;
- `xG`, `xGoT` y `xPTS` si el plan y los endpoints lo permiten;
- AI Coach contextual;
- Match Center.

En esta etapa ya existen:

- un cliente base seguro;
- una capa de normalizacion hacia modelos canonicos;
- un adapter de alto nivel para contexto de partido.

Todavia no se conecta directamente a la UI principal ni a las metricas propietarias.

## Que no se debe asumir

- No asumir coordenadas `x/y` de eventos.
- No asumir tracking.
- No asumir que todos los endpoints estan disponibles en todos los planes.
- No asumir que todas las ligas tienen la misma profundidad de datos.
- No construir mapas tacticos espaciales desde Sportmonks si no hay coordenadas confirmadas.
- No usar raw responses del provider como producto final.

## Configuracion

Para habilitar el cliente:

1. Crear la variable de entorno `SPORTMONKS_API_KEY`.
2. No subir `.env` al repositorio.
3. Usar `.env.example` como referencia de configuracion.

Ejemplo:

```env
SPORTMONKS_API_KEY=
```

## Cliente

El cliente vive en `src/services/providers/sportmonks_client.py`.

Funciones principales:

- `get_sportmonks_api_key()`
- `is_sportmonks_configured()`
- `build_sportmonks_url()`
- `sportmonks_get()`
- `extract_sportmonks_payload()`

### Respuesta normalizada

Todas las consultas GET devuelven una estructura normalizada:

```python
{
    "ok": True,
    "data": ...,
    "error": "",
    "status_code": 200,
    "meta": {...},
    "pagination": {...},
}
```

### Manejo de errores

El cliente maneja de forma segura:

- falta de `SPORTMONKS_API_KEY`;
- timeout;
- errores HTTP;
- problemas de red;
- JSON invalido;
- respuesta vacia;
- formato inesperado;
- autenticacion/autorizacion `401/403`;
- `404`;
- `429` por rate limit;
- errores `500`.

La regla es no propagar excepciones hacia la UI. En su lugar, el cliente devuelve `ok=False` y un mensaje en espanol.

### Includes

`get_sportmonks_fixture_full_context()` usa un include inicial configurable para Match Center:

- `participants`
- `league`
- `season`
- `venue`
- `state`
- `scores`
- `events`
- `lineups`
- `statistics`
- `metadata`

Estos includes son una base tentativa y deben confirmarse con respuestas reales del plan disponible.

### Rate limits

Si Sportmonks devuelve metadatos de rate limit, el cliente los conserva dentro de `meta` para debug interno seguro.

## Normalizacion Sportmonks -> Canonical Models

La normalizacion vive en `src/services/providers/sportmonks_normalizer.py`.

Su responsabilidad es transformar respuestas de Sportmonks hacia modelos canonicos de TIP, sin exponer raw responses a producto y sin calcular metricas propietarias desde datos crudos.

Modelos que hoy puede llenar:

- `CanonicalCompetition`
- `CanonicalSeason`
- `CanonicalTeam`
- `CanonicalPlayer`
- `CanonicalMatch`
- `CanonicalLineup`
- `CanonicalEventTimelineItem`
- `CanonicalTeamExpectedMetrics`
- `CanonicalTeamMatchStats`
- `CanonicalPlayerMatchStats`
- `CanonicalEventAvailability`

## Adapter de alto nivel

El adapter vive en `src/services/providers/sportmonks_adapter.py`.

Funciones principales:

- `get_available_sportmonks_competitions()`
- `get_sportmonks_fixtures_by_date()`
- `get_sportmonks_match_context()`
- `get_sportmonks_data_availability_for_fixture()`

`get_sportmonks_match_context()` devuelve una estructura segura y desacoplada:

```python
{
    "ok": True,
    "data": {
        "match": ...,
        "lineups": [...],
        "timeline_events": [...],
        "expected_metrics": [...],
        "team_stats": [...],
        "player_stats": [...],
        "availability": ...,
    },
    "error": None,
    "source": "sportmonks",
    "raw_meta": {...},
}
```

## Por que no se habilitan mapas tacticos

Sportmonks se usa inicialmente para:

- contexto de partido;
- lineups;
- estadisticas;
- timeline;
- expected metrics.

No se asumen coordenadas `x/y` de eventos. Por eso:

- no se habilitan mapas tacticos;
- no se generan pitch maps;
- no se calculan metricas espaciales;
- `has_coordinates` permanece en `False` salvo confirmacion explicita con API real.

## Campos ya mapeados

Hoy quedan contemplados, cuando la respuesta los trae:

- fixture id;
- liga y temporada;
- estadio y ciudad;
- participantes home/away;
- scores;
- estado del partido;
- eventos principales;
- goles, tarjetas, sustituciones, VAR, penalty y own goal;
- lineups;
- estadisticas de equipo;
- estadisticas de jugador;
- `xG`, `xGoT`, `xPTS`, `npxG`, `xG` open play, `xG` set play, `xG` free kicks, `shooting performance` y `xGA`.

## Campos pendientes de confirmar con API real

- nombres finales de algunos includes segun plan;
- endpoint exacto para fixtures por liga y temporada;
- forma final de `statistics` segun competicion y plan;
- presencia consistente de `xGoT`, `xPTS`, `npxG` y demas expected metrics;
- estructura real de `lineups`, `players` y `coach`;
- si conviene un endpoint especifico para events/lineups/statistics o usar fixture full context.

## Futuro Match Center

Esta capa esta pensada para una futura UI de Match Center donde el frontend consuma contratos canonicos y provider-aware, sin quedar acoplado al payload crudo de Sportmonks.

## UI tecnica inicial / Match Center

Dentro de `Datos por API` en Vertical 2 ahora existe una integracion tecnica inicial para `Sportmonks`.

Esta UI tecnica permite:

- buscar partidos por fecha;
- seleccionar un fixture;
- cargar un Match Center basado en modelos canonicos;
- ver encabezado del partido, resumen, timeline, lineups, estadisticas, expected metrics y disponibilidad de datos.

Esta UI no muestra:

- raw responses del provider como producto;
- mapas tacticos espaciales;
- metricas espaciales derivadas;
- pitch maps o mapas individuales de acciones.

La razon es simple: mientras `availability.has_coordinates` sea `False`, Sportmonks no debe habilitar visualizaciones de cancha.

Esta integracion es tecnica y transitoria. La UX final sera `provider-agnostic`, pero por ahora se mantiene el selector tecnico de provider para validar la arquitectura y el flujo de datos.

## Arquitectura objetivo

- La integracion productiva de `Sportmonks` debe exponerse por `FastAPI`.
- `React` debe consumir endpoints TIP, no payloads ni cliente directo de Sportmonks.
- `Streamlit` no es el frontend principal del producto.
- La UI tecnica anterior basada en Streamlit queda como legacy temporal.

## Proximo paso

El siguiente paso recomendado es crear:

- endpoints `FastAPI` para exponer `get_sportmonks_match_context()` y disponibilidad asociada;
- componente React Match Center en `front-tip/`;
- seleccion provider-aware dentro de Vertical 2;
- validacion con respuestas reales del plan activo.

## TODO

- Confirmar los includes finales soportados por el plan real.
- Confirmar el endpoint exacto para fixtures por liga y temporada.
- Confirmar si conviene usar endpoints especificos para `events`, `lineups` y `statistics` o resolverlos desde fixture full context.
