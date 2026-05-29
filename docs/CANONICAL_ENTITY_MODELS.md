# Modelos Canonicos de Entidades

## Por que se agregan modelos canonicos no-evento

TIP ya cuenta con un flujo orientado a eventos canonicos, especialmente util para `StatsBomb Open Data` y visualizaciones tacticas con coordenadas.

Para integrar providers como `Sportmonks` y `API-Football` sin acoplar la plataforma a sus payloads crudos, ahora se agrega una capa complementaria de modelos canonicos no-evento.

Esta capa permite representar de forma estable:

- competiciones;
- temporadas;
- equipos;
- jugadores;
- partidos;
- lineups;
- timeline de eventos principales;
- estadisticas de equipo;
- estadisticas de jugador;
- metricas esperadas como `xG`, `xGoT` y `xPTS`;
- disponibilidad y calidad de datos.

## Diferencia entre CanonicalEvent espacial y modelos de contexto

El `Canonical Event Model` actual sigue siendo el contrato principal para analitica espacial y visualizaciones tacticas basadas en eventos con coordenadas.

Los nuevos modelos canonicos de entidades cubren el contexto alrededor del partido cuando un provider:

- aporta datos competitivos y estadisticos fuertes;
- trae timeline de eventos principales;
- expone `xG` y metricas esperadas;
- no garantiza coordenadas `x/y` de cada evento.

En otras palabras:

- el modelo de eventos espaciales sirve para mapas tacticos y acciones sobre cancha;
- los modelos de entidades sirven para match center, contexto, timeline, estadisticas y disponibilidad.

## Como Sportmonks debe mapearse a estos modelos

`Sportmonks` debe tratarse como un provider fuerte para:

- fixtures actuales;
- contexto competitivo;
- ligas y temporadas;
- equipos y jugadores;
- lineups;
- estadisticas de equipo;
- estadisticas de jugador;
- timeline de eventos principales;
- `xG`, `xGoT`, `xPTS`, `npxG`, `xG` en jugada, `xG` de pelota parada, `xG` de tiros libres, rendimiento de remate y `xG` concedido.

Su integracion debe mapear el payload raw a:

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
- `CanonicalDataAvailability`

No se debe asumir que `Sportmonks` trae coordenadas de eventos. Por lo tanto, la plataforma no debe habilitar mapas tacticos espaciales salvo que un endpoint o plan lo confirme explicitamente.

## Modelos que sirven para Match Center

- `CanonicalCompetition`
- `CanonicalSeason`
- `CanonicalTeam`
- `CanonicalMatch`
- `CanonicalLineup`
- `CanonicalTeamMatchStats`
- `CanonicalDataAvailability`

## Modelos que sirven para Timeline

- `CanonicalEventTimelineItem`
- `CanonicalEventAvailability`

## Modelos que sirven para xG, xGoT y xPTS

- `CanonicalTeamExpectedMetrics`
- `CanonicalEventAvailability`
- `CanonicalDataAvailability`

## Modelos que sirven para analisis de jugador

- `CanonicalPlayer`
- `CanonicalPlayerMatchStats`
- `CanonicalLineup`
- `CanonicalEventTimelineItem`

## Modelos que sirven para Data Availability

- `CanonicalEventAvailability`
- `CanonicalDataAvailability`

Estos modelos permiten decidir que modulos pueden habilitarse y con que nivel de confianza, sin acoplar la UI a detalles internos del provider.

## Reglas de implementacion

- La UI y las metricas no deben usar raw responses del provider.
- Las metricas propietarias no deben calcularse desde raw provider responses.
- Si no hay coordenadas, no se deben habilitar mapas tacticos espaciales.
- Los nuevos providers deben mapearse a modelos canonicos antes de llegar a la UI.
- `metadata` puede conservar informacion opcional para debug interno, pero no debe exponerse como producto.
