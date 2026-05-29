# Provider Capabilities

## Que es esta capa

La capa de `Provider Capabilities` define, de forma centralizada, que datos y modulos puede habilitar cada proveedor de datos dentro de Tactical Intelligence Platform (TIP).

Su objetivo es que Vertical 2 pueda tomar decisiones de producto y orquestacion sin acoplar la UI al schema crudo de cada provider.

## Por que TIP no debe depender de un proveedor unico

TIP debe mantenerse `provider-agnostic` para poder:

- preservar el flujo actual de `StatsBomb Open Data` sin bloquear la evolucion futura;
- preparar integraciones pagas o comerciales como `Sportmonks` sin reescribir la UI;
- degradar funcionalidades de forma explicita cuando un provider no ofrece ciertos datos;
- evitar que componentes de frontend dependan de nombres de campos o estructuras inestables de cada API.

## Que significa que la UI sea provider-aware

Ser `provider-aware` no significa que la UI conozca el payload de cada proveedor.

Significa que la UI consulta una capa intermedia interna que responde preguntas como:

- si hay contexto suficiente para construir el partido;
- si pueden mostrarse standings o lineups;
- si hay coordenadas para habilitar visualizaciones tacticas;
- si AI Coach tiene suficiente contexto para operar.

La UI debe reaccionar a `capabilities` y modulos habilitados, no al schema original del provider.

## Modulos que se habilitan segun capabilities

La capa expone `get_enabled_modules_for_provider(provider_name)` y devuelve:

```python
{
    "match_context": True,
    "standings": False,
    "lineups": True,
    "team_stats": True,
    "player_stats": True,
    "event_maps": False,
    "shot_map": False,
    "progressive_actions_map": False,
    "ai_coach": True,
}
```

Reglas actuales:

- `match_context` se habilita si el provider aporta fixtures, historial o temporadas actuales.
- `standings` se habilita si existe `has_standings`.
- `lineups` se habilita si existe `has_lineups` o `has_formations`.
- `team_stats` se habilita si existe `has_team_stats`.
- `player_stats` se habilita si existe `has_player_stats`.
- `event_maps`, `shot_map` y `progressive_actions_map` solo se habilitan si `has_event_coordinates` es `true`.
- `ai_coach` puede habilitarse si existe al menos `match_context`, `team_stats`, `player_stats` o `event_data`.

## Regla tactica importante

No se deben simular visualizaciones tacticas si el provider no trae los datos necesarios.

En particular:

- si no hay coordenadas de eventos, no deben mostrarse mapas espaciales;
- si no hay datos suficientes, AI Coach debe degradar de forma controlada;
- si un provider tiene disponibilidad limitada o dependiente del plan, esa restriccion debe representarse en `capabilities`, no esconderse en la UI.

## Rol interno de esta capa

El usuario final idealmente no deberia elegir provider de forma manual.

Esta capa existe principalmente para uso interno de la plataforma, de modo que backend y frontend puedan:

- saber que modulos pueden activarse;
- preparar integraciones futuras sin romper providers existentes;
- mantener la demo coherente aunque convivan fuentes con distintos niveles de profundidad.

## Providers base contemplados

- `statsbomb_open_data`
- `api_football`
- `sportmonks`

`Sportmonks` queda preparado solo a nivel de arquitectura. No se conecta ni se consume todavia.
