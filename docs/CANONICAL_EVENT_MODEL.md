# Canonical Event Model

## Objetivo
Definir una estructura única de eventos para desacoplar métricas/insights/visualizaciones de cualquier provider específico.

## Ejemplo JSON
```json
{
  "event_id": "string",
  "match_id": "string",
  "team_id": "string",
  "team_name": "string",
  "player_id": "string",
  "player_name": "string",
  "minute": 0,
  "second": 0,
  "event_type": "Pass",
  "x": 45.0,
  "y": 30.0,
  "end_x": 70.0,
  "end_y": 35.0,
  "outcome": "Complete",
  "progressive": true,
  "under_pressure": false,
  "xG": 0.0,
  "xA": 0.0
}
```

## Significado de campos
- `event_id`: identificador del evento.
- `match_id`: identificador del partido.
- `team_id`, `team_name`: equipo del evento.
- `player_id`, `player_name`: jugador del evento.
- `minute`, `second`: timestamp del evento.
- `event_type`: tipo normalizado del evento.
- `x`, `y`: coordenadas de inicio.
- `end_x`, `end_y`: coordenadas de fin (si aplica).
- `outcome`: resultado del evento (si aplica).
- `progressive`: flag de progresión.
- `under_pressure`: evento bajo presión.
- `xG`: calidad de remate esperada.
- `xA`: expected assist (actualmente placeholder).

## Campos opcionales
- `x`, `y`, `end_x`, `end_y`, `outcome` pueden ser `null` según tipo de evento.
- Providers como API-Football pueden no entregar coordenadas; en ese caso `x`, `y`, `end_x`, `end_y` quedan en `null`.

## Escala de coordenadas
- StatsBomb usa cancha `120x80`.

## Criterios actuales de normalización
- `progressive=true` si `end_x - x >= 15`.
- `xG` viene de `shot.statsbomb_xg`.
- `xA` queda en `0.0` por ahora.
- En API-Football, `event_id` se genera de forma estable con `fixture_id + índice + tipo + tiempo`.

## Limitaciones actuales
- Algunas fuentes no proveen todos los campos.
- `xA` aún no está calculado de forma real.
- Mapeos de outcome pueden expandirse por provider.
- API-Football puede aportar eventos, estadísticas, lineups y jugadores sin coordenadas tácticas de cancha.

## Regla para nuevos provider adapters
- El adapter debe mapear al modelo canónico sin saltarse ingestion.
- Debe manejar faltantes sin romper el pipeline.
- Debe documentar explícitamente si el provider soporta o no coordenadas para visualizaciones de cancha.
