# Métricas Tácticas — Datos por API

## Principio
Las métricas se calculan sobre `Canonical Event Model`, no sobre payload crudo del provider.

## Compatibilidad por provider
- StatsBomb Open Data ofrece mejor soporte para métricas espaciales y visualizaciones de cancha.
- API-Football puede no entregar coordenadas de eventos; en ese caso, las métricas que dependen de `x/y` pueden devolver `No aplica`.
- Cuando faltan coordenadas, el dashboard prioriza resumen, timeline, estadísticas por equipo, lineups y jugadores.

## Filtro de eventos analíticos
Se usa `filter_analytical_events()` para excluir ruido:
- `Starting XI`
- `Half Start`
- `Half End`
- `Substitution`
- `Tactical Shift`
- `Bad Behaviour`

## A. Métricas básicas
- `total_events`
- `total_passes`
- `total_shots`
- `progressive_actions`
- `final_third_actions`
- `recoveries`
- `total_carries`
- `total_under_pressure`
- `total_xg`

## B. Métricas propietarias iniciales
### 1) Field Tilt Index
- **Propósito**: estimar dominio territorial en último tercio.
- **Fórmula**: acciones en último tercio del equipo / acciones en último tercio de ambos equipos * 100.
- **Campos**: `field_tilt_index`, `field_tilt_label`.
- **Interpretación**: mayor valor implica mayor dominio territorial.
- **Limitación**: si no hay equipo filtrado, puede no aplicar.
- **Dependencia espacial**: requiere coordenadas `x`.

### 2) Directness Index
- **Propósito**: medir verticalidad de progresión.
- **Fórmula**: acciones progresivas / pases totales * 100.
- **Campos**: `directness_index`, `directness_label`.
- **Interpretación**: más alto, mayor agresividad vertical.
- **Limitación**: con pocos pases puede ser inestable.
- **Dependencia espacial**: mejora cuando existen progresiones detectables.

### 3) Progressive Threat Index
- **Propósito**: medir amenaza ofensiva combinada.
- **Fórmula aproximada**: combinación heurística de progresiones, último tercio, remates y xG; normalizada 0-100.
- **Campos**: `progressive_threat_index`, `progressive_threat_label`.
- **Interpretación**: más alto, mayor potencial de daño ofensivo.
- **Limitación**: heurística MVP, no modelo calibrado final.

### 4) Recovery Height Index
- **Propósito**: medir altura territorial de recuperación.
- **Fórmula**: promedio de `x` en recuperaciones favorables / 120 * 100.
- **Campos**: `recovery_height_index`, `recovery_height_label`.
- **Interpretación**: más alto, presión/recuperación más adelantada.
- **Limitación**: depende de cantidad/calidad de eventos defensivos.
- **Dependencia espacial**: requiere coordenadas `x` en recuperaciones.

### 5) Shot Quality Index
- **Propósito**: evaluar calidad media de remate.
- **Fórmula**: `(total_xg / total_shots) * 100`.
- **Campos**: `shot_quality_index`, `shot_quality_label`.
- **Interpretación**: más alto, mejores posiciones de remate.
- **Limitación**: sin remates devuelve 0.

### 6) Player Influence Score
- **Propósito**: estimar influencia del jugador en el recorte.
- **Fórmula aproximada**: combinación heurística de volumen, pases, progresión, último tercio, remates y recuperaciones.
- **Campos**: `player_influence_score`, `player_influence_label`.
- **Interpretación**: más alto, mayor involucramiento global.
- **Limitación**: solo aplica con jugador específico.

## Labels de nivel
- `Bajo`
- `Medio`
- `Alto`
- `No aplica` cuando corresponde.

## Notas de evolución
- Las fórmulas actuales son heurísticas MVP.
- Pueden evolucionar con calibración y validación táctica.
- Si se carga historial viejo sin métricas propietarias, la UI puede recalcular desde `canonical_events`.
