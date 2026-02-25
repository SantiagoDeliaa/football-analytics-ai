# 🧭 Football Tactical Analytics  
## Pipeline Dependency Map (Qué rompe qué)

---

# 🎯 Objetivo

Tener un mapa mental claro del pipeline para debuggear sin mezclar causas.

**Regla:** si una capa está mal, contamina todo lo que está arriba.

---

# 1 Pipeline por capas (de abajo hacia arriba)

## Capa 0 — Input / Video
**Responsable:** calidad visual del broadcast (cámara, zoom, motion blur, cortes)

**Si falla:**
- keypoints del campo pobres
- pelota difícil
- detecciones inestables

**Impacta:** Homografía + Detección + Tracking

---

## Capa 1 — Detección (players/ref/ball)
**Responsable:** detector (YOLO/ultralytics/etc.), umbrales y filtros por clase

**Outputs típicos:**
- `players_detected_count`
- `refs_detected_count`
- `ball_detected`

**Si falla:**
- tracking fragmentado
- posesión desconocida
- formaciones incomputables

**Impacta:** Tracking + Posesión + Formación + Métricas

---

## Capa 2 — Tracking (ByteTrack / IDs)
**Responsable:** persistencia temporal de detecciones

**Outputs típicos:**
- `tracks_active`
- `avg_track_age`
- `churn_ratio`
- `short_tracks_ratio`

**Si falla:**
- velocidades infladas (IDs se resetean)
- jugadores “fantasma”
- conteos erróneos por acumulación de IDs
- formaciones erráticas (posiciones sin continuidad)

**Impacta:** Velocidad + Formación + Métricas agregadas

---

## Capa 3 — Homografía (proyección 2D 105x68m)
**Responsable:** `HomographyManager` (H estable, cut detection, last_good_H)

**Outputs típicos:**
- `reproj_error_m`
- `delta_H`
- `homography_status`
- `homography_mode`
- `cut_detected`

**Si falla:**
- todo lo espacial es basura (distancias, centroides, líneas)
- formaciones inválidas
- velocidades absurdas por ruido geométrico
- heatmaps deformados

**Impacta:** Formación + Velocidad + Métricas tácticas + Heatmaps

> Nota: Homografía “mala” puede verse “más o menos” en radar, pero igual arruina métricas finas.

---

## Capa 4 — Normalización / Orientación (direction + flip)
**Responsable:** definir ataque izquierda→derecha o derecha→izquierda por equipo

**Outputs típicos:**
- `team_attack_direction`
- `flip_applied`

**Si falla:**
- `no_defenders`
- defensas/ataques invertidos
- formaciones imposibles aunque la homografía sea buena

**Impacta:** Formación + Métricas por “líneas” (def_line, press_height)

---

## Capa 5 — Métricas Cinemáticas (Velocidad / Jumps)
**Responsable:** fórmula + smoothing + ventana temporal + gating por health

**Outputs típicos:**
- `p95_speed_mps`
- `speed_violation_ratio`
- `jump_violation_ratio`

**Si falla:**
- picos irreales (40–50 km/h)
- conclusiones falsas (aceleraciones, esfuerzo)

**Depende de:** Tracking + Homografía + Cut detection

---

## Capa 6 — Posesión (ball + proximity logic)
**Responsable:** detección/track de balón + criterio de asignación

**Outputs típicos:**
- `ball_detected_ratio`
- `possession_state`
- `contested_reason`

**Si falla:**
- % posesión “decorativa”
- contested dominante por `no_ball`

**Depende de:** Detección balón + Tracking (balón) + (a veces) Homografía

---

## Capa 7 — Formación (clustering en 2D)
**Responsable:** mapping de posiciones a líneas (DEF/MID/ATT) + smoothing temporal

**Outputs típicos:**
- `formation_label`
- `formation_valid`
- `invalid_formation_ratio`

**Si falla:**
- formaciones 0-x-x
- sum != 10
- jitter frame-a-frame

**Depende de:** Homografía + Direction/Flip + Tracking (posiciones consistentes)

---

## Capa 8 — Agregación / Reporte / UI (demo-safe)
**Responsable:** `health_summary`, thresholds, gating de métricas y visualización

**Si falla:**
- “Confianza Alta” cuando no corresponde
- mostrar métricas inválidas (fruta) en PDF/UI

**Depende de:** todas las capas anteriores (especialmente health flags)

---

# 2 Matriz rápida: Síntoma → Probable causa

| Síntoma visible | Causa más probable | Chequeo inmediato |
|---|---|---|
| fallback_ratio alto | Homografía no estabiliza / estados | `homography_mode/status`, `H_accept_reason`, `reproj_error_m`, `delta_H` |
| cut_detected 30–40% | delta_H sensible / sin debounce | `delta_H_smoothed`, cooldown |
| velocidades 45–50 km/h | speed frame-a-frame + ruido | `p95_speed`, ventana temporal, smoothing |
| contested ~98% | no hay ball | `ball_detected_ratio`, `contested_reason=no_ball` |
| 0 defensores | direction/flip | `team_attack_direction`, `flip_applied` |
| player_count enorme | re-ID / churn | `churn_ratio`, `new_tracks_count`, `avg_track_age` |

---

# 3 Orden obligatorio de debugging

1. Homografía  
2. Tracking / churn  
3. Velocidad (ventana + smoothing)  
4. Balón / posesión (gating)  
5. Direction/flip  
6. Formación (smoothing + validación)  
7. UI demo-safe (bloquear métricas)

---

# 4 Reglas “Demo-safe” (no mostrar fruta)

- Si `ball_detected_ratio < 0.3` → posesión = **unknown** (no %)
- Si `invalid_formation_ratio > 0.3` → ocultar formaciones
- Si `p95_speed_mps > 9` → ocultar velocidad o usar “strict speed”
- Si `fallback_ratio > 0.2` o `p95_reproj_error_m > 1.5` → ocultar métricas espaciales

---

# 5 Checklist mínimo por clip (QA rápido)

- Homografía: fallback_ratio, p95_reproj_error_m, cut_detected_ratio  
- Tracking: churn_p95, avg_track_age  
- Velocidad: p95_speed_mps, violation_ratio  
- Balón: ball_detected_ratio, contested_reason  
- Formación: invalid_ratio, direction definida  
- UI: no mostrar métricas si fallan gates

---