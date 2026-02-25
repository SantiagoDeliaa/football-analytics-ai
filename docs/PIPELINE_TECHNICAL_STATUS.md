# ⚽ Football Tactical Analytics Pipeline  
## Technical Status & QA Report

---

# 1️⃣ Overview

Sistema de análisis táctico para fútbol broadcast basado en Computer Vision.

## Outputs

- 🎥 MP4 anotado
- 📄 JSON estructurado con:
  - Métricas tácticas (mean/std/min/max/current)
  - Timeline por frame
  - Heatmaps
  - `health_summary`
  - `timeline.health` (instrumentación avanzada QA)

---

# 2️⃣ Arquitectura

## Core Modules
src/
├── controllers/
│ └── process_video.py
├── utils/
│ └── homography_manager.py
├── tactical_metrics.py
├── formation_detector.py
├── app.py (Streamlit UI)
└── quality_config.py

---

# 3️⃣ Instrumentación Actual

## timeline.health (por frame)

- homography_mode
- homography_status
- reproj_error_m
- delta_H
- cut_detected
- H_accept_reason
- last_good_age_frames
- tracks_active
- new_tracks_count
- ended_tracks_count
- churn_ratio
- ball_detected
- possession_state
- contested_reason
- max_player_speed_mps
- speed_violation
- max_player_jump_m
- jump_violation

## health_summary (global)

- fallback_ratio
- invalid_formation_ratio
- p95_reproj_error_m
- p95_delta_H
- speed_violation_ratio
- jump_violation_ratio
- churn_p95
- ball_detected_ratio
- avg_tracks_active
- etc.

---

# 4️⃣ Problemas Detectados (Confirmados en Múltiples Clips)

## 4.1 Homografía

- homography_mode = "fallback" en 100% de frames
- H_accept_reason muestra accepted_ok / accepted_warn
- cut_detected ~35–40% frames (excesivo)
- delta_H p95 ~1.8–2.1 (muy alto)

### Diagnóstico

La máquina de estados de homografía está mal sincronizada.
Se aceptan matrices H pero no se transiciona correctamente de fallback → track/inertia.

---

## 4.2 Tracking

- tracks_active razonable (~18–20)
- churn presente por re-ID (IDs se regeneran si salen de frame)
- player_count inflado por acumulación histórica de IDs

Tracking no es el cuello principal, pero afecta:

- velocidad
- estabilidad de métricas
- posesión

---

## 4.3 Velocidad

- speed_violation_ratio muy alto (0.7–0.99)
- p95_speed ~12.8 m/s (~46 km/h)

### Causa

- cálculo frame-a-frame
- ruido geométrico amplificado
- sin suavizado temporal

---

## 4.4 Posesión

- ball_detected_ratio ~2–3%
- contested_reason dominante: "no_ball"
- posesión calculada sin señal estable

### Conclusión

La posesión actual no es fiable.

---

## 4.5 Formaciones

- invalid_formation_ratio = 1.0
- razones: no_defenders, sum_not_10

### Probable causa

- direction-of-play no definido
- flip inconsistente
- clasificación por eje X incorrecta

---

# 5️⃣ quality_config.py Actual

```python
REPROJ_OK_MAX = 0.8
REPROJ_WARN_MAX = 1.5
REPROJ_INVALID = 3.0
REPROJ_WARMUP_MAX = 2.0
DELTA_H_WARN = 0.1
DELTA_H_CUT = 0.2
MIN_TRACKS_ACTIVE = 8
MIN_DETECTIONS = 8
SHORT_TRACK_AGE = 10
IMPROVE_MARGIN = 0.05
WARMUP_FRAMES = 120
REACQUIRE_FRAMES = 40
```

# 6️⃣ Objetivo: Estado Demo-Ready

Para considerarse **demo-ready**, se deben cumplir estos umbrales mínimos:

## Homografía
- `fallback_ratio` ≤ 20%
- `cut_detected` realista (no disparado por ruido)
- `p95_reproj_error_m` ≤ 1.5 m

## Velocidad
- `p95_speed_mps` ≤ 9 m/s
- `speed_violation_ratio` ≤ 10%

## Posesión
- `ball_detected_ratio` ≥ 30%
- `contested` no dominante (idealmente < 40% si hay señal de balón)

## Formación
- `invalid_formation_ratio` ≤ 30%
- `direction` definida (team_attack_direction estable y consistente)

---

# 7️⃣ Plan de Corrección

## Fase 1 – Homografía estable
- Corregir transición de estados (`fallback` → `track/inertia/reacquire`) de forma coherente con `H_accept_reason`
- Implementar **debounce + cooldown** en `cut_detected`
- Cache coherente de `last_good_H` (no reemplazar H buena por H mala)

## Fase 2 – Velocidad robusta
- Calcular velocidad por **ventana temporal** (5–10 frames), no frame-a-frame
- Aplicar **suavizado** (EMA o median filter)
- Ignorar frames con `cut_detected` y/o `homography_status != ok` para velocidad “strict”

## Fase 3 – Posesión confiable
- Introducir estado `possession_state = "unknown"` cuando `ball_detected == False`
- No calcular %posesión si `ball_detected_ratio` es bajo (gating por calidad)
- Refinar `contested_reason` para diagnóstico (no_ball, low_conf, far, multiple_candidates)

## Fase 4 – Formación robusta
- Definir `team_attack_direction` (por warm-up/centroid/GK si disponible)
- Aplicar flip/orientación consistente antes de clasificar defenders/mids/attackers
- Validar líneas tras normalización (sum==10, defenders>=2, etc.) + smoothing temporal

---

# 8️⃣ Criterio QA Final

Sistema aprobado cuando:
- `health_summary` es consistente con `timeline.health`
- métricas se **bloquean** cuando health es bajo (no mostrar valores “fruta”)
- UI no muestra métricas no confiables (gating demo-safe)
- resultados estables en **≥ 2 clips distintos** (misma cámara + distinta escena)

---

# 9️⃣ Estado Actual Global

| Área | Estado |
|------|--------|
| Instrumentación | ✅ Avanzada |
| Homografía | ❌ Inestable |
| Tracking | ⚠️ Aceptable |
| Velocidad | ❌ Inflada |
| Posesión | ❌ No confiable |
| Formación | ❌ Inválida |
| Arquitectura | ✅ Bien estructurada |

---

# 🔒 Backup Status

Este documento refleja el estado técnico tras:
- Implementación de warm-up homography
- Instrumentación avanzada QA
- Detección de churn, velocidad y salud de balón
- Análisis sobre múltiples clips