# ⚙️ Football Tactical Analytics  
## Clip Debug Workflow Operativo

---

# 🎯 Objetivo

Establecer un procedimiento profesional y reproducible para debuggear cada clip procesado.

**Regla principal:**  
Nunca arreglar dos capas al mismo tiempo.  
Siempre seguir el orden del pipeline.

---

# 🔵 Paso 0 – Pre-Check (30 segundos)

Antes de abrir cualquier chat o modificar código:

- ¿El clip tiene cámara abierta?
- ¿Se ven líneas del campo?
- ¿No es replay?
- ¿No es zoom extremo?
- ¿No es cámara lateral extremadamente cerrada?

⚠️ Si el clip es visualmente malo, no debuggear el sistema. Cambiar clip.

---

# 🟢 Paso 1 – Homografía (Siempre primero)

Abrir chat especializado: **HOMOGRAFÍA**

## Métricas a revisar

- `fallback_ratio`
- `p95_reproj_error_m`
- `p95_delta_H`
- `cut_detected_ratio`
- distribución de `H_accept_reason`

## Stop Condition

No avanzar hasta que:

- `fallback_ratio ≤ 20%`
- `p95_reproj_error_m ≤ 1.5m`
- `cut_detected` no supere ~10–15% sin cortes reales

❗ Si homografía está mal → todo lo demás es inválido.

---

# 🟡 Paso 2 – Tracking / Re-ID

Abrir chat especializado: **TRACKING**

## Métricas a revisar

- `tracks_active`
- `churn_ratio`
- `avg_track_age`
- `short_tracks_ratio`

## Stop Condition

- `churn_ratio p95 < 0.5`
- `avg_track_age` razonable (> 20 frames en clips normales)
- `short_tracks_ratio` no dominante

❗ Si tracking está fragmentado → velocidad y posesión serán falsas.

---

# 🟠 Paso 3 – Velocidad

Abrir chat especializado: **VELOCIDAD**

## Métricas a revisar

- `p95_speed_mps`
- `speed_violation_ratio`
- `jump_violation_ratio`

## Stop Condition

- `p95_speed_mps ≤ 9 m/s`
- `speed_violation_ratio ≤ 10%`

Si no se cumple:

- Cambiar fórmula (ventana temporal)
- Agregar smoothing (EMA o median)
- Ignorar frames inestables

---

# 🔴 Paso 4 – Posesión

Abrir chat especializado: **POSESIÓN**

## Métricas a revisar

- `ball_detected_ratio`
- `ball_track_age`
- `contested_ratio`
- `contested_reason`

## Stop Condition

- `ball_detected_ratio ≥ 30%`
- `contested` no dominante
- señal de balón consistente

Si no hay balón → `possession_state = "unknown"`  
No forzar porcentajes.

---

# 🟣 Paso 5 – Formación

Abrir chat especializado: **FORMACIÓN**

## Métricas a revisar

- `invalid_formation_ratio`
- `team_attack_direction`
- `sum == 10`
- estabilidad temporal

## Stop Condition

- `invalid_ratio ≤ 30%`
- direction definida
- formación estable por ventana temporal

---

# 🧠 Principio Clave

Nunca:

- Arreglar velocidad si homografía está mal.
- Arreglar formación si direction no está definida.
- Ajustar thresholds sin mirar distribución real.

Siempre:

1. Medir  
2. Corregir  
3. Reprocesar  
4. Volver a medir  

---

# 📊 Criterio Final Demo-Ready

Un clip se considera aprobado cuando:

- Homografía estable
- Tracking consistente
- Velocidad físicamente plausible
- Posesión basada en señal real
- Formación válida

Y esto se cumple en **al menos 2 clips distintos**.

---

# 🧩 Nivel Profesional (Siguiente Escalón)

Cuando el sistema esté estable:

- Crear dataset de 10 clips variados
- Ejecutar regresión automática
- Generar tabla comparativa de `health_summary`
- Detectar regresiones entre versiones

---

# 🔥 Regla de Oro

Si una capa está mal,  
todo lo que está encima está mal.

Orden obligatorio:

Homografía → Tracking → Velocidad → Posesión → Formación