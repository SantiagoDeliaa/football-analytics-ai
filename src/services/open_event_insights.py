from __future__ import annotations

from typing import Any


def generate_open_event_insights(
    metrics: dict[str, Any],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> list[str]:
    if selected_player:
        scope = "El jugador seleccionado"
    elif selected_team:
        scope = "El equipo seleccionado"
    else:
        scope = "El recorte seleccionado"
    insights: list[str] = []

    total_events = int(metrics.get("total_events", 0))
    passes = int(metrics.get("total_passes", 0))
    shots = int(metrics.get("total_shots", 0))
    progressive = int(metrics.get("progressive_actions", 0))
    final_third = int(metrics.get("final_third_actions", 0))
    recoveries = int(metrics.get("recoveries", 0))
    field_tilt = metrics.get("field_tilt_index")
    directness = float(metrics.get("directness_index", 0.0) or 0.0)
    threat = float(metrics.get("progressive_threat_index", 0.0) or 0.0)
    recovery_height = metrics.get("recovery_height_index")
    player_influence = metrics.get("player_influence_score")

    if total_events == 0:
        return ["No se detectaron eventos para los filtros seleccionados."]

    insights.append(f"{scope} participó en {total_events} eventos registrados.")
    insights.append(f"Se registraron {passes} pases y {shots} remates en el recorte analizado.")
    insights.append(f"Se detectaron {progressive} acciones progresivas y {final_third} acciones en último tercio.")
    if isinstance(field_tilt, (int, float)) and field_tilt >= 70:
        insights.append("El equipo seleccionado tuvo un Field Tilt alto, señal de dominio territorial en campo rival.")
    if directness >= 65:
        insights.append("El índice de verticalidad indica una progresión ofensiva agresiva.")
    if threat >= 65:
        insights.append("La amenaza progresiva se mantuvo en niveles altos durante el recorte.")
    if isinstance(recovery_height, (int, float)) and recovery_height >= 60:
        insights.append("La altura promedio de recuperación sugiere presión efectiva en zonas adelantadas.")
    if selected_player and isinstance(player_influence, (int, float)) and player_influence >= 70:
        insights.append("El jugador seleccionado muestra alta influencia en las acciones ofensivas del equipo.")
    if recoveries > 0 and len(insights) < 5:
        insights.append(f"El volumen de recuperaciones defensivas fue de {recoveries} acciones.")

    return insights[:5]
