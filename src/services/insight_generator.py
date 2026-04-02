from __future__ import annotations

from typing import Any


def _get_signal(normalized_payload: dict[str, Any], section: str, key: str) -> float:
    section_data = normalized_payload.get(section, {})
    signals = section_data.get("signals", {})
    try:
        return float(signals.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _metric_score(proprietary_metrics: dict[str, dict[str, Any]], key: str) -> int:
    try:
        return int(proprietary_metrics.get(key, {}).get("score", 0) or 0)
    except Exception:
        return 0


def generate_match_insights(
    normalized_payload: dict[str, Any],
    proprietary_metrics: dict[str, dict[str, Any]],
) -> list[str]:
    insights: list[str] = []

    control = _metric_score(proprietary_metrics, "field_tilt_index")
    verticality = _metric_score(proprietary_metrics, "directness_index")
    press = _metric_score(proprietary_metrics, "pressing_efficiency")
    risk = _metric_score(proprietary_metrics, "risk_exposure_score")

    final_third = _get_signal(normalized_payload, "attack", "final third")
    crosses = _get_signal(normalized_payload, "attack", "crosses")
    box_entries = _get_signal(normalized_payload, "attack", "box entries")
    shots = _get_signal(normalized_payload, "attack", "shots")
    turnovers = _get_signal(normalized_payload, "transitions", "turnover")
    recoveries = _get_signal(normalized_payload, "defense", "recoveries")
    regain = _get_signal(normalized_payload, "transitions", "regain")

    if control >= 70:
        insights.append("El equipo dominó territorio y sostuvo juego en campo rival.")
    elif control <= 40:
        insights.append("Faltó control territorial y el equipo pasó poco tiempo en campo rival.")
    else:
        insights.append("El control territorial fue intermitente, con tramos de dominio parcial.")

    if verticality >= 70:
        insights.append("El ataque fue directo y avanzó rápido hacia zonas de remate.")
    elif verticality <= 40:
        insights.append("La progresión fue lenta y costó acelerar hacia el área rival.")
    else:
        insights.append("El ritmo de progresión fue moderado, sin mucha ruptura vertical.")

    if press >= 65 and (recoveries + regain) > 0:
        insights.append("Las recuperaciones altas se tradujeron en secuencias con peligro real.")
    elif press <= 40:
        insights.append("La presión alta recuperó poco o no logró transformar recuperaciones en daño.")
    else:
        insights.append("La presión tuvo impacto mixto: recuperó, pero con amenaza irregular.")

    if risk >= 65 or turnovers >= 3:
        insights.append("Las pérdidas en salida expusieron al equipo en zonas sensibles.")
    elif risk <= 35:
        insights.append("La salida fue segura y el equipo protegió bien zonas de riesgo.")
    else:
        insights.append("La salida mostró momentos de riesgo, aunque sin descontrol constante.")

    left_bias_signal = crosses + box_entries
    final_third_activity = final_third + shots
    if left_bias_signal >= 3:
        insights.append("Gran parte del volumen ofensivo llegó por carriles exteriores.")
    elif final_third_activity <= 1:
        insights.append("Costó sostener actividad ofensiva estable en el último tercio.")
    else:
        insights.append("La producción ofensiva fue repartida, sin una vía dominante muy marcada.")

    unique_insights: list[str] = []
    for item in insights:
        if item not in unique_insights:
            unique_insights.append(item)
    return unique_insights[:5]
