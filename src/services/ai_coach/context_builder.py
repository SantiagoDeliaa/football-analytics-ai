from __future__ import annotations

from collections import Counter
from typing import Any


RECOVERY_EVENT_TYPES = {"Ball Recovery", "Interception", "Duel"}
DEFAULT_SUGGESTED_QUESTIONS = [
    "¿Cómo estuvo el equipo en términos generales?",
    "¿Dónde generó más peligro?",
    "¿Qué debería corregir el cuerpo técnico?",
    "¿Qué jugador fue más influyente?",
    "¿Qué indican las métricas propietarias?",
    "¿El equipo fue vertical o más paciente?",
    "¿Qué limitaciones tienen estos datos?",
]
TACTICAL_METRIC_KEYS = [
    "total_events",
    "total_passes",
    "total_shots",
    "progressive_actions",
    "final_third_actions",
    "recoveries",
    "total_carries",
    "total_under_pressure",
    "total_xg",
    "field_tilt_index",
    "field_tilt_label",
    "directness_index",
    "directness_label",
    "progressive_threat_index",
    "progressive_threat_label",
    "recovery_height_index",
    "recovery_height_label",
    "shot_quality_index",
    "shot_quality_label",
    "player_influence_score",
    "player_influence_label",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _provider_slug(provider: Any) -> str:
    return str(provider or "unknown").strip().lower().replace(" ", "_")


def _normalize_match_metadata(match_metadata: dict[str, Any] | None, provider: Any) -> dict[str, Any]:
    metadata = match_metadata or {}
    return {
        "provider": str(provider or metadata.get("provider", "unknown")),
        "match_id": metadata.get("match_id"),
        "competition_name": metadata.get("competition_name"),
        "season_name": metadata.get("season_name"),
        "home_team": metadata.get("home_team"),
        "away_team": metadata.get("away_team"),
        "match_date": metadata.get("match_date"),
    }


def _infer_provider_capabilities(
    provider: Any,
    canonical_events: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    provider_capabilities: dict[str, Any] | None,
) -> dict[str, bool]:
    if provider_capabilities:
        return {
            "has_event_coordinates": bool(provider_capabilities.get("has_event_coordinates", False)),
            "has_lineups": bool(provider_capabilities.get("has_lineups", False)),
            "has_team_stats": bool(provider_capabilities.get("has_team_stats", False)),
            "has_player_stats": bool(provider_capabilities.get("has_player_stats", False)),
            "has_xg": bool(provider_capabilities.get("has_xg", False)),
            "has_event_timeline": bool(provider_capabilities.get("has_event_timeline", False)),
        }

    provider_key = _provider_slug(provider)
    has_coordinates = any(
        isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
        for event in canonical_events
    )
    has_xg = any((_safe_float(event.get("xG")) or 0.0) > 0 for event in canonical_events) or (
        (_safe_float((metrics or {}).get("total_xg")) or 0.0) > 0
    )
    return {
        "has_event_coordinates": has_coordinates,
        "has_lineups": provider_key in {"api_football", "api-football"},
        "has_team_stats": provider_key in {"api_football", "api-football"},
        "has_player_stats": provider_key in {"api_football", "api-football"},
        "has_xg": has_xg,
        "has_event_timeline": len(canonical_events) > 0,
    }


def _build_provider_limitations(capabilities: dict[str, bool]) -> list[str]:
    limitations: list[str] = []
    if not capabilities.get("has_event_coordinates", False):
        limitations.append("Este provider no entrega coordenadas de eventos.")
        limitations.append("Las métricas espaciales pueden no aplicar.")
    if not capabilities.get("has_xg", False):
        limitations.append("La calidad de remate y amenaza ofensiva pueden tener menor precisión.")
    if not capabilities.get("has_player_stats", False):
        limitations.append("Los datos disponibles son agregados o cronológicos.")
    return limitations


def _build_tactical_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    payload = metrics or {}
    return {key: payload.get(key) for key in TACTICAL_METRIC_KEYS}


def _build_tactical_summary(metrics: dict[str, Any], selected_player: str | None) -> dict[str, str]:
    field_tilt = _safe_float(metrics.get("field_tilt_index"))
    directness = _safe_float(metrics.get("directness_index"))
    threat = _safe_float(metrics.get("progressive_threat_index"))
    recovery_height = _safe_float(metrics.get("recovery_height_index"))
    shot_quality = _safe_float(metrics.get("shot_quality_index"))
    total_shots = _safe_float(metrics.get("total_shots")) or 0.0
    player_influence = _safe_float(metrics.get("player_influence_score"))

    attacking_profile = "Ataque con señales mixtas o insuficientes para una lectura fuerte."
    if threat is not None and threat > 65:
        attacking_profile = "Buena amenaza progresiva y capacidad de transformar avances en peligro."
    elif directness is not None and directness > 60:
        attacking_profile = "Equipo con tendencia vertical para acelerar sus ataques."
    elif directness is not None and directness < 40:
        attacking_profile = "Ataque más paciente o con menor agresividad en la progresión."

    territorial_profile = "Presencia territorial equilibrada o sin dominio claro."
    if field_tilt is not None and field_tilt > 65:
        territorial_profile = "Alta presencia territorial en campo rival."
    elif field_tilt is not None and field_tilt < 40:
        territorial_profile = "Presencia territorial baja en campo rival."

    pressing_profile = "Sin evidencia clara de una presión particularmente alta."
    if recovery_height is not None and recovery_height > 60:
        pressing_profile = "Recuperaciones relativamente altas, compatibles con una presión adelantada."
    elif recovery_height is not None and recovery_height < 40:
        pressing_profile = "Recuperaciones más retrasadas, con menor presión en campo rival."

    risk_profile = "Riesgo ofensivo y de finalización sin alertas claras."
    if shot_quality is not None and shot_quality < 20 and total_shots >= 5:
        risk_profile = "Volumen ofensivo con baja calidad media de remate."
    elif shot_quality is not None and shot_quality >= 35:
        risk_profile = "Remates de calidad media o alta respecto del volumen generado."

    player_profile = "No aplica para esta selección."
    if selected_player and selected_player != "Todos":
        if player_influence is not None and player_influence > 65:
            player_profile = "Jugador con alta influencia en las acciones del partido."
        elif player_influence is not None and player_influence > 40:
            player_profile = "Jugador con influencia intermedia en el desarrollo del partido."
        else:
            player_profile = "Jugador con influencia acotada o poco volumen de intervención."

    data_quality_note = "Contexto generado desde métricas e insights estructurados, sin exponer raw data completo."
    if metrics.get("field_tilt_index") is None and metrics.get("recovery_height_index") is None:
        data_quality_note = "Hay señales tácticas disponibles, pero parte del análisis espacial puede no aplicar."

    return {
        "attacking_profile": attacking_profile,
        "territorial_profile": territorial_profile,
        "pressing_profile": pressing_profile,
        "risk_profile": risk_profile,
        "player_profile": player_profile,
        "data_quality_note": data_quality_note,
    }


def _sorted_counter_items(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def _build_event_summary(canonical_events: list[dict[str, Any]]) -> dict[str, Any]:
    events = canonical_events or []
    event_type_counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    player_counts: Counter[str] = Counter()
    progressive_player_counts: Counter[str] = Counter()
    shot_player_counts: Counter[str] = Counter()
    recovery_player_counts: Counter[str] = Counter()
    events_with_coordinates = 0

    for event in events:
        event_type = str(event.get("event_type") or "Unknown")
        team_name = str(event.get("team_name") or "Equipo desconocido")
        player_name = str(event.get("player_name") or "Jugador desconocido")
        has_coordinates = isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
        if has_coordinates:
            events_with_coordinates += 1
        event_type_counts[event_type] += 1
        team_counts[team_name] += 1
        player_counts[player_name] += 1
        if bool(event.get("progressive")):
            progressive_player_counts[player_name] += 1
        if event_type == "Shot":
            shot_player_counts[player_name] += 1
        if event_type in RECOVERY_EVENT_TYPES:
            recovery_player_counts[player_name] += 1

    total_events = len(events)
    return {
        "total_canonical_events": total_events,
        "events_with_coordinates": events_with_coordinates,
        "events_without_coordinates": max(0, total_events - events_with_coordinates),
        "event_type_counts": dict(event_type_counts),
        "top_teams_by_events": _sorted_counter_items(team_counts),
        "top_players_by_events": _sorted_counter_items(player_counts),
        "top_players_by_progressive_actions": _sorted_counter_items(progressive_player_counts),
        "top_players_by_shots": _sorted_counter_items(shot_player_counts),
        "top_players_by_recoveries": _sorted_counter_items(recovery_player_counts),
    }


def _build_spatial_summary(provider: Any, canonical_events: list[dict[str, Any]]) -> dict[str, Any]:
    events = canonical_events or []
    coordinate_events = [
        event for event in events
        if isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
    ]
    if not coordinate_events:
        return {
            "has_spatial_data": False,
            "note": "No hay coordenadas suficientes para análisis espacial.",
        }

    final_third_actions = sum(1 for event in coordinate_events if (_safe_float(event.get("x")) or 0.0) >= 80.0)
    recovery_events = [event for event in coordinate_events if str(event.get("event_type") or "") in RECOVERY_EVENT_TYPES]
    average_event_x = sum(float(event["x"]) for event in coordinate_events) / len(coordinate_events)
    average_recovery_x = (
        sum(float(event["x"]) for event in recovery_events) / len(recovery_events)
        if recovery_events else None
    )
    provider_key = _provider_slug(provider)

    return {
        "has_spatial_data": True,
        "final_third_actions_count": final_third_actions,
        "average_event_x": round(average_event_x, 2),
        "average_recovery_x": round(average_recovery_x, 2) if average_recovery_x is not None else None,
        "coordinate_system": "StatsBomb 120x80" if "statsbomb" in provider_key else "No especificado",
    }


def build_match_context(
    provider: Any,
    match_metadata: dict[str, Any] | None,
    canonical_events: list[dict[str, Any]] | None,
    metrics: dict[str, Any] | None,
    insights: list[str] | None = None,
    selected_team: str | None = None,
    selected_player: str | None = None,
    provider_capabilities: dict[str, Any] | None = None,
    raw_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_events = canonical_events or []
    normalized_metrics = _build_tactical_metrics(metrics)
    match_section = _normalize_match_metadata(match_metadata, provider)
    match_section["selected_team"] = selected_team
    match_section["selected_player"] = selected_player

    capabilities = _infer_provider_capabilities(provider, normalized_events, metrics, provider_capabilities)
    provider_context = {
        "provider": str(provider or "unknown"),
        "capabilities": capabilities,
        "limitations": _build_provider_limitations(capabilities),
    }

    context = {
        "match": match_section,
        "provider_context": provider_context,
        "tactical_metrics": normalized_metrics,
        "tactical_summary": _build_tactical_summary(normalized_metrics, selected_player),
        "event_summary": _build_event_summary(normalized_events),
        "spatial_summary": _build_spatial_summary(provider, normalized_events),
        "insights": list(insights or []),
        "suggested_questions": list(DEFAULT_SUGGESTED_QUESTIONS),
    }
    if raw_summary is not None:
        context["raw_summary"] = raw_summary
    return context
