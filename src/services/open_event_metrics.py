from __future__ import annotations

from typing import Any


IGNORED_EVENT_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Substitution",
    "Tactical Shift",
    "Bad Behaviour",
}


def filter_analytical_events(canonical_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for event in canonical_events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type in IGNORED_EVENT_TYPES:
            continue
        filtered.append(event)
    return filtered


def _is_favorable_duel(outcome: str | None) -> bool:
    if not outcome:
        return False
    normalized = outcome.strip().lower()
    return any(token in normalized for token in ("won", "success", "success in play", "tackle"))


def safe_percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(100.0, (numerator / denominator) * 100.0))


def get_metric_label(score: float | None) -> str:
    if score is None:
        return "No aplica"
    if score < 40:
        return "Bajo"
    if score < 70:
        return "Medio"
    return "Alto"


def calculate_field_tilt(
    all_events: list[dict[str, Any]],
    selected_team: str | None,
) -> float | None:
    if not selected_team:
        return None
    total_final_third_all = sum(
        1
        for event in all_events
        if isinstance(event.get("x"), (int, float)) and float(event["x"]) >= 80.0
    )
    team_final_third = sum(
        1
        for event in all_events
        if event.get("team_name") == selected_team
        and isinstance(event.get("x"), (int, float))
        and float(event["x"]) >= 80.0
    )
    return round(safe_percentage(float(team_final_third), float(total_final_third_all)), 1)


def calculate_directness(progressive_actions: int, total_passes: int) -> float:
    return round(safe_percentage(float(progressive_actions), float(total_passes)), 1)


def calculate_progressive_threat(
    total_events: int,
    progressive_actions: int,
    final_third_actions: int,
    total_shots: int,
    total_xg: float,
) -> float:
    # Heurística inicial:
    # pondera progresiones, presencia en último tercio, volumen de remate y calidad (xG).
    # Se normaliza por volumen para mantener escala 0-100 estable entre partidos.
    raw_points = (
        progressive_actions * 2.0
        + final_third_actions * 1.5
        + total_shots * 3.0
        + total_xg * 25.0
    )
    max_points = max(1.0, total_events * 3.5)
    return round(min(100.0, (raw_points / max_points) * 100.0), 1)


def calculate_recovery_height(recovery_events: list[dict[str, Any]]) -> float | None:
    x_values = [
        float(event["x"])
        for event in recovery_events
        if isinstance(event.get("x"), (int, float))
    ]
    if not x_values:
        return None
    avg_x = sum(x_values) / len(x_values)
    return round(safe_percentage(avg_x, 120.0), 1)


def calculate_shot_quality(total_xg: float, total_shots: int) -> float:
    if total_shots <= 0:
        return 0.0
    return round((float(total_xg) / float(total_shots)) * 100.0, 1)


def calculate_player_influence(
    selected_player: str | None,
    total_events: int,
    total_passes: int,
    progressive_actions: int,
    final_third_actions: int,
    total_shots: int,
    recoveries: int,
) -> float | None:
    if not selected_player:
        return None
    activity = min(1.0, total_events / 25.0)
    creation = min(1.0, (total_passes + progressive_actions * 2.0) / 30.0)
    threat = min(1.0, (total_shots * 3.0 + final_third_actions) / 20.0)
    defensive = min(1.0, recoveries / 8.0)
    score = (0.35 * activity + 0.25 * creation + 0.25 * threat + 0.15 * defensive) * 100.0
    return round(min(100.0, max(0.0, score)), 1)


def calculate_open_event_metrics(
    canonical_events: list[dict[str, Any]],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> dict[str, int | float]:
    analytical_events = filter_analytical_events(canonical_events)
    filtered_events: list[dict[str, Any]] = []
    for event in analytical_events:
        if selected_team and event.get("team_name") != selected_team:
            continue
        if selected_player and event.get("player_name") != selected_player:
            continue
        filtered_events.append(event)

    total_events = len(filtered_events)
    total_passes = sum(1 for event in filtered_events if event.get("event_type") == "Pass")
    total_shots = sum(1 for event in filtered_events if event.get("event_type") == "Shot")
    total_carries = sum(1 for event in filtered_events if event.get("event_type") == "Carry")
    progressive_actions = sum(1 for event in filtered_events if bool(event.get("progressive", False)))
    total_under_pressure = sum(1 for event in filtered_events if bool(event.get("under_pressure", False)))
    total_xg = sum(float(event.get("xG", 0.0) or 0.0) for event in filtered_events if event.get("event_type") == "Shot")
    final_third_actions = sum(
        1
        for event in filtered_events
        if isinstance(event.get("x"), (int, float)) and float(event["x"]) >= 80.0
    )
    recovery_events: list[dict[str, Any]] = []
    for event in filtered_events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type in {"Ball Recovery", "Interception"}:
            recovery_events.append(event)
            continue
        if event_type == "Duel" and _is_favorable_duel(event.get("outcome")):
            recovery_events.append(event)
    recoveries = len(recovery_events)

    field_tilt_index = calculate_field_tilt(analytical_events, selected_team)
    directness_index = calculate_directness(progressive_actions, total_passes)
    progressive_threat_index = calculate_progressive_threat(
        total_events=total_events,
        progressive_actions=progressive_actions,
        final_third_actions=final_third_actions,
        total_shots=total_shots,
        total_xg=total_xg,
    )
    recovery_height_index = calculate_recovery_height(recovery_events)
    shot_quality_index = calculate_shot_quality(total_xg, total_shots)
    player_influence_score = calculate_player_influence(
        selected_player=selected_player,
        total_events=total_events,
        total_passes=total_passes,
        progressive_actions=progressive_actions,
        final_third_actions=final_third_actions,
        total_shots=total_shots,
        recoveries=recoveries,
    )

    return {
        "total_events": total_events,
        "total_passes": total_passes,
        "total_shots": total_shots,
        "total_carries": total_carries,
        "total_under_pressure": total_under_pressure,
        "total_xg": round(total_xg, 3),
        "progressive_actions": progressive_actions,
        "final_third_actions": final_third_actions,
        "recoveries": recoveries,
        "field_tilt_index": field_tilt_index,
        "field_tilt_label": get_metric_label(field_tilt_index),
        "directness_index": directness_index,
        "directness_label": get_metric_label(directness_index),
        "progressive_threat_index": progressive_threat_index,
        "progressive_threat_label": get_metric_label(progressive_threat_index),
        "recovery_height_index": recovery_height_index,
        "recovery_height_label": get_metric_label(recovery_height_index),
        "shot_quality_index": shot_quality_index,
        "shot_quality_label": get_metric_label(shot_quality_index),
        "player_influence_score": player_influence_score,
        "player_influence_label": get_metric_label(player_influence_score),
    }
