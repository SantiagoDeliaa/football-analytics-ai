from __future__ import annotations

from typing import Any


def _object_value(item: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def _render_stat_value(value: Any) -> str:
    if value is None or value == "":
        return "No disponible"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _extract_stat_value(entry: Any) -> Any:
    if isinstance(entry, dict) and "value" in entry:
        return entry.get("value")
    return entry


def format_sportmonks_match_title(match: Any) -> str:
    home_team = str(_object_value(match, "home_team_name", "") or "")
    away_team = str(_object_value(match, "away_team_name", "") or "")
    home_score = _object_value(match, "home_score")
    away_score = _object_value(match, "away_score")
    if home_score is not None and away_score is not None:
        return f"{home_team} {home_score} - {away_score} {away_team}".strip()
    return f"{home_team} vs {away_team}".strip()


def format_availability_status(value: bool) -> str:
    return "Disponible" if bool(value) else "No disponible"


def build_expected_metrics_table(expected_metrics: list[Any]) -> list[dict[str, Any]]:
    metric_definitions = [
        ("xg", "Goles esperados (xG)"),
        ("xgot", "Goles esperados al arco (xGoT)"),
        ("xpts", "Puntos esperados (xPTS)"),
        ("npxg", "xG sin penales"),
        ("xg_open_play", "xG en jugada"),
        ("xg_set_play", "xG pelota parada"),
        ("xg_free_kicks", "xG tiros libres"),
        ("shooting_performance", "Rendimiento de remate"),
        ("xga", "xG concedido"),
    ]
    teams = [item for item in expected_metrics if item is not None]
    team_names = [str(_object_value(item, "team_name", "") or f"Equipo {index + 1}") for index, item in enumerate(teams)]
    rows: list[dict[str, Any]] = []
    for metric_key, label in metric_definitions:
        row: dict[str, Any] = {"Métrica": label}
        for index, team_metric in enumerate(teams):
            team_name = team_names[index]
            row[team_name] = _render_stat_value(_object_value(team_metric, metric_key))
        rows.append(row)
    return rows


def build_timeline_table(timeline_events: list[Any], event_filter: str = "Todos") -> list[dict[str, Any]]:
    filter_map = {
        "Todos": None,
        "Goles": {"goal", "penalty", "own_goal"},
        "Tarjetas": {"yellowcard", "redcard"},
        "Cambios": {"substitution"},
        "VAR": {"var"},
    }
    allowed_types = filter_map.get(event_filter)
    sorted_events = sorted(
        timeline_events or [],
        key=lambda item: (
            _object_value(item, "minute", 0) or 0,
            _object_value(item, "extra_minute", 0) or 0,
        ),
    )
    rows: list[dict[str, Any]] = []
    for event in sorted_events:
        event_type = str(_object_value(event, "event_type", "unknown") or "unknown")
        if allowed_types is not None and event_type not in allowed_types:
            continue
        minute = _object_value(event, "minute")
        extra_minute = _object_value(event, "extra_minute")
        minute_label = str(minute if minute is not None else "-")
        if extra_minute not in {None, 0, "0"}:
            minute_label = f"{minute_label}+{extra_minute}"
        rows.append(
            {
                "Minuto": minute_label,
                "Evento": str(_object_value(event, "event_label", "Evento") or "Evento"),
                "Equipo": str(_object_value(event, "team_name", "") or "No disponible"),
                "Jugador": str(_object_value(event, "player_name", "") or "No disponible"),
                "Relacionado": str(_object_value(event, "related_player_name", "") or "No disponible"),
                "Resultado": str(_object_value(event, "result", "") or "No disponible"),
                "Descripción": str(_object_value(event, "description", "") or "No disponible"),
            }
        )
    return rows


def build_player_stats_table(player_stats: Any) -> list[dict[str, Any]]:
    stats = _object_value(player_stats, "stats", {}) or {}
    rows: list[dict[str, Any]] = []
    for key, value in stats.items():
        label = str(value.get("label") or key) if isinstance(value, dict) else key.replace("_", " ").title()
        rows.append(
            {
                "Métrica": label,
                "Valor": _render_stat_value(_extract_stat_value(value)),
            }
        )
    return rows


def build_team_stats_table(team_stats: Any) -> list[dict[str, Any]]:
    stats = _object_value(team_stats, "stats", {}) or {}
    rows: list[dict[str, Any]] = []
    for key, value in stats.items():
        label = str(value.get("label") or key) if isinstance(value, dict) else key.replace("_", " ").title()
        rows.append({"Métrica": label, "Valor": _render_stat_value(_extract_stat_value(value))})
    return rows


def build_sportmonks_player_insights(player_stats: Any) -> list[str]:
    stats = _object_value(player_stats, "stats", {}) or {}
    insights: list[str] = []
    passes = _extract_stat_value(stats.get("passes"))
    pass_accuracy = _extract_stat_value(stats.get("accurate_passes_percentage"))
    duels_won = _extract_stat_value(stats.get("duels_won"))
    duels_total = _extract_stat_value(stats.get("total_duels"))
    minutes = _object_value(player_stats, "minutes_played")

    if passes not in {None, ""}:
        insights.append(f"El jugador registro {passes} pases.")
    if pass_accuracy not in {None, ""}:
        insights.append(f"El jugador tuvo una precision de pase de {pass_accuracy}%.")
    if duels_won not in {None, ""} and duels_total not in {None, ""}:
        insights.append(f"El jugador gano {duels_won} de {duels_total} duelos.")
    if minutes not in {None, ""}:
        insights.append(f"El jugador participo {minutes} minutos.")
    return insights
