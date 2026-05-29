from __future__ import annotations

from typing import Any

from src.services.presentation import build_sportmonks_player_insights
from src.services.providers.provider_capabilities import get_enabled_modules_for_provider
from src.services.providers.sportmonks_adapter import get_sportmonks_match_context

SPORTMONKS_PROVIDER = "sportmonks"
NO_COORDINATES_MESSAGE = (
    "Este partido tiene estadísticas, lineups, xG y eventos principales. "
    "No incluye coordenadas de eventos, por lo que los mapas tácticos espaciales no están disponibles."
)


class SportmonksMatchCenterError(Exception):
    pass


class SportmonksMatchCenterNotFoundError(SportmonksMatchCenterError):
    pass


def _value(item: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def _empty_expected_metrics(team_name: str | None = None) -> dict[str, Any]:
    return {
        "team_name": team_name,
        "xg": None,
        "xgot": None,
        "xpts": None,
        "npxg": None,
        "xg_open_play": None,
        "xg_set_play": None,
        "xg_free_kicks": None,
        "shooting_performance": None,
        "xga": None,
    }


def _empty_derived_metrics() -> dict[str, float | None]:
    return {
        "eficacia_ofensiva": None,
        "rendimiento_definicion": None,
        "amenaza_jugada": None,
        "amenaza_pelota_parada": None,
    }


def _extract_stat_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _stat_label(key: str, value: Any) -> str:
    if isinstance(value, dict) and value.get("label"):
        return str(value.get("label"))
    return key.replace("_", " ").strip().capitalize() or "Métrica"


def _build_match_section(match: Any) -> dict[str, Any]:
    return {
        "match_id": str(_value(match, "provider_match_id", "") or ""),
        "competition": _value(match, "competition_name"),
        "season": _value(match, "season_name"),
        "date": _value(match, "match_date"),
        "status": _value(match, "status"),
        "venue": {
            "name": _value(match, "venue_name"),
            "city": _value(match, "venue_city"),
        },
        "home_team": {
            "id": _value(match, "home_team_id"),
            "name": _value(match, "home_team_name"),
            "score": _value(match, "home_score"),
        },
        "away_team": {
            "id": _value(match, "away_team_id"),
            "name": _value(match, "away_team_name"),
            "score": _value(match, "away_score"),
        },
    }


def _team_side_from_match(match: Any, side: str) -> dict[str, Any]:
    is_home = side == "home"
    return {
        "id": _value(match, "home_team_id" if is_home else "away_team_id"),
        "name": _value(match, "home_team_name" if is_home else "away_team_name"),
        "score": _value(match, "home_score" if is_home else "away_score"),
    }


def _resolve_side(team_id: Any, team_name: Any, match: Any) -> str | None:
    normalized_team_id = str(team_id or "").strip()
    normalized_team_name = str(team_name or "").strip().lower()
    if normalized_team_id and normalized_team_id == str(_value(match, "home_team_id", "") or "").strip():
        return "home"
    if normalized_team_id and normalized_team_id == str(_value(match, "away_team_id", "") or "").strip():
        return "away"
    if normalized_team_name and normalized_team_name == str(_value(match, "home_team_name", "") or "").strip().lower():
        return "home"
    if normalized_team_name and normalized_team_name == str(_value(match, "away_team_name", "") or "").strip().lower():
        return "away"
    return None


def _build_expected_metrics(expected_metrics: list[Any], match: Any) -> dict[str, Any]:
    result = {
        "home": _empty_expected_metrics(_value(match, "home_team_name")),
        "away": _empty_expected_metrics(_value(match, "away_team_name")),
    }
    for item in expected_metrics or []:
        side = _resolve_side(_value(item, "team_id"), _value(item, "team_name"), match)
        if side is None:
            continue
        result[side] = {
            "team_name": _value(item, "team_name") or result[side]["team_name"],
            "xg": _value(item, "xg"),
            "xgot": _value(item, "xgot"),
            "xpts": _value(item, "xpts"),
            "npxg": _value(item, "npxg"),
            "xg_open_play": _value(item, "xg_open_play"),
            "xg_set_play": _value(item, "xg_set_play"),
            "xg_free_kicks": _value(item, "xg_free_kicks"),
            "shooting_performance": _value(item, "shooting_performance"),
            "xga": _value(item, "xga"),
        }
    return result


def _build_timeline(timeline_events: list[Any]) -> list[dict[str, Any]]:
    ordered = sorted(
        timeline_events or [],
        key=lambda item: (
            _value(item, "minute", 0) or 0,
            _value(item, "extra_minute", 0) or 0,
        ),
    )
    return [
        {
            "minute": _value(item, "minute"),
            "extra_minute": _value(item, "extra_minute"),
            "team_name": _value(item, "team_name"),
            "player_name": _value(item, "player_name"),
            "related_player_name": _value(item, "related_player_name"),
            "event_type": _value(item, "event_type"),
            "event_label": _value(item, "event_label"),
            "result": _value(item, "result"),
            "description": _value(item, "description"),
        }
        for item in ordered
    ]


def _format_lineup_players(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": player.get("player_id"),
            "player_name": player.get("player_name"),
            "position": player.get("position"),
            "jersey_number": player.get("jersey_number"),
            "minutes_played": player.get("minutes_played"),
            "rating": player.get("rating"),
            "is_starter": player.get("is_starter"),
        }
        for player in players or []
    ]


def _empty_lineup(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "formation": None,
        "coach": None,
        "starters": [],
        "substitutes": [],
    }


def _build_lineups(lineups: list[Any], match: Any) -> dict[str, Any]:
    result = {
        "home": _empty_lineup(_team_side_from_match(match, "home")),
        "away": _empty_lineup(_team_side_from_match(match, "away")),
    }
    for lineup in lineups or []:
        side = _resolve_side(_value(lineup, "team_id"), _value(lineup, "team_name"), match)
        if side is None:
            continue
        result[side] = {
            "team_id": _value(lineup, "team_id") or result[side]["team_id"],
            "team_name": _value(lineup, "team_name") or result[side]["team_name"],
            "formation": _value(lineup, "formation"),
            "coach": _value(lineup, "coach"),
            "starters": _format_lineup_players(_value(lineup, "starters", []) or []),
            "substitutes": _format_lineup_players(_value(lineup, "substitutes", []) or []),
        }
    return result


def _format_stats(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in (stats or {}).items():
        rows.append(
            {
                "key": str(key),
                "label": _stat_label(str(key), value),
                "value": _extract_stat_value(value),
            }
        )
    return rows


def _empty_team_stats(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "team_id": team.get("id"),
        "team_name": team.get("name"),
        "stats": [],
    }


def _build_team_stats(team_stats: list[Any], match: Any) -> dict[str, Any]:
    result = {
        "home": _empty_team_stats(_team_side_from_match(match, "home")),
        "away": _empty_team_stats(_team_side_from_match(match, "away")),
    }
    for item in team_stats or []:
        side = _resolve_side(_value(item, "team_id"), _value(item, "team_name"), match)
        if side is None:
            continue
        result[side] = {
            "team_id": _value(item, "team_id") or result[side]["team_id"],
            "team_name": _value(item, "team_name") or result[side]["team_name"],
            "stats": _format_stats(_value(item, "stats", {}) or {}),
        }
    return result


def _build_player_stats(player_stats: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in player_stats or []:
        normalized.append(
            {
                "player_id": _value(item, "player_id"),
                "player_name": _value(item, "player_name"),
                "team_id": _value(item, "team_id"),
                "team_name": _value(item, "team_name"),
                "position": _value(item, "position"),
                "jersey_number": _value(item, "jersey_number"),
                "is_starter": _value(item, "is_starter"),
                "minutes_played": _value(item, "minutes_played"),
                "rating": _value(item, "rating"),
                "stats": _format_stats(_value(item, "stats", {}) or {}),
                "insights": build_sportmonks_player_insights(item),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            str(item.get("team_name") or ""),
            str(item.get("player_name") or ""),
        ),
    )


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in {None, 0, 0.0}:
        return None
    return round(float(numerator) / float(denominator), 2)


def _safe_difference(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 2)


def _build_derived_side(team: dict[str, Any], expected: dict[str, Any]) -> dict[str, float | None]:
    score = team.get("score")
    xg = expected.get("xg")
    xg_open_play = expected.get("xg_open_play")
    xg_set_play = expected.get("xg_set_play")
    return {
        "eficacia_ofensiva": _safe_ratio(score, xg),
        "rendimiento_definicion": _safe_difference(score, xg),
        "amenaza_jugada": _safe_ratio(xg_open_play, xg),
        "amenaza_pelota_parada": _safe_ratio(xg_set_play, xg),
    }


def _build_derived_metrics(match_section: dict[str, Any], expected_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "home": _build_derived_side(match_section["home_team"], expected_metrics["home"]),
        "away": _build_derived_side(match_section["away_team"], expected_metrics["away"]),
    }


def _winner_side(match_section: dict[str, Any]) -> str | None:
    home_score = match_section["home_team"].get("score")
    away_score = match_section["away_team"].get("score")
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return None


def _build_result_insight(match_section: dict[str, Any], expected_metrics: dict[str, Any]) -> str | None:
    winner = _winner_side(match_section)
    home_name = str(match_section["home_team"].get("name") or "Local")
    away_name = str(match_section["away_team"].get("name") or "Visitante")
    home_score = match_section["home_team"].get("score")
    away_score = match_section["away_team"].get("score")
    home_xg = expected_metrics["home"].get("xg")
    away_xg = expected_metrics["away"].get("xg")

    if home_score is None or away_score is None:
        return None

    if home_xg is not None and away_xg is not None:
        xg_gap = abs(float(home_xg) - float(away_xg))
        if xg_gap <= 0.35:
            if winner == "home":
                return f"{home_name} ganó {home_score}-{away_score} en un partido equilibrado según goles esperados."
            if winner == "away":
                return f"{away_name} ganó {away_score}-{home_score} en un partido equilibrado según goles esperados."
            return f"{home_name} y {away_name} empataron {home_score}-{away_score} en un partido equilibrado según xG."
        if winner == "home":
            return f"{home_name} ganó {home_score}-{away_score} y también generó más peligro esperado que {away_name}."
        if winner == "away":
            return f"{away_name} ganó {away_score}-{home_score} y también generó más peligro esperado que {home_name}."

    return f"{home_name} y {away_name} terminaron {home_score}-{away_score}."


def _build_xgot_insight(match_section: dict[str, Any], expected_metrics: dict[str, Any]) -> str | None:
    winner = _winner_side(match_section)
    if winner is None:
        return None
    loser = "away" if winner == "home" else "home"
    loser_xgot = expected_metrics[loser].get("xgot")
    winner_xgot = expected_metrics[winner].get("xgot")
    loser_name = str(match_section[f"{loser}_team"].get("name") or "El rival")
    if loser_xgot is None or winner_xgot is None:
        return None
    if float(loser_xgot) <= float(winner_xgot):
        return None
    return (
        f"{loser_name} generó un xGoT superior, pero no logró convertir esa calidad de remate en resultado."
    )


def _build_finishing_insight(match_section: dict[str, Any], expected_metrics: dict[str, Any]) -> str | None:
    home_name = str(match_section["home_team"].get("name") or "Local")
    away_name = str(match_section["away_team"].get("name") or "Visitante")
    home_delta = _safe_difference(match_section["home_team"].get("score"), expected_metrics["home"].get("xg"))
    away_delta = _safe_difference(match_section["away_team"].get("score"), expected_metrics["away"].get("xg"))
    candidates = [
        ("home", home_name, home_delta),
        ("away", away_name, away_delta),
    ]
    strongest = max(candidates, key=lambda item: abs(item[2]) if item[2] is not None else -1)
    side, team_name, delta = strongest
    if delta is None or abs(delta) < 0.5:
        return None
    if delta > 0:
        return f"{team_name} convirtió por encima de lo esperado en relación con su xG."
    return f"{team_name} convirtió por debajo de lo esperado en relación con su xG."


def _build_coordinate_insight(data_quality: dict[str, Any]) -> str | None:
    if data_quality.get("has_event_coordinates"):
        return None
    return "No hay coordenadas de eventos confirmadas, por lo que no corresponden mapas tácticos espaciales."


def _build_insights(
    match_section: dict[str, Any],
    expected_metrics: dict[str, Any],
    data_quality: dict[str, Any],
) -> list[str]:
    insights: list[str] = []
    for builder in (
        _build_result_insight,
        _build_xgot_insight,
        _build_finishing_insight,
    ):
        insight = builder(match_section, expected_metrics)
        if insight and insight not in insights:
            insights.append(insight)
    coordinate_insight = _build_coordinate_insight(data_quality)
    if coordinate_insight and coordinate_insight not in insights:
        insights.append(coordinate_insight)
    return insights


def _build_enabled_modules(has_coordinates: bool, availability: Any) -> dict[str, bool]:
    base_modules = get_enabled_modules_for_provider(SPORTMONKS_PROVIDER)
    timeline_enabled = bool(_value(availability, "has_event_timeline", False))
    lineups_enabled = bool(_value(availability, "has_lineups", False))
    team_stats_enabled = bool(_value(availability, "has_team_stats", False))
    player_stats_enabled = bool(_value(availability, "has_player_stats", False))
    expected_metrics_enabled = any(
        bool(_value(availability, field_name, False))
        for field_name in ("has_xg", "has_xgot", "has_xpts")
    )
    spatial_enabled = bool(has_coordinates)
    return {
        "match_center": True,
        "expected_metrics": expected_metrics_enabled,
        "timeline": timeline_enabled,
        "lineups": lineups_enabled,
        "team_stats": team_stats_enabled,
        "player_stats": player_stats_enabled,
        "event_maps": base_modules.get("event_maps", False) and spatial_enabled,
        "shot_map": base_modules.get("shot_map", False) and spatial_enabled,
        "pass_network": spatial_enabled,
    }


def _build_quality_level(availability: Any) -> str:
    has_context = bool(
        _value(availability, "has_event_timeline", False)
        or _value(availability, "has_lineups", False)
        or _value(availability, "has_team_stats", False)
        or _value(availability, "has_player_stats", False)
    )
    has_expected = any(
        bool(_value(availability, field_name, False))
        for field_name in ("has_xg", "has_xgot", "has_xpts")
    )
    has_coordinates = bool(_value(availability, "has_coordinates", False))
    if has_context and has_expected and has_coordinates:
        return "alta"
    if has_context and has_expected:
        return "media"
    if has_context:
        return "media"
    return "baja"


def _build_data_quality(availability: Any) -> dict[str, Any]:
    has_coordinates = bool(_value(availability, "has_coordinates", False))
    enabled_modules = _build_enabled_modules(has_coordinates, availability)
    level = _build_quality_level(availability)
    message = NO_COORDINATES_MESSAGE if not has_coordinates else "Este partido tiene datos suficientes para Match Center y visualizaciones compatibles."
    return {
        "level": level,
        "has_xg": bool(_value(availability, "has_xg", False)),
        "has_xgot": bool(_value(availability, "has_xgot", False)),
        "has_xpts": bool(_value(availability, "has_xpts", False)),
        "has_lineups": bool(_value(availability, "has_lineups", False)),
        "has_player_stats": bool(_value(availability, "has_player_stats", False)),
        "has_team_stats": bool(_value(availability, "has_team_stats", False)),
        "has_event_timeline": bool(_value(availability, "has_event_timeline", False)),
        "has_event_coordinates": has_coordinates,
        "enabled_modules": enabled_modules,
        "message": message,
    }


def build_sportmonks_match_center(match_id: str) -> dict[str, Any]:
    normalized_match_id = str(match_id or "").strip()
    if not normalized_match_id:
        raise SportmonksMatchCenterError("`match_id` es obligatorio.")

    context_response = get_sportmonks_match_context(normalized_match_id)
    if not context_response.get("ok"):
        error = str(context_response.get("error") or "No se pudo cargar el Match Center desde Sportmonks.")
        raise SportmonksMatchCenterError(f"No se pudo cargar información desde Sportmonks: {error}")

    data = context_response.get("data") or {}
    match = data.get("match")
    if match is None:
        raise SportmonksMatchCenterNotFoundError(
            f"No se encontró el partido {normalized_match_id} en Sportmonks."
        )

    match_section = _build_match_section(match)
    expected_metrics = _build_expected_metrics(data.get("expected_metrics") or [], match)
    timeline = _build_timeline(data.get("timeline_events") or [])
    lineups = _build_lineups(data.get("lineups") or [], match)
    team_stats = _build_team_stats(data.get("team_stats") or [], match)
    player_stats = _build_player_stats(data.get("player_stats") or [])
    derived_metrics = _build_derived_metrics(match_section, expected_metrics)
    data_quality = _build_data_quality(data.get("availability"))
    insights = _build_insights(match_section, expected_metrics, data_quality)

    return {
        "provider": SPORTMONKS_PROVIDER,
        "match": match_section,
        "expected_metrics": expected_metrics,
        "timeline": timeline,
        "lineups": lineups,
        "team_stats": team_stats,
        "player_stats": player_stats,
        "derived_metrics": derived_metrics,
        "insights": insights,
        "data_quality": data_quality,
    }
