from __future__ import annotations

import re
from typing import Any


def _sanitize_key_fragment(value: Any) -> str:
    raw_value = str(value or "none").strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", raw_value)
    sanitized = sanitized.strip("_")
    return sanitized or "none"


def _has_pitch_coordinates(canonical_events: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
        for event in canonical_events
    )


def _has_xg_data(canonical_events: list[dict[str, Any]], metrics: dict[str, Any]) -> bool:
    if any((event.get("xG") or 0) not in {0, 0.0, None} for event in canonical_events):
        return True
    return bool((metrics or {}).get("total_xg"))


def build_ai_coach_state_key(
    prefix: str,
    provider: Any,
    match_id: Any,
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> str:
    return "_".join(
        [
            prefix,
            _sanitize_key_fragment(provider),
            _sanitize_key_fragment(match_id),
            _sanitize_key_fragment(selected_team),
            _sanitize_key_fragment(selected_player),
        ]
    )


def build_provider_capabilities(
    provider: Any,
    canonical_events: list[dict[str, Any]] | None,
    metrics: dict[str, Any] | None,
    raw_payload: Any,
) -> dict[str, bool]:
    normalized_events = canonical_events or []
    normalized_metrics = metrics or {}
    provider_key = str(provider or "").strip().lower()
    has_coordinates = _has_pitch_coordinates(normalized_events)
    has_xg = _has_xg_data(normalized_events, normalized_metrics)

    if provider_key == "statsbomb":
        return {
            "has_event_coordinates": has_coordinates,
            "has_lineups": False,
            "has_team_stats": False,
            "has_player_stats": False,
            "has_xg": has_xg,
            "has_event_timeline": bool(normalized_events),
        }

    if provider_key == "api_football":
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        return {
            "has_event_coordinates": has_coordinates,
            "has_lineups": bool(payload.get("lineups")),
            "has_team_stats": bool(payload.get("statistics")),
            "has_player_stats": bool(payload.get("players")),
            "has_xg": has_xg,
            "has_event_timeline": bool(payload.get("events")) or bool(normalized_events),
        }

    return {
        "has_event_coordinates": has_coordinates,
        "has_lineups": False,
        "has_team_stats": False,
        "has_player_stats": False,
        "has_xg": has_xg,
        "has_event_timeline": bool(normalized_events),
    }


def build_match_metadata_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": result.get("provider"),
        "match_id": result.get("match_id"),
        "competition_name": result.get("competition_name"),
        "season_name": result.get("season_name"),
        "home_team": result.get("home_team"),
        "away_team": result.get("away_team"),
        "match_date": result.get("match_date"),
    }
