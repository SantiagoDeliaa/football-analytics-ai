from __future__ import annotations

from typing import Any


def _get_location(point: list[Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(point, list) or len(point) < 2:
        return None, None
    try:
        return float(point[0]), float(point[1])
    except Exception:
        return None, None


def _extract_outcome(event_type: str, event: dict[str, Any]) -> str | None:
    if event_type == "Pass":
        pass_data = event.get("pass", {}) or {}
        outcome = (pass_data.get("outcome") or {}).get("name")
        return str(outcome) if outcome else "Complete"
    if event_type == "Shot":
        shot_data = event.get("shot", {}) or {}
        outcome = (shot_data.get("outcome") or {}).get("name")
        return str(outcome) if outcome else None
    if event_type == "Duel":
        duel_data = event.get("duel", {}) or {}
        outcome = (duel_data.get("outcome") or {}).get("name")
        return str(outcome) if outcome else None
    return None


def normalize_events_to_canonical(
    raw_events: list[dict[str, Any]],
    match_id: int | str | None = None,
) -> list[dict[str, Any]]:
    canonical_events: list[dict[str, Any]] = []

    for idx, event in enumerate(raw_events):
        event_type = str(event.get("type", {}).get("name", "Unknown") or "Unknown")
        team_data = event.get("team", {}) or {}
        player_data = event.get("player", {}) or {}
        pass_data = event.get("pass", {}) or {}
        shot_data = event.get("shot", {}) or {}
        carry_data = event.get("carry", {}) or {}

        x, y = _get_location(event.get("location"))
        if event_type == "Pass":
            end_x, end_y = _get_location(pass_data.get("end_location"))
        elif event_type == "Carry":
            end_x, end_y = _get_location(carry_data.get("end_location"))
        else:
            end_x, end_y = None, None

        outcome = _extract_outcome(event_type, event)

        progressive = bool(
            isinstance(x, (int, float))
            and isinstance(end_x, (int, float))
            and (float(end_x) - float(x) >= 15.0)
        )
        under_pressure = bool(event.get("under_pressure", False))
        xg_value = shot_data.get("statsbomb_xg", 0.0)
        xa_value = pass_data.get("xA", 0.0) if isinstance(pass_data.get("xA"), (int, float)) else 0.0
        resolved_match_id = event.get("match_id", match_id if match_id is not None else "unknown-match")

        canonical_events.append(
            {
                "event_id": str(event.get("id", f"mock-event-{idx + 1}")),
                "match_id": str(resolved_match_id),
                "team_id": str(team_data.get("id", "unknown-team")),
                "team_name": str(team_data.get("name", "Equipo desconocido")),
                "player_id": str(player_data.get("id", "unknown-player")),
                "player_name": str(player_data.get("name", "Jugador desconocido")),
                "minute": int(event.get("minute", 0) or 0),
                "second": int(event.get("second", 0) or 0),
                "event_type": event_type,
                "x": x,
                "y": y,
                "end_x": end_x,
                "end_y": end_y,
                "outcome": outcome,
                "progressive": progressive,
                "under_pressure": under_pressure,
                "xG": float(xg_value or 0.0),
                "xA": float(xa_value or 0.0),
            }
        )

    return canonical_events
