from __future__ import annotations

from typing import Any


def normalize_api_football_events_to_canonical(
    raw_events: list[dict[str, Any]],
    fixture_id: int | str | None = None,
) -> list[dict[str, Any]]:
    canonical_events: list[dict[str, Any]] = []
    resolved_match_id = str(fixture_id if fixture_id is not None else "unknown-match")

    for index, event in enumerate(raw_events):
        team_data = event.get("team", {}) or {}
        player_data = event.get("player", {}) or {}
        time_data = event.get("time", {}) or {}
        event_type = str(event.get("type") or event.get("detail") or "Unknown")
        detail = event.get("detail")
        elapsed = time_data.get("elapsed", 0)

        canonical_events.append(
            {
                "event_id": f"{resolved_match_id}-{index}-{event_type}-{elapsed}",
                "match_id": resolved_match_id,
                "team_id": str(team_data.get("id", "unknown-team")),
                "team_name": str(team_data.get("name", "Equipo desconocido")),
                "player_id": str(player_data.get("id", "unknown-player")),
                "player_name": str(player_data.get("name", "Jugador desconocido")),
                "minute": int(elapsed or 0),
                "second": 0,
                "event_type": event_type,
                "x": None,
                "y": None,
                "end_x": None,
                "end_y": None,
                "outcome": str(detail) if detail is not None else None,
                "progressive": False,
                "under_pressure": False,
                "xG": 0.0,
                "xA": 0.0,
            }
        )

    return canonical_events
