from __future__ import annotations

from copy import deepcopy
from typing import Any

from .sportmonks_client import (
    get_sportmonks_fixture_full_context as client_get_sportmonks_fixture_full_context,
    get_sportmonks_fixtures_by_date as client_get_sportmonks_fixtures_by_date,
    get_sportmonks_leagues as client_get_sportmonks_leagues,
)
from .sportmonks_normalizer import (
    SPORTMONKS_PROVIDER,
    build_sportmonks_event_availability,
    normalize_sportmonks_fixture,
    normalize_sportmonks_full_context,
    normalize_sportmonks_league,
)


def _build_adapter_response(
    ok: bool,
    data: Any,
    error: str | None = None,
    raw_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "data": data,
        "error": error,
        "source": SPORTMONKS_PROVIDER,
        "raw_meta": deepcopy(raw_meta or {}),
    }


def get_available_sportmonks_competitions(params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client_get_sportmonks_leagues(params=params)
    if not response.get("ok"):
        return _build_adapter_response(
            ok=False,
            data=[],
            error=str(response.get("error") or "No se pudieron obtener competiciones desde Sportmonks."),
            raw_meta=response.get("meta"),
        )

    competitions = [
        normalize_sportmonks_league(item)
        for item in (response.get("data") or [])
        if isinstance(item, dict)
    ]
    return _build_adapter_response(
        ok=True,
        data=competitions,
        error=None,
        raw_meta=response.get("meta"),
    )


def get_sportmonks_fixtures_by_date(date: str, include: str | None = None) -> dict[str, Any]:
    response = client_get_sportmonks_fixtures_by_date(date, include=include)
    if not response.get("ok"):
        return _build_adapter_response(
            ok=False,
            data=[],
            error=str(response.get("error") or "No se pudieron obtener fixtures desde Sportmonks."),
            raw_meta=response.get("meta"),
        )

    fixtures = [
        normalize_sportmonks_fixture(item)
        for item in (response.get("data") or [])
        if isinstance(item, dict)
    ]
    return _build_adapter_response(
        ok=True,
        data=fixtures,
        error=None,
        raw_meta=response.get("meta"),
    )


def get_sportmonks_match_context(fixture_id: str) -> dict[str, Any]:
    response = client_get_sportmonks_fixture_full_context(fixture_id)
    if not response.get("ok"):
        return _build_adapter_response(
            ok=False,
            data={},
            error=str(response.get("error") or "No se pudo obtener el contexto del partido desde Sportmonks."),
            raw_meta=response.get("meta"),
        )

    normalized = normalize_sportmonks_full_context(response)
    return _build_adapter_response(
        ok=True,
        data={
            "match": normalized.get("match"),
            "lineups": normalized.get("lineups", []),
            "timeline_events": normalized.get("timeline_events", []),
            "expected_metrics": normalized.get("expected_metrics", []),
            "team_stats": normalized.get("team_stats", []),
            "player_stats": normalized.get("player_stats", []),
            "availability": normalized.get("availability"),
        },
        error=None,
        raw_meta=response.get("meta"),
    )


def get_sportmonks_data_availability_for_fixture(fixture_id: str) -> dict[str, Any]:
    context_response = get_sportmonks_match_context(fixture_id)
    if not context_response.get("ok"):
        return _build_adapter_response(
            ok=False,
            data=None,
            error=str(context_response.get("error") or "No se pudo obtener la disponibilidad de datos desde Sportmonks."),
            raw_meta=context_response.get("raw_meta"),
        )

    data = context_response.get("data") or {}
    availability = data.get("availability")
    if availability is None:
        availability = build_sportmonks_event_availability(
            match_id=fixture_id,
            timeline_events=data.get("timeline_events"),
            lineups=data.get("lineups"),
            team_stats=data.get("team_stats"),
            player_stats=data.get("player_stats"),
            expected_metrics=data.get("expected_metrics"),
        )

    return _build_adapter_response(
        ok=True,
        data=availability,
        error=None,
        raw_meta=context_response.get("raw_meta"),
    )
