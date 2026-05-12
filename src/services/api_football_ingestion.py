from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_API_FOOTBALL_URL = "https://v3.football.api-sports.io"
DEFAULT_TIMEOUT_SECONDS = 8.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_LOADED = False
_LAST_API_FOOTBALL_STATUS: dict[str, Any] = {
    "source": "idle",
    "status": "idle",
    "message": "",
    "errors": [],
    "requests_remaining": None,
}


def _load_local_env_file(force_reload: bool = False) -> None:
    global _ENV_LOADED
    if _ENV_LOADED and not force_reload:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            # Si .env está malformado o no puede leerse, no rompemos la app.
            pass
    _ENV_LOADED = True


def get_api_football_api_key() -> str | None:
    api_key = str(os.getenv("API_FOOTBALL_KEY", "") or "").strip()
    if not api_key:
        _load_local_env_file()
        api_key = str(os.getenv("API_FOOTBALL_KEY", "") or "").strip()
    if not api_key:
        _load_local_env_file(force_reload=True)
        api_key = str(os.getenv("API_FOOTBALL_KEY", "") or "").strip()
    return api_key or None


def get_api_football_config_status() -> dict[str, Any]:
    api_key = get_api_football_api_key()
    return {
        "configured": bool(api_key),
        "message": (
            "API-Football configurado correctamente."
            if api_key
            else "Falta configurar API_FOOTBALL_KEY en el entorno."
        ),
    }


def _set_status(
    source: str,
    status: str,
    message: str,
    errors: list[str] | None = None,
    requests_remaining: str | None = None,
) -> None:
    _LAST_API_FOOTBALL_STATUS.update(
        {
            "source": source,
            "status": status,
            "message": message,
            "errors": list(errors or []),
            "requests_remaining": requests_remaining,
        }
    )


def get_api_football_status() -> dict[str, Any]:
    return deepcopy(_LAST_API_FOOTBALL_STATUS)


def get_api_football_user_message(status: dict[str, Any] | None = None) -> str:
    resolved_status = status or get_api_football_status()
    errors = [str(item) for item in resolved_status.get("errors", []) or [] if item]
    lowered_errors = [item.lower() for item in errors]
    if any("this season" in item and "free plans" in item for item in lowered_errors):
        return "Tu plan actual no tiene acceso a la temporada seleccionada. Probá una temporada habilitada por tu plan."
    if any("last parameter" in item and "free plans" in item for item in lowered_errors):
        return "Tu plan actual no tiene acceso al filtro de ultimos partidos. Probá buscar la temporada sin ese filtro."
    if errors:
        return f"{resolved_status.get('message', 'API-Football devolvió un error.')} Detalle: {errors[0]}"
    return str(resolved_status.get("message", "") or "")


def api_football_request(endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    api_key = get_api_football_api_key()
    if not api_key:
        _set_status(
            source="config",
            status="missing_key",
            message="Falta configurar API_FOOTBALL_KEY en el entorno.",
            errors=["missing_api_key"],
        )
        return []

    endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    url = f"{BASE_API_FOOTBALL_URL}{endpoint_path}"
    if query:
        url = f"{url}?{query}"

    request = Request(
        url,
        headers={
            "User-Agent": "football-analytics-ai/1.0",
            "x-apisports-key": api_key,
        },
    )

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
            requests_remaining = (
                response.headers.get("x-ratelimit-requests-remaining")
                or response.headers.get("x-ratelimit-remaining")
                or response.headers.get("X-RateLimit-Requests-Remaining")
            )
        parsed = json.loads(payload)
        errors_payload = parsed.get("errors", [])
        error_list = []
        if isinstance(errors_payload, dict):
            error_list = [f"{key}: {value}" for key, value in errors_payload.items() if value]
        elif isinstance(errors_payload, list):
            error_list = [str(item) for item in errors_payload if item]

        if error_list:
            _set_status(
                source="api_football",
                status="error",
                message="API-Football devolvió errores en la respuesta.",
                errors=error_list,
                requests_remaining=requests_remaining,
            )
            return []

        response_payload = parsed.get("response", [])
        if not isinstance(response_payload, list):
            _set_status(
                source="api_football",
                status="error",
                message="API-Football devolvió una respuesta con formato inesperado.",
                errors=["invalid_response_shape"],
                requests_remaining=requests_remaining,
            )
            return []

        _set_status(
            source="api_football",
            status="real",
            message="Datos cargados correctamente desde API-Football.",
            errors=[],
            requests_remaining=requests_remaining,
        )
        return response_payload
    except HTTPError as exc:
        _set_status(
            source="api_football",
            status="error",
            message=f"Error HTTP consultando API-Football: {exc}",
            errors=[str(exc)],
        )
    except URLError as exc:
        _set_status(
            source="api_football",
            status="error",
            message=f"Error de red consultando API-Football: {exc}",
            errors=[str(exc)],
        )
    except TimeoutError as exc:
        _set_status(
            source="api_football",
            status="error",
            message=f"Timeout consultando API-Football: {exc}",
            errors=[str(exc)],
        )
    except json.JSONDecodeError as exc:
        _set_status(
            source="api_football",
            status="error",
            message=f"Respuesta JSON inválida desde API-Football: {exc}",
            errors=[str(exc)],
        )
    except Exception as exc:
        _set_status(
            source="api_football",
            status="error",
            message=f"Fallo inesperado consultando API-Football: {exc}",
            errors=[str(exc)],
        )
    return []


def get_api_football_countries() -> list[dict[str, Any]]:
    countries = api_football_request("/countries")
    normalized: list[dict[str, Any]] = []
    for item in countries:
        name = str(item.get("name", "") or "")
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "code": str(item.get("code", "") or ""),
                "flag": str(item.get("flag", "") or ""),
                "display_name": name,
            }
        )
    normalized.sort(key=lambda item: (item["name"] != "Argentina", item["name"]))
    return normalized


def get_api_football_leagues(
    country: str | None = None,
    season: int | str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    response = api_football_request(
        "/leagues",
        params={"country": country, "season": season, "search": search},
    )
    leagues: list[dict[str, Any]] = []
    for item in response:
        league_info = item.get("league", {}) or {}
        country_info = item.get("country", {}) or {}
        seasons = item.get("seasons", []) or []
        league_name = str(league_info.get("name", "Liga"))
        country_name = str(country_info.get("name", country or ""))
        season_years = [
            int(season_item.get("year"))
            for season_item in seasons
            if isinstance(season_item.get("year"), int)
        ]
        current_season = next(
            (season_item.get("year") for season_item in seasons if season_item.get("current") is True),
            season_years[0] if season_years else None,
        )
        leagues.append(
            {
                "league_id": league_info.get("id"),
                "league_name": league_name,
                "country_name": country_name,
                "type": str(league_info.get("type", "") or ""),
                "logo": str(league_info.get("logo", "") or ""),
                "seasons": seasons,
                "current_season": current_season,
                "display_name": f"{league_name} ({country_name})".strip(),
            }
        )
    leagues.sort(key=lambda item: (item["country_name"] != "Argentina", item["league_name"]))
    return leagues


def get_api_football_fixtures(
    league_id: int | str,
    season: int | str,
    last: int | None = None,
    next: int | None = None,
    date: str | None = None,
) -> list[dict[str, Any]]:
    params = {
        "league": league_id,
        "season": season,
        "last": last,
        "next": next,
        "date": date,
    }
    response = api_football_request("/fixtures", params=params)
    fixtures: list[dict[str, Any]] = []
    for item in response:
        fixture = item.get("fixture", {}) or {}
        league = item.get("league", {}) or {}
        teams = item.get("teams", {}) or {}
        home = teams.get("home", {}) or {}
        away = teams.get("away", {}) or {}
        fixture_id = fixture.get("id")
        match_date = str(fixture.get("date", "") or "")[:10]
        home_name = str(home.get("name", "Local"))
        away_name = str(away.get("name", "Visitante"))
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "match_id": fixture_id,
                "home_team": home_name,
                "away_team": away_name,
                "match_date": match_date,
                "competition": str(league.get("name", "Liga")),
                "season": str(league.get("season", season)),
                "status": str((fixture.get("status") or {}).get("long", "") or ""),
                "display_name": f"{home_name} vs {away_name} — {match_date}",
            }
        )
    return fixtures


def get_api_football_fixture_events(fixture_id: int | str) -> list[dict[str, Any]]:
    return api_football_request("/fixtures/events", params={"fixture": fixture_id})


def get_api_football_fixture_lineups(fixture_id: int | str) -> list[dict[str, Any]]:
    return api_football_request("/fixtures/lineups", params={"fixture": fixture_id})


def get_api_football_fixture_statistics(fixture_id: int | str) -> list[dict[str, Any]]:
    return api_football_request("/fixtures/statistics", params={"fixture": fixture_id})


def get_api_football_fixture_players(fixture_id: int | str) -> list[dict[str, Any]]:
    return api_football_request("/fixtures/players", params={"fixture": fixture_id})
