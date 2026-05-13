from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_OPEN_DATA_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_TIMEOUT_SECONDS = 3.0

_LAST_INGESTION_STATUS: dict[str, dict[str, str]] = {
    "competitions": {"source": "real", "message": ""},
    "matches": {"source": "real", "message": ""},
    "events": {"source": "real", "message": ""},
}
_URL_CACHE: dict[str, dict[str, Any]] = {}
_URL_CACHE_TTL_SECONDS = 300.0
_FAILURE_COOLDOWN_SECONDS = 60.0

_MOCK_COMPETITIONS: list[dict[str, Any]] = [
    {
        "competition_id": 43,
        "season_id": 106,
        "competition_name": "FIFA World Cup",
        "season_name": "2022",
        "country_name": "International",
        "display_name": "FIFA World Cup - 2022",
    },
    {
        "competition_id": 55,
        "season_id": 282,
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "country_name": "International",
        "display_name": "UEFA Euro - 2020",
    },
]

_MOCK_MATCHES: dict[tuple[int, int], list[dict[str, Any]]] = {
    (43, 106): [
        {
            "match_id": 3869685,
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
            "competition": "FIFA World Cup",
            "season": "2022",
            "display_name": "Argentina vs Francia — 2022-12-18",
        }
    ],
    (55, 282): [
        {
            "match_id": 3795220,
            "home_team": "Italia",
            "away_team": "Inglaterra",
            "match_date": "2021-07-11",
            "competition": "UEFA Euro",
            "season": "2020",
            "display_name": "Italia vs Inglaterra — 2021-07-11",
        }
    ],
}

_MOCK_EVENTS: dict[int, list[dict[str, Any]]] = {
    3869685: [
        {
            "id": "e-1",
            "match_id": 3869685,
            "minute": 3,
            "second": 14,
            "type": {"name": "Pass"},
            "team": {"id": 779, "name": "Argentina"},
            "player": {"id": 5503, "name": "Lionel Messi"},
            "location": [48.2, 34.1],
            "pass": {"end_location": [71.0, 26.5], "outcome": {"name": "Complete"}},
        },
        {
            "id": "e-2",
            "match_id": 3869685,
            "minute": 3,
            "second": 36,
            "type": {"name": "Shot"},
            "team": {"id": 779, "name": "Argentina"},
            "player": {"id": 5503, "name": "Lionel Messi"},
            "location": [102.0, 34.0],
            "shot": {"outcome": {"name": "On Target"}, "statsbomb_xg": 0.24},
        },
        {
            "id": "e-3",
            "match_id": 3869685,
            "minute": 7,
            "second": 2,
            "type": {"name": "Ball Recovery"},
            "team": {"id": 771, "name": "Francia"},
            "player": {"id": 3097, "name": "Antoine Griezmann"},
            "location": [42.3, 50.1],
        },
    ]
}


def _set_status(scope: str, source: str, message: str = "") -> None:
    _LAST_INGESTION_STATUS[scope] = {"source": source, "message": message}


def get_ingestion_status() -> dict[str, dict[str, str]]:
    return deepcopy(_LAST_INGESTION_STATUS)


def _load_remote_json(url: str) -> Any:
    now = time.time()
    cached = _URL_CACHE.get(url)
    if cached:
        age = now - float(cached.get("ts", 0.0))
        if cached.get("ok") and age <= _URL_CACHE_TTL_SECONDS:
            return cached.get("payload")
        if (not cached.get("ok")) and age <= _FAILURE_COOLDOWN_SECONDS:
            raise URLError(str(cached.get("error", "Fallo reciente de red en caché.")))

    request = Request(url, headers={"User-Agent": "football-analytics-ai/1.0"})
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        _URL_CACHE[url] = {"ok": True, "payload": parsed, "ts": now}
        return parsed
    except Exception as exc:
        _URL_CACHE[url] = {"ok": False, "error": str(exc), "ts": now}
        raise


def _safe_int(value: int | str | None) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None

def get_available_competitions() -> list[dict[str, Any]]:
    url = f"{BASE_OPEN_DATA_URL}/competitions.json"
    try:
        payload = _load_remote_json(url)
        competitions: list[dict[str, Any]] = []
        for item in payload if isinstance(payload, list) else []:
            competition_id = item.get("competition_id")
            season_id = item.get("season_id")
            competition_name = str(item.get("competition_name", "Competición"))
            season_name = str(item.get("season_name", "Temporada"))
            country_name = str(item.get("country_name", ""))
            competitions.append(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "competition_name": competition_name,
                    "season_name": season_name,
                    "country_name": country_name,
                    "display_name": f"{competition_name} - {season_name}",
                }
            )
        if competitions:
            _set_status("competitions", "real")
            return competitions
        _set_status("competitions", "mock", "Competitions vacío o con formato inesperado.")
        return deepcopy(_MOCK_COMPETITIONS)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        _set_status("competitions", "mock", f"No se pudo cargar competitions.json: {exc}")
        return deepcopy(_MOCK_COMPETITIONS)


def get_available_matches(competition_id: int | str, season_id: int | str) -> list[dict[str, Any]]:
    url = f"{BASE_OPEN_DATA_URL}/matches/{competition_id}/{season_id}.json"
    try:
        payload = _load_remote_json(url)
        matches: list[dict[str, Any]] = []
        for item in payload if isinstance(payload, list) else []:
            home_team = str((item.get("home_team") or {}).get("home_team_name", "Local"))
            away_team = str((item.get("away_team") or {}).get("away_team_name", "Visitante"))
            match_date = str(item.get("match_date", "sin fecha"))
            competition = str((item.get("competition") or {}).get("competition_name", "Competición"))
            season = str((item.get("season") or {}).get("season_name", "Temporada"))
            matches.append(
                {
                    "match_id": item.get("match_id"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_date": match_date,
                    "competition": competition,
                    "season": season,
                    "display_name": f"{home_team} vs {away_team} — {match_date}",
                }
            )
        if matches:
            _set_status("matches", "real")
            return matches
        _set_status("matches", "mock", "Matches vacío o con formato inesperado.")
        comp_id = _safe_int(competition_id)
        seas_id = _safe_int(season_id)
        return deepcopy(_MOCK_MATCHES.get((comp_id, seas_id), []))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        _set_status("matches", "mock", f"No se pudo cargar matches/{competition_id}/{season_id}.json: {exc}")
        comp_id = _safe_int(competition_id)
        seas_id = _safe_int(season_id)
        return deepcopy(_MOCK_MATCHES.get((comp_id, seas_id), []))


def get_match_events(match_id: int | str) -> list[dict[str, Any]]:
    url = f"{BASE_OPEN_DATA_URL}/events/{match_id}.json"
    try:
        payload = _load_remote_json(url)
        if isinstance(payload, list):
            _set_status("events", "real")
            return payload
        _set_status("events", "mock", "Events con formato inesperado.")
        return deepcopy(_MOCK_EVENTS.get(_safe_int(match_id), []))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        _set_status("events", "mock", f"No se pudo cargar events/{match_id}.json: {exc}")
        return deepcopy(_MOCK_EVENTS.get(_safe_int(match_id), []))
