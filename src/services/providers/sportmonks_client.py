from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3/football"
DEFAULT_TIMEOUT_SECONDS = 20
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_LOADED = False

# TODO: Confirmar con respuesta real los includes finales disponibles segun plan y endpoint.
SPORTMONKS_DEFAULT_FIXTURE_INCLUDE = ",".join(
    [
        "participants",
        "league",
        "season",
        "venue",
        "state",
        "scores",
        "events",
        "lineups",
        "statistics",
        "metadata",
    ]
)


def get_sportmonks_api_key() -> str | None:
    api_key = str(os.getenv("SPORTMONKS_API_KEY", "") or "").strip()
    if not api_key:
        _load_local_env_file()
        api_key = str(os.getenv("SPORTMONKS_API_KEY", "") or "").strip()
    if not api_key:
        _load_local_env_file(force_reload=True)
        api_key = str(os.getenv("SPORTMONKS_API_KEY", "") or "").strip()
    return api_key or None


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
            pass
    _ENV_LOADED = True


def is_sportmonks_configured() -> bool:
    return bool(get_sportmonks_api_key())


def _normalize_query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return str(value)


def build_sportmonks_url(endpoint: str, params: dict[str, Any] | None = None) -> str:
    normalized_endpoint = str(endpoint or "").strip().lstrip("/")
    base_url = SPORTMONKS_BASE_URL.rstrip("/")
    url = f"{base_url}/{normalized_endpoint}" if normalized_endpoint else base_url

    query_params: dict[str, Any] = dict(params or {})
    api_key = get_sportmonks_api_key()
    if api_key:
        query_params["api_token"] = api_key

    encoded_items: list[tuple[str, str]] = []
    for key, value in query_params.items():
        if value is None:
            continue
        encoded_items.append((str(key), _normalize_query_value(value)))

    if not encoded_items:
        return url
    return f"{url}?{urlencode(encoded_items)}"


def extract_sportmonks_payload(response_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response_json, dict):
        return {
            "data": None,
            "meta": {},
            "pagination": {},
            "rate_limit": {},
        }

    data = response_json.get("data")
    pagination = response_json.get("pagination")
    rate_limit = response_json.get("rate_limit")

    meta: dict[str, Any] = {}
    for key, value in response_json.items():
        if key in {"data", "pagination", "rate_limit"}:
            continue
        meta[str(key)] = value

    return {
        "data": data,
        "meta": meta,
        "pagination": pagination if isinstance(pagination, dict) else {},
        "rate_limit": rate_limit if isinstance(rate_limit, dict) else {},
    }


def _build_error_response(
    error: str,
    status_code: int | None = None,
    data: Any = None,
    meta: dict[str, Any] | None = None,
    pagination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_meta = deepcopy(meta or {})
    resolved_pagination = deepcopy(pagination or {})
    return {
        "ok": False,
        "data": data,
        "error": error,
        "status_code": status_code,
        "meta": resolved_meta,
        "pagination": resolved_pagination,
    }


def _build_success_response(
    data: Any,
    status_code: int | None = None,
    meta: dict[str, Any] | None = None,
    pagination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_meta = deepcopy(meta or {})
    resolved_pagination = deepcopy(pagination or {})
    return {
        "ok": True,
        "data": data,
        "error": "",
        "status_code": status_code,
        "meta": resolved_meta,
        "pagination": resolved_pagination,
    }


def _extract_error_message_from_body(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                for key in ("message", "detail", "error"):
                    value = first.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        if isinstance(errors, dict):
            for value in errors.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _read_http_error_payload(exc: HTTPError) -> Any:
    try:
        body = exc.read().decode("utf-8")
    except Exception:
        return None

    if not body.strip():
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def sportmonks_get(
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if not is_sportmonks_configured():
        return _build_error_response(
            error="Falta configurar SPORTMONKS_API_KEY en el entorno.",
            status_code=None,
            meta={"provider": "sportmonks", "configured": False},
        )

    url = build_sportmonks_url(endpoint, params=params)
    request = Request(url, headers={"User-Agent": "football-analytics-ai/1.0"})

    try:
        with urlopen(request, timeout=int(timeout)) as response:
            payload = response.read().decode("utf-8")
            status_code = getattr(response, "status", None) or response.getcode()
    except HTTPError as exc:
        payload = _read_http_error_payload(exc)
        status_code = int(getattr(exc, "code", 0) or 0) or None
        extracted = extract_sportmonks_payload(payload) if isinstance(payload, dict) else {
            "data": None,
            "meta": {},
            "pagination": {},
            "rate_limit": {},
        }
        if extracted["rate_limit"]:
            extracted["meta"]["rate_limit"] = extracted["rate_limit"]

        if status_code == 429:
            return _build_error_response(
                error="Sportmonks respondio con rate limit. Reintentá en unos instantes.",
                status_code=status_code,
                meta=extracted["meta"],
                pagination=extracted["pagination"],
            )
        if status_code in {401, 403}:
            return _build_error_response(
                error="No fue posible autenticarse contra Sportmonks con la configuracion actual.",
                status_code=status_code,
                meta=extracted["meta"],
                pagination=extracted["pagination"],
            )
        if status_code == 404:
            return _build_error_response(
                error="El recurso solicitado no existe o no esta disponible en Sportmonks.",
                status_code=status_code,
                meta=extracted["meta"],
                pagination=extracted["pagination"],
            )
        if status_code and status_code >= 500:
            return _build_error_response(
                error="Sportmonks devolvio un error interno del servidor.",
                status_code=status_code,
                meta=extracted["meta"],
                pagination=extracted["pagination"],
            )

        detail = _extract_error_message_from_body(payload)
        return _build_error_response(
            error=detail or "No se pudo completar la consulta a Sportmonks.",
            status_code=status_code,
            meta=extracted["meta"],
            pagination=extracted["pagination"],
        )
    except TimeoutError:
        return _build_error_response(
            error="La consulta a Sportmonks excedio el tiempo de espera.",
            status_code=None,
            meta={"provider": "sportmonks"},
        )
    except URLError:
        return _build_error_response(
            error="No se pudo conectar con Sportmonks por un problema de red.",
            status_code=None,
            meta={"provider": "sportmonks"},
        )
    except Exception:
        return _build_error_response(
            error="Ocurrio un error inesperado consultando Sportmonks.",
            status_code=None,
            meta={"provider": "sportmonks"},
        )

    if not payload.strip():
        return _build_error_response(
            error="Sportmonks devolvio una respuesta vacia.",
            status_code=status_code,
            meta={"provider": "sportmonks"},
        )

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return _build_error_response(
            error="Sportmonks devolvio una respuesta JSON invalida.",
            status_code=status_code,
            meta={"provider": "sportmonks"},
        )

    if not isinstance(parsed, dict):
        return _build_error_response(
            error="Sportmonks devolvio una respuesta con formato no esperado.",
            status_code=status_code,
            meta={"provider": "sportmonks"},
        )

    extracted = extract_sportmonks_payload(parsed)
    meta = extracted["meta"]
    if extracted["rate_limit"]:
        meta = deepcopy(meta)
        meta["rate_limit"] = extracted["rate_limit"]

    return _build_success_response(
        data=extracted["data"],
        status_code=status_code,
        meta=meta,
        pagination=extracted["pagination"],
    )


def get_sportmonks_leagues(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return sportmonks_get("leagues", params=params)


def get_sportmonks_seasons(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return sportmonks_get("seasons", params=params)


def get_sportmonks_teams(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return sportmonks_get("teams", params=params)


def get_sportmonks_players(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return sportmonks_get("players", params=params)


def get_sportmonks_fixtures_by_date(date: str, include: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if include:
        params["include"] = include
    return sportmonks_get(f"fixtures/date/{date}", params=params or None)


def get_sportmonks_fixture(fixture_id: str, include: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if include:
        params["include"] = include
    return sportmonks_get(f"fixtures/{fixture_id}", params=params or None)


def get_sportmonks_fixture_full_context(fixture_id: str) -> dict[str, Any]:
    return get_sportmonks_fixture(fixture_id, include=SPORTMONKS_DEFAULT_FIXTURE_INCLUDE)


def get_sportmonks_fixtures_by_league_and_season(
    league_id: str,
    season_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_params = dict(params or {})
    resolved_params.setdefault("filters", {"league_id": league_id, "season_id": season_id})
    # TODO: Confirmar endpoint exacto para fixtures por league + season en Sportmonks v3.
    return sportmonks_get("fixtures", params=resolved_params)


def get_sportmonks_fixture_events(fixture_id: str) -> dict[str, Any]:
    # TODO: Confirmar si conviene endpoint especifico o include final segun plan.
    return get_sportmonks_fixture(fixture_id, include="events")


def get_sportmonks_fixture_lineups(fixture_id: str) -> dict[str, Any]:
    # TODO: Confirmar si conviene endpoint especifico o include final segun plan.
    return get_sportmonks_fixture(fixture_id, include="lineups")


def get_sportmonks_fixture_statistics(fixture_id: str) -> dict[str, Any]:
    # TODO: Confirmar si conviene endpoint especifico o include final segun plan.
    return get_sportmonks_fixture(fixture_id, include="statistics")
