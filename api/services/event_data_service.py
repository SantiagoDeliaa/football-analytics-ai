from __future__ import annotations

from io import BytesIO
from typing import Any

from fastapi import HTTPException, UploadFile, status

from src.services.ai_coach import answer_coach_question
from src.services.ai_coach import build_match_context
from src.services.ai_coach import generate_tactical_diagnosis
from src.services.ai_coach import get_ai_coach_config_status
from src.services.ai_coach import get_suggested_questions
from src.services.api_football_ingestion import get_api_football_api_key
from src.services.api_football_ingestion import get_api_football_countries
from src.services.api_football_ingestion import get_api_football_fixture_events
from src.services.api_football_ingestion import get_api_football_fixture_lineups
from src.services.api_football_ingestion import get_api_football_fixture_players
from src.services.api_football_ingestion import get_api_football_fixture_statistics
from src.services.api_football_ingestion import get_api_football_fixtures
from src.services.api_football_ingestion import get_api_football_leagues
from src.services.api_football_ingestion import get_api_football_status
from src.services.api_football_ingestion import get_api_football_user_message
from src.services.api_football_normalizer import normalize_api_football_events_to_canonical
from src.services.event_normalizer import normalize_event_data
from src.services.insight_generator import generate_match_insights
from src.services.open_event_data_ingestion import get_available_competitions
from src.services.open_event_data_ingestion import get_available_matches
from src.services.open_event_data_ingestion import get_ingestion_status
from src.services.open_event_data_ingestion import get_match_events
from src.services.open_event_insights import generate_open_event_insights
from src.services.open_event_metrics import calculate_open_event_metrics
from src.services.open_event_normalizer import normalize_events_to_canonical
from src.services.pdf_ingestion import ingest_pdf
from src.services.proprietary_metrics import calculate_proprietary_metrics
from src.services.storage.event_data_repository import delete_processed_match
from src.services.storage.event_data_repository import get_processed_matches
from src.services.storage.event_data_repository import load_processed_match_payloads
from src.services.storage.event_data_repository import save_processed_match
from src.services.providers import SportmonksMatchCenterError
from src.services.providers import SportmonksMatchCenterNotFoundError
from src.services.providers import build_sportmonks_match_center
from src.services.providers import is_sportmonks_configured

STATSBOMB_PROVIDER = "StatsBomb Open Data"
API_FOOTBALL_PROVIDER = "API-Football"
SUPPORTED_EVENT_PROVIDERS = {STATSBOMB_PROVIDER, API_FOOTBALL_PROVIDER}
DEFAULT_API_FOOTBALL_COUNTRY = "Argentina"


def list_api_football_countries() -> list[dict[str, Any]]:
    _ensure_api_football_configured()
    countries = get_api_football_countries()
    _raise_for_api_football_error_if_needed(countries)
    return countries


def list_api_football_leagues(
    country: str,
    season: int | str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_api_football_configured()
    leagues = get_api_football_leagues(country=country, season=season, search=search)
    _raise_for_api_football_error_if_needed(leagues)

    normalized: list[dict[str, Any]] = []
    for league in leagues:
        normalized.append(
            {
                "league_id": int(league.get("league_id") or 0),
                "league_name": str(league.get("league_name") or "Liga"),
                "country_name": str(league.get("country_name") or country or DEFAULT_API_FOOTBALL_COUNTRY),
                "type": str(league.get("type") or ""),
                "logo": str(league.get("logo") or ""),
                "seasons": [
                    int(season_item.get("year"))
                    for season_item in league.get("seasons", []) or []
                    if season_item.get("year") is not None
                ],
                "current_season": _resolve_api_football_season(league),
                "display_name": str(league.get("display_name") or league.get("league_name") or "Liga"),
            }
        )
    return normalized


def list_api_football_fixtures(league_id: int | str, season: int | str) -> list[dict[str, Any]]:
    _ensure_api_football_configured()
    fixtures = get_api_football_fixtures(league_id=league_id, season=season)
    _raise_for_api_football_error_if_needed(fixtures)
    return fixtures


def list_competitions(provider: str = STATSBOMB_PROVIDER) -> list[dict[str, Any]]:
    resolved_provider = _resolve_provider(provider)
    if resolved_provider == STATSBOMB_PROVIDER:
        return get_available_competitions()

    _ensure_api_football_configured()
    leagues = get_api_football_leagues(country=DEFAULT_API_FOOTBALL_COUNTRY)
    _raise_for_api_football_error_if_needed(leagues)

    normalized: list[dict[str, Any]] = []
    for league in leagues:
        current_season = _resolve_api_football_season(league)
        normalized.append(
            {
                "competition_id": int(league.get("league_id") or 0),
                "season_id": current_season,
                "competition_name": str(league.get("league_name") or "Liga"),
                "season_name": str(current_season) if current_season is not None else "",
                "country_name": str(league.get("country_name") or DEFAULT_API_FOOTBALL_COUNTRY),
                "display_name": (
                    f"{league.get('display_name', 'Liga')} - {current_season}"
                    if current_season is not None
                    else str(league.get("display_name") or "Liga")
                ),
            }
        )
    return normalized


def list_matches(
    competition_id: int | str,
    season_id: int | str,
    provider: str = STATSBOMB_PROVIDER,
) -> list[dict[str, Any]]:
    resolved_provider = _resolve_provider(provider)
    if resolved_provider == STATSBOMB_PROVIDER:
        return get_available_matches(competition_id=competition_id, season_id=season_id)

    _ensure_api_football_configured()
    fixtures = get_api_football_fixtures(league_id=competition_id, season=season_id)
    _raise_for_api_football_error_if_needed(fixtures)
    return fixtures


def get_sportmonks_match_center_payload(match_id: str) -> dict[str, Any]:
    normalized_match_id = str(match_id or "").strip()
    if not normalized_match_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`match_id` es obligatorio.",
        )
    if not is_sportmonks_configured():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Sportmonks no está configurado. Agregá SPORTMONKS_API_KEY en el archivo .env "
                "o en las variables de entorno para usar este proveedor."
            ),
        )
    try:
        return build_sportmonks_match_center(normalized_match_id)
    except SportmonksMatchCenterNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except SportmonksMatchCenterError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


def analyze_match(payload: dict[str, Any]) -> dict[str, Any]:
    provider = _resolve_provider(payload.get("provider"))
    match_id = str(payload.get("match_id") or "").strip()
    if not match_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`match_id` es obligatorio.",
        )

    selected_team = _normalize_filter(payload.get("team"))
    selected_player = _normalize_filter(payload.get("player"))
    metadata = _build_metadata(payload, provider)

    if provider == STATSBOMB_PROVIDER:
        return _analyze_statsbomb_match(
            match_id=match_id,
            selected_team=selected_team,
            selected_player=selected_player,
            metadata=metadata,
            payload=payload,
        )
    return _analyze_api_football_match(
        match_id=match_id,
        selected_team=selected_team,
        selected_player=selected_player,
        metadata=metadata,
        payload=payload,
    )


def _analyze_statsbomb_match(
    match_id: str,
    selected_team: str | None,
    selected_player: str | None,
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    raw_events = get_match_events(match_id)
    canonical_events = normalize_events_to_canonical(raw_events, match_id=match_id)
    metrics = calculate_open_event_metrics(
        canonical_events=canonical_events,
        selected_team=selected_team,
        selected_player=selected_player,
    )
    full_match_metrics = calculate_open_event_metrics(
        canonical_events=canonical_events,
        selected_team=None,
        selected_player=None,
    )
    insights = generate_open_event_insights(
        metrics=metrics,
        selected_team=selected_team,
        selected_player=selected_player,
    )

    save_processed_match(
        provider=STATSBOMB_PROVIDER,
        match_id=match_id,
        match_metadata=metadata,
        raw_events=raw_events,
        canonical_events=canonical_events,
        metrics=full_match_metrics,
    )

    ingestion_status = get_ingestion_status().get("events", {})
    used_fallback_events = ingestion_status.get("source") == "mock"
    default_match_label = payload.get("match_label") or _build_match_label(metadata, match_id)

    return {
        "provider": STATSBOMB_PROVIDER,
        "match_id": match_id,
        "competition_name": str(metadata["competition_name"]),
        "season_name": str(metadata["season_name"]),
        "match_label": str(default_match_label),
        "home_team": str(metadata["home_team"]),
        "away_team": str(metadata["away_team"]),
        "match_date": str(metadata["match_date"]),
        "canonical_events": canonical_events,
        "metrics": metrics,
        "insights": insights,
        "raw_events_count": len(raw_events),
        "raw_payload": raw_events,
        "used_fallback_events": used_fallback_events,
        "events_status_message": str(ingestion_status.get("message") or ""),
    }


def _analyze_api_football_match(
    match_id: str,
    selected_team: str | None,
    selected_player: str | None,
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    _ensure_api_football_configured()
    raw_events = get_api_football_fixture_events(match_id)
    lineups = get_api_football_fixture_lineups(match_id)
    statistics = get_api_football_fixture_statistics(match_id)
    players = get_api_football_fixture_players(match_id)
    _raise_for_api_football_error_if_needed(raw_events)

    canonical_events = normalize_api_football_events_to_canonical(raw_events, fixture_id=match_id)
    metrics = calculate_open_event_metrics(
        canonical_events=canonical_events,
        selected_team=selected_team,
        selected_player=selected_player,
    )
    full_match_metrics = calculate_open_event_metrics(
        canonical_events=canonical_events,
        selected_team=None,
        selected_player=None,
    )
    insights = generate_open_event_insights(
        metrics=metrics,
        selected_team=selected_team,
        selected_player=selected_player,
    )
    raw_payload = {
        "events": raw_events,
        "lineups": lineups,
        "statistics": statistics,
        "players": players,
    }

    save_processed_match(
        provider=API_FOOTBALL_PROVIDER,
        match_id=match_id,
        match_metadata=metadata,
        raw_events=raw_payload,
        canonical_events=canonical_events,
        metrics=full_match_metrics,
    )

    default_match_label = payload.get("match_label") or _build_match_label(metadata, match_id)
    status_payload = get_api_football_status()

    return {
        "provider": API_FOOTBALL_PROVIDER,
        "match_id": match_id,
        "competition_name": str(metadata["competition_name"]),
        "season_name": str(metadata["season_name"]),
        "match_label": str(default_match_label),
        "home_team": str(metadata["home_team"]),
        "away_team": str(metadata["away_team"]),
        "match_date": str(metadata["match_date"]),
        "canonical_events": canonical_events,
        "metrics": metrics,
        "insights": insights,
        "raw_events_count": len(raw_events),
        "raw_payload": raw_payload,
        "used_fallback_events": False,
        "events_status_message": str(status_payload.get("message") or ""),
    }


def list_processed_history(limit: int = 20) -> list[dict[str, Any]]:
    history_rows = get_processed_matches(limit=limit)
    normalized: list[dict[str, Any]] = []
    for row in history_rows:
        normalized.append(
            {
                "provider": str(row.get("provider") or ""),
                "match_id": str(row.get("match_id") or ""),
                "competition_name": str(row.get("competition_name") or ""),
                "season_name": str(row.get("season_name") or ""),
                "home_team": str(row.get("home_team") or ""),
                "away_team": str(row.get("away_team") or ""),
                "match_date": str(row.get("match_date") or ""),
                "created_at": str(row.get("created_at") or ""),
                "updated_at": str(row.get("updated_at") or ""),
            }
        )
    return normalized


def load_processed_history_entry(provider: str, match_id: str) -> dict[str, Any]:
    payloads = load_processed_match_payloads(provider=provider, match_id=match_id)
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró el match procesado {provider}:{match_id}.",
        )

    metadata = payloads.get("metadata", {}) or {}
    canonical_events = payloads.get("canonical_events", []) or []
    metrics = payloads.get("metrics", {}) or {}
    insights = generate_open_event_insights(metrics=metrics)
    raw_events = payloads.get("raw_events", []) or []
    raw_events_count = len(raw_events.get("events", [])) if isinstance(raw_events, dict) else len(raw_events)

    return {
        "provider": str(provider),
        "match_id": str(match_id),
        "competition_name": str(metadata.get("competition_name") or provider or STATSBOMB_PROVIDER),
        "season_name": str(metadata.get("season_name") or ""),
        "match_label": _build_match_label(metadata, str(match_id)),
        "home_team": str(metadata.get("home_team") or ""),
        "away_team": str(metadata.get("away_team") or ""),
        "match_date": str(metadata.get("match_date") or ""),
        "canonical_events": canonical_events,
        "metrics": metrics,
        "insights": insights,
        "raw_events_count": raw_events_count,
        "raw_payload": raw_events,
        "used_fallback_events": False,
        "events_status_message": "",
    }


def delete_processed_history_entry(provider: str, match_id: str) -> dict[str, Any]:
    return delete_processed_match(provider=provider, match_id=match_id)


def get_ai_coach_status() -> dict[str, Any]:
    status_payload = get_ai_coach_config_status()
    return {
        "configured": bool(status_payload.get("configured")),
        "api_key_configured": bool(status_payload.get("api_key_configured")),
        "model_configured": bool(status_payload.get("model_configured")),
        "base_url_configured": bool(status_payload.get("base_url_configured")),
        "model": str(status_payload.get("model") or ""),
        "base_url": str(status_payload.get("base_url") or ""),
        "message": str(status_payload.get("message") or ""),
    }


def generate_ai_coach_diagnosis(payload: dict[str, Any]) -> dict[str, Any]:
    match_context = _build_ai_coach_match_context(payload)
    response = generate_tactical_diagnosis(match_context)
    return {
        "ok": bool(response.get("ok")),
        "diagnosis": str(response.get("diagnosis") or ""),
        "error": str(response.get("error") or ""),
        "suggested_questions": get_suggested_questions(match_context),
    }


def answer_ai_coach_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    match_context = _build_ai_coach_match_context(payload)
    conversation_history = _normalize_conversation_history(payload.get("conversation_history"))
    response = answer_coach_question(
        match_context,
        str(payload.get("question") or ""),
        conversation_history=conversation_history,
    )
    return {
        "ok": bool(response.get("ok")),
        "answer": str(response.get("answer") or ""),
        "error": str(response.get("error") or ""),
        "suggested_questions": get_suggested_questions(match_context),
    }


def analyze_pdf_report(file: UploadFile) -> dict[str, Any]:
    raw_payload = ingest_pdf(_UploadAdapter(file))
    if raw_payload.get("status") != "ok":
        detail = " ".join(raw_payload.get("messages") or []) or "No se pudo extraer texto utilizable del PDF."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    normalized_payload = normalize_event_data(raw_payload)
    proprietary_metrics = calculate_proprietary_metrics(
        normalized_payload=normalized_payload,
        raw_payload=raw_payload,
    )
    metrics = map_pdf_metrics(normalized_payload, proprietary_metrics)
    insights = generate_match_insights(normalized_payload, proprietary_metrics)

    return {
        "normalized_payload": normalized_payload,
        "metrics": metrics,
        "insights": insights,
        "ingestion": {
            "status": str(raw_payload.get("status") or "error"),
            "parser": raw_payload.get("parser_used"),
            "page_count": int(raw_payload.get("page_count") or 0),
            "bytes_size": int(raw_payload.get("bytes_size") or 0),
            "messages": list(raw_payload.get("messages") or []),
        },
    }


def map_pdf_metrics(
    normalized_payload: dict[str, Any],
    proprietary_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attack_signals = _signals_from_section(normalized_payload, "attack")
    defense_signals = _signals_from_section(normalized_payload, "defense")
    transition_signals = _signals_from_section(normalized_payload, "transitions")
    build_up_signals = _signals_from_section(normalized_payload, "build_up")
    finishing_signals = _signals_from_section(normalized_payload, "finishing")

    total_shots = _int_signal(attack_signals, "shots")
    total_passes = _int_signal(build_up_signals, "possession") + _int_signal(build_up_signals, "progression")
    progressive_actions = (
        _int_signal(build_up_signals, "progression")
        + _int_signal(transition_signals, "direct attack")
        + _int_signal(transition_signals, "counter")
    )
    final_third_actions = _int_signal(attack_signals, "final third") + _int_signal(attack_signals, "box entries")
    recoveries = _int_signal(defense_signals, "recoveries") + _int_signal(transition_signals, "regain")
    total_under_pressure = _int_signal(defense_signals, "pressing") + _int_signal(defense_signals, "duels")
    total_events = total_passes + total_shots + progressive_actions + recoveries + total_under_pressure
    total_xg = round((_float_signal(attack_signals, "xg") + _float_signal(finishing_signals, "xg")) * 0.15, 3)

    field_tilt = _metric_score(proprietary_metrics, "field_tilt_index")
    directness = _metric_score(proprietary_metrics, "directness_index")
    pressing = _metric_score(proprietary_metrics, "pressing_efficiency")
    build_up_risk = _metric_score(proprietary_metrics, "risk_exposure_score")
    shot_quality = min(
        100,
        round((_float_signal(finishing_signals, "on target") * 25) + (_float_signal(attack_signals, "xg") * 18), 1),
    )
    progressive_threat = min(
        100,
        round(final_third_actions * 8 + progressive_actions * 10 + total_xg * 30, 1),
    )

    return {
        "total_events": int(total_events),
        "total_passes": int(total_passes),
        "total_shots": int(total_shots),
        "progressive_actions": int(progressive_actions),
        "final_third_actions": int(final_third_actions),
        "recoveries": int(recoveries),
        "total_under_pressure": int(total_under_pressure),
        "total_xg": total_xg,
        "field_tilt_index": field_tilt,
        "field_tilt_label": _score_label(field_tilt),
        "directness_index": directness,
        "directness_label": _score_label(directness),
        "progressive_threat_index": progressive_threat,
        "progressive_threat_label": _score_label(progressive_threat),
        "recovery_height_index": pressing,
        "recovery_height_label": _score_label(pressing),
        "shot_quality_index": shot_quality,
        "shot_quality_label": _score_label(shot_quality),
        "player_influence_score": None,
        "player_influence_label": "No aplica",
    }


def _resolve_provider(provider: Any) -> str:
    resolved = str(provider or "").strip()
    if resolved in SUPPORTED_EVENT_PROVIDERS:
        return resolved
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Provider no soportado: {resolved or 'vacío'}.",
    )


def _build_metadata(payload: dict[str, Any], provider: str) -> dict[str, Any]:
    default_competition = STATSBOMB_PROVIDER if provider == STATSBOMB_PROVIDER else API_FOOTBALL_PROVIDER
    return {
        "competition_name": payload.get("competition_name") or default_competition,
        "season_name": payload.get("season_name") or "",
        "home_team": payload.get("home_team") or "",
        "away_team": payload.get("away_team") or "",
        "match_date": payload.get("match_date") or "",
    }


def _build_ai_coach_match_context(payload: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_match(payload)
    provider = analysis.get("provider") or payload.get("provider")
    provider_capabilities = _build_provider_capabilities_from_analysis(
        provider=provider,
        canonical_events=analysis.get("canonical_events", []) or [],
        metrics=analysis.get("metrics", {}) or {},
    )
    return build_match_context(
        provider=provider,
        match_metadata={
            "provider": provider,
            "match_id": analysis.get("match_id"),
            "competition_name": analysis.get("competition_name"),
            "season_name": analysis.get("season_name"),
            "home_team": analysis.get("home_team"),
            "away_team": analysis.get("away_team"),
            "match_date": analysis.get("match_date"),
        },
        canonical_events=analysis.get("canonical_events", []) or [],
        metrics=analysis.get("metrics", {}) or {},
        insights=analysis.get("insights", []) or [],
        selected_team=_normalize_filter(payload.get("team")),
        selected_player=_normalize_filter(payload.get("player")),
        provider_capabilities=provider_capabilities,
        raw_summary=None,
    )


def _build_provider_capabilities_from_analysis(
    provider: Any,
    canonical_events: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, bool]:
    provider_key = str(provider or "").strip().lower()
    has_coordinates = any(
        isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
        for event in canonical_events
    )
    has_xg = any((event.get("xG") or 0) not in {0, 0.0, None} for event in canonical_events)
    has_xg = has_xg or bool((metrics or {}).get("total_xg"))

    if provider_key == "api-football":
        return {
            "has_event_coordinates": has_coordinates,
            "has_lineups": True,
            "has_team_stats": True,
            "has_player_stats": True,
            "has_xg": has_xg,
            "has_event_timeline": bool(canonical_events),
        }

    return {
        "has_event_coordinates": has_coordinates,
        "has_lineups": False,
        "has_team_stats": False,
        "has_player_stats": False,
        "has_xg": has_xg,
        "has_event_timeline": bool(canonical_events),
    }


def _normalize_conversation_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized[-6:]


def _resolve_api_football_season(league: dict[str, Any]) -> int | None:
    current_season = league.get("current_season")
    if isinstance(current_season, int):
        return current_season
    seasons = league.get("seasons", []) or []
    available_years = [int(item.get("year")) for item in seasons if item.get("year") is not None]
    return max(available_years) if available_years else None


def _ensure_api_football_configured() -> None:
    if get_api_football_api_key():
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Falta configurar API_FOOTBALL_KEY en el entorno.",
    )


def _raise_for_api_football_error_if_needed(items: list[dict[str, Any]]) -> None:
    status_payload = get_api_football_status()
    if items or status_payload.get("status") != "error":
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=get_api_football_user_message(status_payload) or "No se pudieron cargar datos desde API-Football.",
    )


def _normalize_filter(value: Any) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    if not normalized or normalized == "Todos":
        return None
    return normalized


def _build_match_label(metadata: dict[str, Any], match_id: str) -> str:
    home_team = str(metadata.get("home_team") or "").strip()
    away_team = str(metadata.get("away_team") or "").strip()
    match_date = str(metadata.get("match_date") or "").strip()
    if home_team and away_team and match_date:
        return f"{home_team} vs {away_team} — {match_date}"
    if home_team and away_team:
        return f"{home_team} vs {away_team}"
    return f"Match {match_id}"


def _signals_from_section(payload: dict[str, Any], section: str) -> dict[str, Any]:
    section_data = payload.get(section, {}) or {}
    return section_data.get("signals", {}) if isinstance(section_data, dict) else {}


def _metric_score(proprietary_metrics: dict[str, dict[str, Any]], key: str) -> int | float | None:
    raw_value = (proprietary_metrics.get(key) or {}).get("score")
    if isinstance(raw_value, (int, float)):
        return round(float(raw_value), 1)
    return None


def _score_label(score: int | float | None) -> str:
    if score is None:
        return "No aplica"
    if score < 40:
        return "Bajo"
    if score < 70:
        return "Medio"
    return "Alto"


def _int_signal(signals: dict[str, Any], key: str) -> int:
    try:
        return int(float(signals.get(key, 0.0) or 0.0))
    except Exception:
        return 0


def _float_signal(signals: dict[str, Any], key: str) -> float:
    try:
        return float(signals.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


class _UploadAdapter:
    def __init__(self, upload_file: UploadFile) -> None:
        self._payload = BytesIO(upload_file.file.read())
        self.name = upload_file.filename or "unknown.pdf"

    def read(self) -> bytes:
        self._payload.seek(0)
        return self._payload.read()
