from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
import json
from typing import Any, Callable

import streamlit as st

from src.services.ai_coach import get_ai_coach_config_status
from src.services.api_football_ingestion import get_api_football_config_status
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
from src.services.open_event_data_ingestion import (
    get_available_competitions,
    get_available_matches,
    get_ingestion_status,
    get_match_events,
)
from src.services.open_event_insights import generate_open_event_insights
from src.services.open_event_metrics import calculate_open_event_metrics
from src.services.open_event_normalizer import normalize_events_to_canonical
from src.services.open_event_visualizations import create_event_map
from src.services.open_event_visualizations import create_player_action_map
from src.services.open_event_visualizations import create_progressive_actions_map
from src.services.open_event_visualizations import create_recoveries_map
from src.services.open_event_visualizations import create_shot_map
from src.services.presentation import build_expected_metrics_table
from src.services.presentation import build_player_stats_table
from src.services.presentation import build_sportmonks_player_insights
from src.services.presentation import build_team_stats_table
from src.services.presentation import build_timeline_table
from src.services.presentation import _render_stat_value
from src.services.presentation import format_availability_status
from src.services.presentation import format_sportmonks_match_title
from src.services.providers.sportmonks_adapter import (
    get_sportmonks_data_availability_for_fixture as adapter_get_sportmonks_data_availability_for_fixture,
)
from src.services.providers.sportmonks_adapter import (
    get_sportmonks_fixtures_by_date as adapter_get_sportmonks_fixtures_by_date,
)
from src.services.providers.sportmonks_adapter import (
    get_sportmonks_match_context as adapter_get_sportmonks_match_context,
)
from src.services.providers.sportmonks_client import is_sportmonks_configured
from src.services.storage.event_data_repository import get_processed_matches
from src.services.storage.event_data_repository import has_processed_match
from src.services.storage.event_data_repository import initialize_event_data_persistence
from src.services.storage.event_data_repository import load_processed_match_payloads
from src.services.storage.event_data_repository import save_processed_match
from src.utils.ui.ai_coach_panel import render_ai_coach_panel

STORAGE_PROVIDER_STATSBOMB = "statsbomb"
STORAGE_PROVIDER_API_FOOTBALL = "api_football"
STORAGE_PROVIDER_SPORTMONKS = "sportmonks"
SESSION_RESULT_KEY = "vertical2_api_event_result"


def _selectbox(
    label: str,
    options: list[Any],
    key: str,
    format_func: Callable[[Any], str] | None = None,
    index: int = 0,
) -> Any:
    selectbox_fn = getattr(st, "selectbox", None)
    kwargs: dict[str, Any] = {"key": key, "index": index}
    if format_func is not None:
        kwargs["format_func"] = format_func

    if callable(selectbox_fn):
        return selectbox_fn(label, options, **kwargs)
    return st.sidebar.selectbox(label, options, **kwargs)


def _competition_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", "Competición - Temporada"))


def _match_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", "Partido sin nombre"))


def _country_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", item.get("name", "País")))


def _league_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", item.get("league_name", "Liga")))


def _fixture_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", "Partido"))


def _date_input(label: str, value: str, key: str) -> str:
    date_input_fn = getattr(st, "date_input", None)
    if callable(date_input_fn):
        try:
            selected = date_input_fn(label, value=date.fromisoformat(value), key=key)
            return selected.isoformat() if hasattr(selected, "isoformat") else str(selected)
        except Exception:
            pass
    return st.text_input(label, value=value, key=key)


def _object_value(item: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def _object_as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if is_dataclass(item):
        return asdict(item)
    return {}


def _sportmonks_fixture_label(item: Any) -> str:
    competition = str(_object_value(item, "competition_name", "") or "Competición no informada")
    match_date = str(_object_value(item, "match_date", "") or "Fecha no informada")
    status = str(_object_value(item, "status", "") or "Estado no informado")
    fixture_id = str(_object_value(item, "provider_match_id", "") or _object_value(item, "match_id", "") or "")
    return f"{format_sportmonks_match_title(item)} | {competition} | {match_date} | {status} | #{fixture_id}"

def _identity_cache_decorator(*args: Any, **kwargs: Any):
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _decorator


_cache_data = getattr(st, "cache_data", None)
_cache_decorator = _cache_data if callable(_cache_data) else _identity_cache_decorator


@_cache_decorator(ttl=1800, show_spinner=False)
def _cached_competitions() -> list[dict[str, Any]]:
    return get_available_competitions()


@_cache_decorator(ttl=1800, show_spinner=False)
def _cached_matches(competition_id: int | str, season_id: int | str) -> list[dict[str, Any]]:
    return get_available_matches(competition_id, season_id)


@_cache_decorator(ttl=1800, show_spinner=False)
def _cached_events(match_id: int | str) -> list[dict[str, Any]]:
    return get_match_events(match_id)


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_countries() -> list[dict[str, Any]]:
    return get_api_football_countries()


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_leagues(
    country: str | None,
    season: int | str | None,
    search: str | None,
) -> list[dict[str, Any]]:
    return get_api_football_leagues(country=country, season=season, search=search)


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_fixtures(
    league_id: int | str,
    season: int | str,
    last: int | None,
) -> list[dict[str, Any]]:
    return get_api_football_fixtures(league_id=league_id, season=season, last=last)


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_fixture_events(fixture_id: int | str) -> list[dict[str, Any]]:
    return get_api_football_fixture_events(fixture_id)


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_fixture_lineups(fixture_id: int | str) -> list[dict[str, Any]]:
    return get_api_football_fixture_lineups(fixture_id)


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_fixture_statistics(fixture_id: int | str) -> list[dict[str, Any]]:
    return get_api_football_fixture_statistics(fixture_id)


@_cache_decorator(ttl=900, show_spinner=False)
def _cached_api_fixture_players(fixture_id: int | str) -> list[dict[str, Any]]:
    return get_api_football_fixture_players(fixture_id)


@_cache_decorator(ttl=300, show_spinner=False)
def _cached_sportmonks_fixtures_by_date(match_date: str) -> dict[str, Any]:
    return adapter_get_sportmonks_fixtures_by_date(match_date)


@_cache_decorator(ttl=300, show_spinner=False)
def _cached_sportmonks_match_context(fixture_id: str) -> dict[str, Any]:
    return adapter_get_sportmonks_match_context(fixture_id)


@_cache_decorator(ttl=300, show_spinner=False)
def _cached_sportmonks_availability(fixture_id: str) -> dict[str, Any]:
    return adapter_get_sportmonks_data_availability_for_fixture(fixture_id)


def _extract_teams_from_canonical(canonical_events: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(event.get("team_name"))
            for event in canonical_events
            if event.get("team_name") and str(event.get("team_name")).strip()
        }
    )


def _extract_players_from_canonical(canonical_events: list[dict[str, Any]], selected_team: str) -> list[str]:
    players_set: set[str] = set()
    for event in canonical_events:
        team_name = str(event.get("team_name", ""))
        if selected_team != "Todos" and team_name != selected_team:
            continue
        player_name = str(event.get("player_name", ""))
        if player_name and player_name != "Jugador desconocido":
            players_set.add(player_name)
    return sorted(players_set)


def _extract_teams_from_payload(raw_payload: Any) -> list[str]:
    teams: set[str] = set()
    if not isinstance(raw_payload, dict):
        return []
    for event in raw_payload.get("events", []) or []:
        team_name = str((event.get("team") or {}).get("name", "") or "")
        if team_name:
            teams.add(team_name)
    for lineup in raw_payload.get("lineups", []) or []:
        team_name = str((lineup.get("team") or {}).get("name", "") or "")
        if team_name:
            teams.add(team_name)
    for stats in raw_payload.get("statistics", []) or []:
        team_name = str((stats.get("team") or {}).get("name", "") or "")
        if team_name:
            teams.add(team_name)
    for team_players in raw_payload.get("players", []) or []:
        team_name = str((team_players.get("team") or {}).get("name", "") or "")
        if team_name:
            teams.add(team_name)
    return sorted(teams)


def _extract_players_from_payload(raw_payload: Any, selected_team: str) -> list[str]:
    players: set[str] = set()
    if not isinstance(raw_payload, dict):
        return []
    for event in raw_payload.get("events", []) or []:
        team_name = str((event.get("team") or {}).get("name", "") or "")
        if selected_team != "Todos" and team_name != selected_team:
            continue
        player_name = str((event.get("player") or {}).get("name", "") or "")
        if player_name:
            players.add(player_name)
    for lineup in raw_payload.get("lineups", []) or []:
        team_name = str((lineup.get("team") or {}).get("name", "") or "")
        if selected_team != "Todos" and team_name != selected_team:
            continue
        for block in ("startXI", "substitutes"):
            for entry in lineup.get(block, []) or []:
                player_name = str(((entry.get("player") or {}).get("name", "")) or "")
                if player_name:
                    players.add(player_name)
    for team_players in raw_payload.get("players", []) or []:
        team_name = str((team_players.get("team") or {}).get("name", "") or "")
        if selected_team != "Todos" and team_name != selected_team:
            continue
        for player_entry in team_players.get("players", []) or []:
            player_name = str(((player_entry.get("player") or {}).get("name", "")) or "")
            if player_name:
                players.add(player_name)
    return sorted(players)


def _find_default_index(options: list[Any], predicate: Callable[[Any], bool]) -> int:
    for index, option in enumerate(options):
        try:
            if predicate(option):
                return index
        except Exception:
            continue
    return 0


def _result_matches(result: dict[str, Any] | None, provider_key: str, match_key: str) -> bool:
    return bool(
        isinstance(result, dict)
        and str(result.get("provider")) == provider_key
        and str(result.get("match_id")) == match_key
    )


def _render_fallback_warning(scope: str) -> None:
    scope_es = {"competitions": "competiciones", "matches": "partidos", "events": "eventos"}
    status = get_ingestion_status().get(scope, {})
    if status.get("source") == "mock":
        message = status.get("message", "")
        detail = f" ({message})" if message else ""
        st.warning(f"Mostrando datos de fallback para {scope_es.get(scope, scope)}{detail}")


def _metric_value_text(value: Any) -> str:
    if value is None:
        return "No aplica"
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return str(value)


def _render_proprietary_metric_card(
    title: str,
    value: Any,
    label: str | None,
    description: str,
) -> None:
    st.metric(title, _metric_value_text(value))
    if label:
        st.caption(f"Nivel: {label}")
    st.caption(description)


def _render_environment_config_status(show_technical_info: bool = False) -> None:
    if not show_technical_info:
        return

    ai_status = get_ai_coach_config_status()
    api_status = get_api_football_config_status()
    sportmonks_configured = is_sportmonks_configured()

    with st.expander("Estado técnico de variables de entorno"):
        st.json(
            {
                "API_FOOTBALL_KEY": {"configured": bool(api_status.get("configured"))},
                "SPORTMONKS_API_KEY": {"configured": sportmonks_configured},
                "AI_COACH_API_KEY": {"configured": bool(ai_status.get("api_key_configured"))},
                "AI_COACH_MODEL": {
                    "configured": bool(ai_status.get("model_configured")),
                    "using_default": not bool(ai_status.get("model_configured")),
                },
                "AI_COACH_BASE_URL": {
                    "configured": bool(ai_status.get("base_url_configured")),
                    "using_default": not bool(ai_status.get("base_url_configured")),
                },
            }
        )


def _has_pitch_coordinates(canonical_events: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
        for event in canonical_events
    )


def _render_local_history(provider_filter: str) -> None:
    st.markdown("### Historial local")
    try:
        history_rows = get_processed_matches(limit=20)
    except Exception as exc:
        st.warning(f"No se pudo cargar el historial local: {exc}")
        history_rows = []
    if provider_filter:
        history_rows = [row for row in history_rows if row.get("provider") == provider_filter]
    if not history_rows:
        st.info("Todavía no hay partidos guardados localmente.")
        return

    history_for_ui = []
    for row in history_rows:
        competition = str(row.get("competition_name", "") or "")
        season = str(row.get("season_name", "") or "")
        home_team = str(row.get("home_team", "") or "")
        away_team = str(row.get("away_team", "") or "")
        if home_team and away_team:
            match_text = f"{home_team} vs {away_team}"
        else:
            match_text = str(row.get("match_id", "Partido"))
        history_for_ui.append(
            {
                "Proveedor": row.get("provider", ""),
                "Partido": match_text,
                "Competición/Temporada": f"{competition} - {season}".strip(" - "),
                "Procesado": row.get("updated_at", row.get("created_at", "")),
            }
        )
    st.dataframe(history_for_ui, use_container_width=True, hide_index=True)


def _render_common_event_dashboard(
    result: dict[str, Any],
    selected_team: str,
    selected_player: str,
    show_technical_info: bool = False,
) -> None:
    canonical_events = result.get("canonical_events", []) or []
    team_filter = None if selected_team == "Todos" else selected_team
    player_filter = None if selected_player == "Todos" else selected_player
    chart_scope = f"{result.get('provider', 'provider')}-{result.get('match_id', 'match')}-{selected_team}-{selected_player}"
    metrics = calculate_open_event_metrics(
        canonical_events,
        selected_team=team_filter,
        selected_player=player_filter,
    )
    insights = generate_open_event_insights(
        metrics,
        selected_team=team_filter,
        selected_player=player_filter,
    )
    has_coordinates = _has_pitch_coordinates(canonical_events)

    st.markdown("### Resumen del partido")
    st.caption(
        f"Competición: {result.get('competition_name')} | "
        f"Partido: {result.get('match_label')} | "
        f"Equipo: {selected_team} | "
        f"Jugador: {selected_player}"
    )

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    with row1_col1:
        st.metric("Eventos analizados", int(metrics.get("total_events", 0)))
    with row1_col2:
        st.metric("Pases", int(metrics.get("total_passes", 0)))
    with row1_col3:
        st.metric("Remates", int(metrics.get("total_shots", 0)))
    with row1_col4:
        st.metric("xG total", f"{float(metrics.get('total_xg', 0.0)):.2f}")

    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    with row2_col1:
        st.metric("Acciones progresivas", int(metrics.get("progressive_actions", 0)))
    with row2_col2:
        st.metric("Acciones en último tercio", int(metrics.get("final_third_actions", 0)))
    with row2_col3:
        st.metric("Recuperaciones", int(metrics.get("recoveries", 0)))
    with row2_col4:
        st.metric("Acciones bajo presión", int(metrics.get("total_under_pressure", 0)))

    st.markdown("### Métricas propietarias")
    prop_col1, prop_col2, prop_col3 = st.columns(3)
    with prop_col1:
        _render_proprietary_metric_card(
            title="Dominio territorial",
            value=metrics.get("field_tilt_index"),
            label=str(metrics.get("field_tilt_label", "No aplica")),
            description="Mide cuánto peso tuvo el equipo en el último tercio respecto del total del partido.",
        )
    with prop_col2:
        _render_proprietary_metric_card(
            title="Verticalidad",
            value=metrics.get("directness_index"),
            label=str(metrics.get("directness_label", "No aplica")),
            description="Indica qué tan directo progresa el equipo en relación con su volumen de pases.",
        )
    with prop_col3:
        _render_proprietary_metric_card(
            title="Amenaza progresiva",
            value=metrics.get("progressive_threat_index"),
            label=str(metrics.get("progressive_threat_label", "No aplica")),
            description="Resume cuánto peligro genera el equipo al progresar, llegar al último tercio y rematar.",
        )

    prop_row2_col1, prop_row2_col2 = st.columns(2)
    with prop_row2_col1:
        _render_proprietary_metric_card(
            title="Altura de recuperación",
            value=metrics.get("recovery_height_index"),
            label=str(metrics.get("recovery_height_label", "No aplica")),
            description="Refleja en qué zonas del campo recupera la pelota el equipo, desde campo propio hasta campo rival.",
        )
    with prop_row2_col2:
        _render_proprietary_metric_card(
            title="Calidad de remate",
            value=metrics.get("shot_quality_index"),
            label=str(metrics.get("shot_quality_label", "No aplica")),
            description="Estima la calidad promedio de las ocasiones de remate generadas por el equipo.",
        )
    if selected_player != "Todos":
        _render_proprietary_metric_card(
            title="Influencia del jugador",
            value=metrics.get("player_influence_score"),
            label=str(metrics.get("player_influence_label", "No aplica")),
            description="Sintetiza la participación del jugador en volumen de juego, progresión, amenaza y recuperación.",
        )

    render_ai_coach_panel(
        result=result,
        metrics=metrics,
        insights=insights,
        selected_team=selected_team,
        selected_player=selected_player,
        show_technical_info=show_technical_info,
    )

    st.markdown("### Visualizaciones tácticas")
    if not has_coordinates:
        st.info(
            "Este provider no entrega coordenadas de eventos para este partido. "
            "Se muestran métricas y eventos disponibles."
        )
    else:
        map_tab, shots_tab, progressive_tab, recoveries_tab = st.tabs(
            ["Mapa de eventos", "Remates", "Progresiones", "Recuperaciones"]
        )
        with map_tab:
            st.plotly_chart(
                create_event_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
                use_container_width=True,
                key=f"event-map-{chart_scope}",
            )
        with shots_tab:
            st.plotly_chart(
                create_shot_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
                use_container_width=True,
                key=f"shot-map-{chart_scope}",
            )
        with progressive_tab:
            st.plotly_chart(
                create_progressive_actions_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
                use_container_width=True,
                key=f"progressive-map-{chart_scope}",
            )
        with recoveries_tab:
            st.plotly_chart(
                create_recoveries_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
                use_container_width=True,
                key=f"recoveries-map-{chart_scope}",
            )

    st.markdown("#### Insights iniciales")
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No hay insights disponibles para esta selección.")

    st.markdown("### Análisis de jugador")
    if selected_player != "Todos":
        player_metrics = calculate_open_event_metrics(
            canonical_events,
            selected_team=team_filter,
            selected_player=selected_player,
        )
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
        with p_col1:
            st.metric("Eventos", int(player_metrics.get("total_events", 0)))
        with p_col2:
            st.metric("Pases", int(player_metrics.get("total_passes", 0)))
        with p_col3:
            st.metric("Remates", int(player_metrics.get("total_shots", 0)))
        with p_col4:
            st.metric("Progresiones", int(player_metrics.get("progressive_actions", 0)))
        with p_col5:
            st.metric("Recuperaciones", int(player_metrics.get("recoveries", 0)))
        if has_coordinates:
            st.plotly_chart(
                create_player_action_map(canonical_events, selected_player=selected_player),
                use_container_width=True,
                key=f"player-action-map-{chart_scope}",
            )
        else:
            st.info("No hay coordenadas disponibles para mostrar el mapa de acciones del jugador.")
    else:
        st.info("Seleccioná un jugador específico para ver su mapa de acciones.")

    if show_technical_info:
        st.markdown("### Información técnica")
        with st.expander("Ver preview del Canonical Event Model (JSON)"):
            st.code(json.dumps(canonical_events[:10], indent=2, ensure_ascii=False), language="json")
        with st.expander("Ver payload del provider (JSON)"):
            raw_payload_preview = result.get("raw_payload", {})
            if isinstance(raw_payload_preview, list):
                raw_payload_preview = raw_payload_preview[:10]
            st.code(json.dumps(raw_payload_preview, indent=2, ensure_ascii=False), language="json")


def _summarize_api_football_events(raw_events: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"goles": 0, "tarjetas": 0, "sustituciones": 0}
    for event in raw_events:
        event_type = str(event.get("type", "") or "").lower()
        detail = str(event.get("detail", "") or "").lower()
        if "goal" in event_type or "goal" in detail:
            summary["goles"] += 1
        if "card" in event_type or "card" in detail:
            summary["tarjetas"] += 1
        if "subst" in event_type or "substitution" in event_type or "subst" in detail:
            summary["sustituciones"] += 1
    return summary


def _render_api_football_provider_sections(result: dict[str, Any], show_technical_info: bool = False) -> None:
    raw_payload = result.get("raw_payload", {}) or {}
    raw_events = raw_payload.get("events", []) or []
    lineups = raw_payload.get("lineups", []) or []
    statistics = raw_payload.get("statistics", []) or []
    players = raw_payload.get("players", []) or []
    summary = _summarize_api_football_events(raw_events)

    st.markdown("### Datos disponibles del provider")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Eventos crudos", len(raw_events))
    with info_col2:
        st.metric("Lineups", len(lineups))
    with info_col3:
        st.metric("Bloques estadísticos", len(statistics))
    with info_col4:
        st.metric("Bloques de jugadores", len(players))

    extra_col1, extra_col2, extra_col3 = st.columns(3)
    with extra_col1:
        st.metric("Goles detectados", summary["goles"])
    with extra_col2:
        st.metric("Tarjetas detectadas", summary["tarjetas"])
    with extra_col3:
        st.metric("Sustituciones detectadas", summary["sustituciones"])

    st.markdown("### Resumen de formaciones")
    if not lineups:
        st.info("No hay formaciones disponibles para este partido.")
    else:
        lineup_columns = st.columns(max(1, min(2, len(lineups))))
        for index, lineup in enumerate(lineups):
            team_name = str((lineup.get("team") or {}).get("name", "") or "Equipo")
            formation = str(lineup.get("formation", "") or "Sin formación reportada")
            starters = [
                str(((entry.get("player") or {}).get("name", "")) or "")
                for entry in lineup.get("startXI", []) or []
                if ((entry.get("player") or {}).get("name"))
            ]
            substitutes = [
                str(((entry.get("player") or {}).get("name", "")) or "")
                for entry in lineup.get("substitutes", []) or []
                if ((entry.get("player") or {}).get("name"))
            ]
            with lineup_columns[index % len(lineup_columns)]:
                st.markdown(f"**{team_name}**")
                st.caption(f"Formación: {formation}")
                st.caption(f"Titulares reportados: {len(starters)}")
                st.caption(f"Suplentes reportados: {len(substitutes)}")
                if starters:
                    st.caption("Primeros titulares detectados: " + ", ".join(starters[:5]))

    st.markdown("### Estadísticas destacadas")
    if not statistics:
        st.info("No hay estadísticas por equipo disponibles.")
    else:
        stat_columns = st.columns(max(1, min(2, len(statistics))))
        for index, team_block in enumerate(statistics):
            team_name = str((team_block.get("team") or {}).get("name", "") or "Equipo")
            with stat_columns[index % len(stat_columns)]:
                st.markdown(f"**{team_name}**")
                for stat in (team_block.get("statistics", []) or [])[:6]:
                    stat_type = str(stat.get("type", "") or "Métrica")
                    stat_value = stat.get("value", "N/D")
                    st.caption(f"{stat_type}: {stat_value}")

    if show_technical_info:
        st.markdown("### Información técnica del provider")
        if raw_events:
            with st.expander("Ver timeline de eventos"):
                timeline_rows = []
                for event in raw_events[:50]:
                    team_name = str((event.get("team") or {}).get("name", "") or "")
                    player_name = str((event.get("player") or {}).get("name", "") or "")
                    elapsed = (event.get("time") or {}).get("elapsed", 0)
                    timeline_rows.append(
                        {
                            "Minuto": elapsed,
                            "Equipo": team_name,
                            "Jugador": player_name or "Jugador desconocido",
                            "Tipo": event.get("type", ""),
                            "Detalle": event.get("detail", ""),
                            "Comentario": event.get("comments", ""),
                        }
                    )
                st.dataframe(timeline_rows, use_container_width=True, hide_index=True)
        if statistics:
            with st.expander("Ver tabla técnica de estadísticas por equipo"):
                stats_rows = []
                for team_block in statistics:
                    team_name = str((team_block.get("team") or {}).get("name", "") or "Equipo")
                    for stat in team_block.get("statistics", []) or []:
                        stats_rows.append(
                            {
                                "Equipo": team_name,
                                "Métrica": stat.get("type", ""),
                                "Valor": stat.get("value", ""),
                            }
                        )
                st.dataframe(stats_rows, use_container_width=True, hide_index=True)
        if players:
            with st.expander("Ver detalle técnico de jugadores"):
                player_rows = []
                for team_block in players:
                    team_name = str((team_block.get("team") or {}).get("name", "") or "Equipo")
                    for player_block in team_block.get("players", []) or []:
                        player_info = player_block.get("player", {}) or {}
                        player_rows.append(
                            {
                                "Equipo": team_name,
                                "Jugador": player_info.get("name", ""),
                                "Edad": player_info.get("age", ""),
                                "Posición": player_info.get("pos", ""),
                                "Número": player_info.get("number", ""),
                            }
                        )
                st.dataframe(player_rows, use_container_width=True, hide_index=True)


def _build_team_and_player_options(
    canonical_events: list[dict[str, Any]],
    raw_payload: Any,
    selected_team_key: str,
    selected_player_key: str,
) -> tuple[str, str]:
    teams = _extract_teams_from_canonical(canonical_events)
    if not teams:
        teams = _extract_teams_from_payload(raw_payload)
    selected_team = _selectbox("Equipo", ["Todos"] + teams, key=selected_team_key)

    players = _extract_players_from_canonical(canonical_events, selected_team)
    if not players:
        players = _extract_players_from_payload(raw_payload, selected_team)
    selected_player = _selectbox("Jugador", ["Todos"] + players, key=selected_player_key)
    return selected_team, selected_player


def _render_statsbomb_provider(show_technical_info: bool = False) -> None:
    st.markdown("#### Configuración de StatsBomb Open Data")
    competitions = _cached_competitions()
    _render_fallback_warning("competitions")
    if not competitions:
        st.warning("No hay competiciones disponibles en este momento.")
        return

    selected_competition = _selectbox(
        "Competición / temporada",
        competitions,
        key="vertical2_api_competition",
        format_func=_competition_label,
    )
    competition_id = selected_competition.get("competition_id")
    season_id = selected_competition.get("season_id")

    matches = _cached_matches(competition_id, season_id)
    _render_fallback_warning("matches")
    if not matches:
        st.warning("No se encontraron partidos para la competición seleccionada.")
        return

    selected_match = _selectbox(
        "Partido",
        matches,
        key="vertical2_api_match",
        format_func=_match_label,
    )
    current_match_key = str(selected_match.get("match_id"))
    loaded_payload = st.session_state.get(SESSION_RESULT_KEY, {})
    current_canonical = loaded_payload.get("canonical_events", []) if _result_matches(loaded_payload, STORAGE_PROVIDER_STATSBOMB, current_match_key) else []
    current_raw_payload = loaded_payload.get("raw_payload", []) if _result_matches(loaded_payload, STORAGE_PROVIDER_STATSBOMB, current_match_key) else []
    selected_team, selected_player = _build_team_and_player_options(
        current_canonical,
        current_raw_payload,
        selected_team_key="vertical2_api_team",
        selected_player_key="vertical2_api_player",
    )

    try:
        already_processed = has_processed_match(STORAGE_PROVIDER_STATSBOMB, current_match_key)
    except Exception as exc:
        st.warning(f"No se pudo consultar el historial local: {exc}")
        already_processed = False

    if already_processed:
        st.info("Este partido ya existe en el historial local.")
        if st.button("Cargar desde historial local", key="vertical2_api_load_local_button", use_container_width=True):
            try:
                local_payload = load_processed_match_payloads(STORAGE_PROVIDER_STATSBOMB, current_match_key)
                if not local_payload:
                    st.warning("No fue posible cargar los archivos guardados localmente para este partido.")
                else:
                    raw_events = local_payload.get("raw_events", []) or []
                    st.session_state[SESSION_RESULT_KEY] = {
                        "provider": STORAGE_PROVIDER_STATSBOMB,
                        "match_id": current_match_key,
                        "competition_name": selected_competition.get("competition_name", "Competición"),
                        "season_name": selected_competition.get("season_name", ""),
                        "home_team": selected_match.get("home_team", ""),
                        "away_team": selected_match.get("away_team", ""),
                        "match_date": selected_match.get("match_date", ""),
                        "match_label": selected_match.get("display_name", "Partido"),
                        "raw_events_count": len(raw_events),
                        "raw_payload": raw_events,
                        "canonical_events": local_payload.get("canonical_events", []),
                        "used_fallback_events": False,
                        "events_status_message": "",
                        "loaded_from_local": True,
                        "precomputed_metrics": local_payload.get("metrics", {}),
                    }
                    st.rerun()
            except Exception as exc:
                st.warning(f"No se pudo cargar el partido desde historial local: {exc}")

    if st.button("Cargar datos", key="vertical2_api_load_button", use_container_width=True):
        raw_events = _cached_events(selected_match.get("match_id"))
        canonical_events = normalize_events_to_canonical(raw_events, match_id=selected_match.get("match_id"))
        precomputed_metrics = calculate_open_event_metrics(canonical_events)
        st.session_state[SESSION_RESULT_KEY] = {
            "provider": STORAGE_PROVIDER_STATSBOMB,
            "match_id": current_match_key,
            "competition_name": selected_competition.get("competition_name", "Competición"),
            "season_name": selected_competition.get("season_name", ""),
            "home_team": selected_match.get("home_team", ""),
            "away_team": selected_match.get("away_team", ""),
            "match_date": selected_match.get("match_date", ""),
            "match_label": selected_match.get("display_name", "Partido"),
            "raw_events_count": len(raw_events),
            "raw_payload": raw_events,
            "canonical_events": canonical_events,
            "used_fallback_events": get_ingestion_status().get("events", {}).get("source") == "mock",
            "events_status_message": get_ingestion_status().get("events", {}).get("message", ""),
            "loaded_from_local": False,
            "precomputed_metrics": precomputed_metrics,
        }
        try:
            save_processed_match(
                provider=STORAGE_PROVIDER_STATSBOMB,
                match_id=current_match_key,
                match_metadata={
                    "competition_name": selected_competition.get("competition_name", ""),
                    "season_name": selected_competition.get("season_name", ""),
                    "home_team": selected_match.get("home_team", ""),
                    "away_team": selected_match.get("away_team", ""),
                    "match_date": selected_match.get("match_date", ""),
                },
                raw_events=raw_events,
                canonical_events=canonical_events,
                metrics=precomputed_metrics,
            )
        except Exception as exc:
            st.warning(f"No se pudo guardar el partido en historial local: {exc}")
        st.rerun()

    _render_local_history(STORAGE_PROVIDER_STATSBOMB)

    result = st.session_state.get(SESSION_RESULT_KEY)
    if not _result_matches(result, STORAGE_PROVIDER_STATSBOMB, current_match_key):
        st.info("Configurá filtros y presioná 'Cargar datos' para ver métricas e insights.")
        return
    if result.get("used_fallback_events"):
        status_msg = result.get("events_status_message", "")
        detail = f" ({status_msg})" if status_msg else ""
        st.warning(f"Mostrando datos de fallback para eventos{detail}")

    _render_common_event_dashboard(result, selected_team, selected_player, show_technical_info=show_technical_info)


def _render_api_football_provider(show_technical_info: bool = False) -> None:
    st.markdown("#### Configuración de API-Football")
    api_key = get_api_football_api_key()
    if not api_key:
        st.warning("Falta configurar API_FOOTBALL_KEY en el entorno.")
        _render_local_history(STORAGE_PROVIDER_API_FOOTBALL)
        return

    countries = _cached_api_countries()
    if not countries:
        status = get_api_football_status()
        st.warning(status.get("message", "No se pudieron cargar países desde API-Football."))
        _render_local_history(STORAGE_PROVIDER_API_FOOTBALL)
        return

    argentina_index = _find_default_index(countries, lambda item: str(item.get("name", "")).lower() == "argentina")
    selected_country = _selectbox(
        "País",
        countries,
        key="vertical2_api_football_country",
        format_func=_country_label,
        index=argentina_index,
    )
    country_name = str(selected_country.get("name", "Argentina"))

    leagues = _cached_api_leagues(country=country_name, season=None, search=None)
    if not leagues:
        status = get_api_football_status()
        st.warning(status.get("message", "No se pudieron cargar ligas para el país seleccionado."))
        _render_local_history(STORAGE_PROVIDER_API_FOOTBALL)
        return

    selected_league = _selectbox(
        "Liga",
        leagues,
        key="vertical2_api_football_league",
        format_func=_league_label,
    )
    seasons = [
        season_item.get("year")
        for season_item in selected_league.get("seasons", []) or []
        if season_item.get("year") is not None
    ]
    seasons = sorted({int(season) for season in seasons}, reverse=True)
    if not seasons:
        st.warning("La liga seleccionada no informa temporadas disponibles.")
        _render_local_history(STORAGE_PROVIDER_API_FOOTBALL)
        return
    default_season = selected_league.get("current_season")
    season_index = _find_default_index(seasons, lambda season: season == default_season)
    selected_season = _selectbox(
        "Temporada",
        seasons,
        key="vertical2_api_football_season",
        index=season_index,
    )

    search_signature = f"{selected_league.get('league_id')}-{selected_season}"
    if st.button("Buscar partidos", key="vertical2_api_football_search_button", use_container_width=True):
        fixtures = _cached_api_fixtures(selected_league.get("league_id"), selected_season, None)
        st.session_state["vertical2_api_football_fixture_search"] = {
            "signature": search_signature,
            "fixtures": fixtures,
            "status": get_api_football_status(),
        }
        st.rerun()

    fixture_search = st.session_state.get("vertical2_api_football_fixture_search", {})
    fixtures = fixture_search.get("fixtures", []) if fixture_search.get("signature") == search_signature else []
    search_status = fixture_search.get("status", {}) if fixture_search.get("signature") == search_signature else {}
    if not fixtures:
        if search_status.get("status") == "error":
            st.warning(get_api_football_user_message(search_status))
        else:
            st.info("Seleccioná una liga y temporada, luego presioná 'Buscar partidos' para cargar partidos disponibles.")
        _render_local_history(STORAGE_PROVIDER_API_FOOTBALL)
        return

    selected_fixture = _selectbox(
        "Partido",
        fixtures,
        key="vertical2_api_football_fixture",
        format_func=_fixture_label,
    )
    current_match_key = str(selected_fixture.get("fixture_id"))
    loaded_payload = st.session_state.get(SESSION_RESULT_KEY, {})
    current_canonical = loaded_payload.get("canonical_events", []) if _result_matches(loaded_payload, STORAGE_PROVIDER_API_FOOTBALL, current_match_key) else []
    current_raw_payload = loaded_payload.get("raw_payload", {}) if _result_matches(loaded_payload, STORAGE_PROVIDER_API_FOOTBALL, current_match_key) else {}
    selected_team, selected_player = _build_team_and_player_options(
        current_canonical,
        current_raw_payload,
        selected_team_key="vertical2_api_football_team",
        selected_player_key="vertical2_api_football_player",
    )

    try:
        already_processed = has_processed_match(STORAGE_PROVIDER_API_FOOTBALL, current_match_key)
    except Exception as exc:
        st.warning(f"No se pudo consultar el historial local: {exc}")
        already_processed = False

    if already_processed:
        st.info("Este partido ya existe en el historial local.")
        if st.button("Cargar desde historial local", key="vertical2_api_football_load_local_button", use_container_width=True):
            try:
                local_payload = load_processed_match_payloads(STORAGE_PROVIDER_API_FOOTBALL, current_match_key)
                if not local_payload:
                    st.warning("No fue posible cargar los archivos guardados localmente para este partido.")
                else:
                    raw_payload = local_payload.get("raw_events", {}) or {}
                    st.session_state[SESSION_RESULT_KEY] = {
                        "provider": STORAGE_PROVIDER_API_FOOTBALL,
                        "match_id": current_match_key,
                        "competition_name": selected_league.get("league_name", "Liga"),
                        "season_name": str(selected_season),
                        "home_team": selected_fixture.get("home_team", ""),
                        "away_team": selected_fixture.get("away_team", ""),
                        "match_date": selected_fixture.get("match_date", ""),
                        "match_label": selected_fixture.get("display_name", "Partido"),
                        "raw_events_count": len((raw_payload.get("events", []) if isinstance(raw_payload, dict) else [])),
                        "raw_payload": raw_payload,
                        "canonical_events": local_payload.get("canonical_events", []),
                        "loaded_from_local": True,
                        "precomputed_metrics": local_payload.get("metrics", {}),
                    }
                    st.rerun()
            except Exception as exc:
                st.warning(f"No se pudo cargar el partido desde historial local: {exc}")

    status = get_api_football_status()
    if status.get("status") == "error":
        st.warning(get_api_football_user_message(status))

    if st.button("Cargar datos", key="vertical2_api_football_load_button", use_container_width=True):
        fixture_id = selected_fixture.get("fixture_id")
        raw_events = _cached_api_fixture_events(fixture_id)
        lineups = _cached_api_fixture_lineups(fixture_id)
        statistics = _cached_api_fixture_statistics(fixture_id)
        players = _cached_api_fixture_players(fixture_id)
        canonical_events = normalize_api_football_events_to_canonical(raw_events, fixture_id=fixture_id)
        precomputed_metrics = calculate_open_event_metrics(canonical_events)
        raw_payload = {
            "events": raw_events,
            "lineups": lineups,
            "statistics": statistics,
            "players": players,
        }
        st.session_state[SESSION_RESULT_KEY] = {
            "provider": STORAGE_PROVIDER_API_FOOTBALL,
            "match_id": current_match_key,
            "competition_name": selected_league.get("league_name", "Liga"),
            "season_name": str(selected_season),
            "home_team": selected_fixture.get("home_team", ""),
            "away_team": selected_fixture.get("away_team", ""),
            "match_date": selected_fixture.get("match_date", ""),
            "match_label": selected_fixture.get("display_name", "Partido"),
            "raw_events_count": len(raw_events),
            "raw_payload": raw_payload,
            "canonical_events": canonical_events,
            "loaded_from_local": False,
            "precomputed_metrics": precomputed_metrics,
        }
        try:
            save_processed_match(
                provider=STORAGE_PROVIDER_API_FOOTBALL,
                match_id=current_match_key,
                match_metadata={
                    "competition_name": selected_league.get("league_name", ""),
                    "season_name": str(selected_season),
                    "home_team": selected_fixture.get("home_team", ""),
                    "away_team": selected_fixture.get("away_team", ""),
                    "match_date": selected_fixture.get("match_date", ""),
                },
                raw_events=raw_payload,
                canonical_events=canonical_events,
                metrics=precomputed_metrics,
            )
        except Exception as exc:
            st.warning(f"No se pudo guardar el partido en historial local: {exc}")
        st.rerun()

    _render_local_history(STORAGE_PROVIDER_API_FOOTBALL)

    result = st.session_state.get(SESSION_RESULT_KEY)
    if not _result_matches(result, STORAGE_PROVIDER_API_FOOTBALL, current_match_key):
        st.info("Seleccioná país, liga, temporada y partido; luego presioná 'Cargar datos'.")
        return

    _render_api_football_provider_sections(result, show_technical_info=show_technical_info)
    _render_common_event_dashboard(result, selected_team, selected_player, show_technical_info=show_technical_info)


def _render_sportmonks_expected_metrics(expected_metrics: list[Any]) -> None:
    st.markdown("### Rendimiento esperado")
    if not expected_metrics:
        st.info("No hay métricas esperadas disponibles para este partido.")
        return

    metric_rows = build_expected_metrics_table(expected_metrics)
    team_names = [str(_object_value(item, "team_name", "") or f"Equipo {index + 1}") for index, item in enumerate(expected_metrics)]
    for row in metric_rows:
        st.markdown(f"**{row.get('Métrica', 'Métrica')}**")
        metric_columns = st.columns(max(1, len(team_names)))
        for index, team_name in enumerate(team_names):
            with metric_columns[index]:
                st.metric(team_name, row.get(team_name, "No disponible"))


def _render_sportmonks_timeline(timeline_events: list[Any]) -> None:
    st.markdown("### Timeline de eventos")
    if not timeline_events:
        st.info("No hay eventos principales disponibles para este partido.")
        return

    event_filter = _selectbox(
        "Filtro de timeline",
        ["Todos", "Goles", "Tarjetas", "Cambios", "VAR"],
        key="vertical2_sportmonks_timeline_filter",
    )
    timeline_rows = build_timeline_table(timeline_events, event_filter=event_filter)
    if not timeline_rows:
        st.info("No hay eventos principales disponibles para el filtro seleccionado.")
        return
    st.dataframe(timeline_rows, use_container_width=True, hide_index=True)


def _render_sportmonks_lineups(lineups: list[Any]) -> None:
    st.markdown("### Lineups")
    if not lineups:
        st.info("No hay alineaciones disponibles para este partido.")
        return

    lineup_tabs = st.tabs([str(_object_value(item, "team_name", "") or f"Equipo {index + 1}") for index, item in enumerate(lineups)])
    for index, lineup in enumerate(lineups):
        with lineup_tabs[index]:
            st.caption(f"Formación: {_object_value(lineup, 'formation', 'No disponible') or 'No disponible'}")
            st.caption(f"Coach: {_object_value(lineup, 'coach', 'No disponible') or 'No disponible'}")
            starters = _object_value(lineup, "starters", []) or []
            substitutes = _object_value(lineup, "substitutes", []) or []
            st.markdown("**Titulares**")
            if starters:
                st.dataframe(
                    [
                        {
                            "Dorsal": item.get("jersey_number", "No disponible"),
                            "Jugador": item.get("player_name", "No disponible"),
                            "Posición": item.get("position", "No disponible"),
                            "Minutos": item.get("minutes_played", "No disponible"),
                            "Rating": item.get("rating", "No disponible"),
                        }
                        for item in starters
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No hay titulares disponibles para este equipo.")
            st.markdown("**Suplentes**")
            if substitutes:
                st.dataframe(
                    [
                        {
                            "Dorsal": item.get("jersey_number", "No disponible"),
                            "Jugador": item.get("player_name", "No disponible"),
                            "Posición": item.get("position", "No disponible"),
                            "Minutos": item.get("minutes_played", "No disponible"),
                            "Rating": item.get("rating", "No disponible"),
                        }
                        for item in substitutes
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No hay suplentes disponibles para este equipo.")


def _render_sportmonks_team_stats(team_stats: list[Any]) -> None:
    st.markdown("### Estadísticas de equipo")
    if not team_stats:
        st.info("No hay estadísticas de equipo disponibles para este partido.")
        return

    stats_tabs = st.tabs([str(_object_value(item, "team_name", "") or f"Equipo {index + 1}") for index, item in enumerate(team_stats)])
    for index, item in enumerate(team_stats):
        with stats_tabs[index]:
            st.dataframe(build_team_stats_table(item), use_container_width=True, hide_index=True)


def _render_sportmonks_player_analysis(player_stats: list[Any]) -> None:
    st.markdown("### Análisis de jugador")
    if not player_stats:
        st.info("No hay estadísticas de jugador disponibles para este partido.")
        return

    selected_player = _selectbox(
        "Jugador para análisis",
        player_stats,
        key="vertical2_sportmonks_player_selector",
        format_func=lambda item: str(_object_value(item, "player_name", "Jugador")) + " — " + str(_object_value(item, "team_name", "Equipo")),
    )
    info_columns = st.columns(6)
    with info_columns[0]:
        st.metric("Jugador", str(_object_value(selected_player, "player_name", "No disponible")))
    with info_columns[1]:
        st.metric("Equipo", str(_object_value(selected_player, "team_name", "No disponible")))
    with info_columns[2]:
        st.metric("Posición", str(_object_value(selected_player, "position", "No disponible") or "No disponible"))
    with info_columns[3]:
        st.metric("Dorsal", str(_object_value(selected_player, "jersey_number", "No disponible") or "No disponible"))
    with info_columns[4]:
        st.metric("Minutos", str(_object_value(selected_player, "minutes_played", "No disponible") or "No disponible"))
    with info_columns[5]:
        st.metric("Rating", _render_stat_value(_object_value(selected_player, "rating")))

    stats_table = build_player_stats_table(selected_player)
    if stats_table:
        st.dataframe(stats_table, use_container_width=True, hide_index=True)

    insights = build_sportmonks_player_insights(selected_player)
    if insights:
        st.markdown("**Insights simples**")
        for insight in insights:
            st.markdown(f"- {insight}")


def _render_sportmonks_availability(availability: Any) -> None:
    with st.expander("Fuentes y calidad de datos"):
        availability_rows = [
            {"Campo": "Fuente", "Estado": "Sportmonks"},
            {"Campo": "Timeline de eventos", "Estado": format_availability_status(_object_value(availability, "has_event_timeline", False))},
            {"Campo": "Lineups", "Estado": format_availability_status(_object_value(availability, "has_lineups", False))},
            {"Campo": "Estadísticas de equipo", "Estado": format_availability_status(_object_value(availability, "has_team_stats", False))},
            {"Campo": "Estadísticas de jugador", "Estado": format_availability_status(_object_value(availability, "has_player_stats", False))},
            {"Campo": "xG", "Estado": format_availability_status(_object_value(availability, "has_xg", False))},
            {"Campo": "xGoT", "Estado": format_availability_status(_object_value(availability, "has_xgot", False))},
            {"Campo": "xPTS", "Estado": format_availability_status(_object_value(availability, "has_xpts", False))},
            {"Campo": "Coordenadas de eventos", "Estado": format_availability_status(_object_value(availability, "has_coordinates", False))},
            {"Campo": "Tracking", "Estado": format_availability_status(_object_value(availability, "has_tracking", False))},
        ]
        st.dataframe(availability_rows, use_container_width=True, hide_index=True)
        st.warning(
            "Este partido no incluye coordenadas de eventos desde Sportmonks. "
            "Por eso los mapas tácticos espaciales no están disponibles."
        )


def _render_sportmonks_visualization_notice(availability: Any) -> None:
    st.markdown("### Visualizaciones tácticas")
    if not bool(_object_value(availability, "has_coordinates", False)):
        st.info(
            "Las visualizaciones de cancha no están disponibles para Sportmonks con los datos actuales "
            "porque no se confirmaron coordenadas de eventos."
        )
        return
    st.info("La visualización espacial para Sportmonks quedará disponible cuando se confirme soporte estable de coordenadas.")


def _render_sportmonks_match_center(result: dict[str, Any], show_technical_info: bool = False) -> None:
    context = result.get("sportmonks_context", {}) or {}
    match = context.get("match")
    expected_metrics = context.get("expected_metrics", []) or []
    timeline_events = context.get("timeline_events", []) or []
    lineups = context.get("lineups", []) or []
    team_stats = context.get("team_stats", []) or []
    player_stats = context.get("player_stats", []) or []
    availability = context.get("availability")

    if match is None:
        st.warning("No se pudo construir el Match Center técnico para este partido.")
        return

    st.markdown("### Sportmonks — Match Center técnico")
    st.subheader(format_sportmonks_match_title(match))
    st.caption(
        f"{_object_value(match, 'competition_name', 'Competición no informada') or 'Competición no informada'} | "
        f"{_object_value(match, 'season_name', 'Temporada no informada') or 'Temporada no informada'}"
    )

    st.markdown("### Resumen del partido")
    summary_columns = st.columns(5)
    result_text = format_sportmonks_match_title(match)
    with summary_columns[0]:
        st.metric("Resultado", result_text)
    with summary_columns[1]:
        st.metric("Estado", str(_object_value(match, "status", "No disponible") or "No disponible"))
    with summary_columns[2]:
        st.metric("Competición", str(_object_value(match, "competition_name", "No disponible") or "No disponible"))
    with summary_columns[3]:
        venue_text = str(_object_value(match, "venue_name", "No disponible") or "No disponible")
        if _object_value(match, "venue_city"):
            venue_text = f"{venue_text} ({_object_value(match, 'venue_city')})"
        st.metric("Estadio", venue_text)
    with summary_columns[4]:
        st.metric("Fecha", str(_object_value(match, "match_date", "No disponible") or "No disponible"))

    _render_sportmonks_expected_metrics(expected_metrics)
    _render_sportmonks_timeline(timeline_events)
    _render_sportmonks_lineups(lineups)
    _render_sportmonks_team_stats(team_stats)
    _render_sportmonks_player_analysis(player_stats)
    _render_sportmonks_availability(availability)
    _render_sportmonks_visualization_notice(availability)

    if show_technical_info:
        with st.expander("Vista técnica del Match Center (canónico)"):
            st.json(
                {
                    "match": _object_as_dict(match),
                    "availability": _object_as_dict(availability),
                    "timeline_events": [_object_as_dict(item) for item in timeline_events[:20]],
                    "lineups": [_object_as_dict(item) for item in lineups[:2]],
                }
            )


def _render_sportmonks_provider(show_technical_info: bool = False) -> None:
    st.markdown("#### Configuración de Sportmonks")
    if not is_sportmonks_configured():
        st.warning(
            "Sportmonks no está configurado. Agregá SPORTMONKS_API_KEY en el archivo .env "
            "o en las variables de entorno para usar este proveedor."
        )
        return

    default_date = date.today().isoformat()
    selected_date = _date_input("Fecha del partido", default_date, key="vertical2_sportmonks_date")
    if st.button("Buscar partidos", key="vertical2_sportmonks_search_button", use_container_width=True):
        fixtures_response = _cached_sportmonks_fixtures_by_date(selected_date)
        st.session_state["vertical2_sportmonks_fixture_search"] = {
            "date": selected_date,
            "response": fixtures_response,
        }
        st.rerun()

    search_state = st.session_state.get("vertical2_sportmonks_fixture_search", {})
    search_response = search_state.get("response", {}) if search_state.get("date") == selected_date else {}
    fixtures = search_response.get("data", []) or []
    if not fixtures:
        if search_response.get("ok") is False:
            st.error(f"No se pudo cargar información desde Sportmonks: {search_response.get('error')}")
        else:
            st.info("Seleccioná una fecha y presioná 'Buscar partidos' para cargar partidos disponibles.")
        return

    selected_fixture = _selectbox(
        "Partido",
        fixtures,
        key="vertical2_sportmonks_fixture",
        format_func=_sportmonks_fixture_label,
    )
    current_match_key = str(_object_value(selected_fixture, "provider_match_id", "") or "")

    st.caption(f"Fixture interno Sportmonks: #{current_match_key}")

    if st.button("Cargar análisis del partido", key="vertical2_sportmonks_load_button", use_container_width=True):
        context_response = _cached_sportmonks_match_context(current_match_key)
        if context_response.get("ok"):
            data = context_response.get("data", {}) or {}
            availability_response = _cached_sportmonks_availability(current_match_key)
            availability = data.get("availability")
            if availability_response.get("ok") and availability_response.get("data") is not None:
                availability = availability_response.get("data")
                data["availability"] = availability
            match = data.get("match")
            st.session_state[SESSION_RESULT_KEY] = {
                "provider": STORAGE_PROVIDER_SPORTMONKS,
                "match_id": current_match_key,
                "competition_name": _object_value(match, "competition_name", "") if match else "",
                "season_name": _object_value(match, "season_name", "") if match else "",
                "home_team": _object_value(match, "home_team_name", "") if match else "",
                "away_team": _object_value(match, "away_team_name", "") if match else "",
                "match_date": _object_value(match, "match_date", "") if match else "",
                "match_label": format_sportmonks_match_title(match) if match else "Partido",
                "raw_payload": {},
                "canonical_events": [],
                "sportmonks_context": data,
            }
            st.rerun()
            return
        st.error(f"No se pudo cargar información desde Sportmonks: {context_response.get('error')}")

    result = st.session_state.get(SESSION_RESULT_KEY)
    if not _result_matches(result, STORAGE_PROVIDER_SPORTMONKS, current_match_key):
        st.info("Buscá partidos y luego presioná 'Cargar análisis del partido' para ver el Match Center técnico.")
        return

    _render_sportmonks_match_center(result, show_technical_info=show_technical_info)


def render_vertical2_api_event() -> None:
    st.subheader("API Event Data")
    st.caption("Conectá datos de eventos desde proveedores externos para generar métricas tácticas propietarias.")
    show_technical_info = st.checkbox("Mostrar información técnica", value=False)
    _render_environment_config_status(show_technical_info=show_technical_info)
    try:
        persistence_status = initialize_event_data_persistence()
        if show_technical_info:
            st.caption(
                "Persistencia activa: "
                f"{persistence_status['persistence_backend']} / storage {persistence_status['storage_backend']}"
            )
    except Exception as exc:
        st.warning(f"No se pudo inicializar la persistencia configurada: {exc}")

    provider = _selectbox(
        "Proveedor de datos",
        ["StatsBomb Open Data", "API-Football", "Sportmonks"],
        key="vertical2_api_provider",
    )

    if provider == "StatsBomb Open Data":
        _render_statsbomb_provider(show_technical_info=show_technical_info)
        return

    if provider == "API-Football":
        _render_api_football_provider(show_technical_info=show_technical_info)
        return

    _render_sportmonks_provider(show_technical_info=show_technical_info)
