from __future__ import annotations

import json
from typing import Any, Callable

import streamlit as st

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
from src.services.storage.database import initialize_event_data_db
from src.services.storage.event_data_repository import get_processed_matches
from src.services.storage.event_data_repository import has_processed_match
from src.services.storage.event_data_repository import load_processed_match_payloads
from src.services.storage.event_data_repository import save_processed_match

STORAGE_PROVIDER = "statsbomb"


def _selectbox(
    label: str,
    options: list[Any],
    key: str,
    format_func: Callable[[Any], str] | None = None,
) -> Any:
    selectbox_fn = getattr(st, "selectbox", None)
    kwargs: dict[str, Any] = {"key": key}
    if format_func is not None:
        kwargs["format_func"] = format_func

    if callable(selectbox_fn):
        return selectbox_fn(label, options, **kwargs)
    return st.sidebar.selectbox(label, options, **kwargs)


def _competition_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", "Competición - Temporada"))


def _match_label(item: dict[str, Any]) -> str:
    return str(item.get("display_name", "Partido sin nombre"))


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


def _extract_teams_from_canonical(canonical_events: list[dict[str, Any]]) -> list[str]:
    teams = sorted(
        {
            str(event.get("team_name"))
            for event in canonical_events
            if event.get("team_name") and str(event.get("team_name")).strip()
        }
    )
    return teams


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


def render_vertical2_api_event() -> None:
    st.subheader("Datos por API")
    st.caption("Conectá event data desde proveedores externos para generar métricas tácticas propietarias.")
    try:
        initialize_event_data_db()
    except Exception as exc:
        st.warning(f"No se pudo inicializar la persistencia local: {exc}")

    provider = _selectbox(
        "Proveedor de datos",
        ["StatsBomb Open Data", "API-Football (próximamente)"],
        key="vertical2_api_provider",
    )

    if provider == "API-Football (próximamente)":
        st.info("Este proveedor estará disponible próximamente.")
        return

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
    match_id = selected_match.get("match_id")
    current_match_key = str(match_id)

    loaded_payload = st.session_state.get("vertical2_api_event_result", {})
    canonical_events_loaded = []
    if isinstance(loaded_payload, dict) and str(loaded_payload.get("match_id")) == current_match_key:
        canonical_events_loaded = loaded_payload.get("canonical_events", []) or []

    team_options = ["Todos"] + _extract_teams_from_canonical(canonical_events_loaded)
    selected_team = _selectbox("Equipo", team_options, key="vertical2_api_team")

    player_options = ["Todos"] + _extract_players_from_canonical(canonical_events_loaded, selected_team)
    selected_player = _selectbox("Jugador", player_options, key="vertical2_api_player")

    already_processed = False
    try:
        already_processed = has_processed_match(STORAGE_PROVIDER, current_match_key)
    except Exception as exc:
        st.warning(f"No se pudo consultar el historial local: {exc}")

    if already_processed:
        st.info("Este partido ya existe en el historial local.")
        if st.button("Cargar desde historial local", key="vertical2_api_load_local_button", use_container_width=True):
            try:
                local_payload = load_processed_match_payloads(STORAGE_PROVIDER, current_match_key)
                if not local_payload:
                    st.warning("No fue posible cargar los archivos guardados localmente para este partido.")
                else:
                    st.session_state["vertical2_api_event_result"] = {
                        "match_id": current_match_key,
                        "competition_name": selected_competition.get("competition_name", "Competición"),
                        "match_label": selected_match.get("display_name", "Partido"),
                        "raw_events_count": len(local_payload.get("raw_events", [])),
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
        raw_events = _cached_events(match_id)
        canonical_events = normalize_events_to_canonical(raw_events, match_id=match_id)
        precomputed_metrics = calculate_open_event_metrics(canonical_events)
        st.session_state["vertical2_api_event_result"] = {
            "match_id": current_match_key,
            "competition_name": selected_competition.get("competition_name", "Competición"),
            "match_label": selected_match.get("display_name", "Partido"),
            "raw_events_count": len(raw_events),
            "canonical_events": canonical_events,
            "used_fallback_events": get_ingestion_status().get("events", {}).get("source") == "mock",
            "events_status_message": get_ingestion_status().get("events", {}).get("message", ""),
            "loaded_from_local": False,
            "precomputed_metrics": precomputed_metrics,
        }
        try:
            save_processed_match(
                provider=STORAGE_PROVIDER,
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

    st.markdown("### Historial local")
    try:
        history_rows = get_processed_matches(limit=20)
    except Exception as exc:
        st.warning(f"No se pudo cargar el historial local: {exc}")
        history_rows = []
    if not history_rows:
        st.info("Todavía no hay partidos guardados localmente.")
    else:
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

    result = st.session_state.get("vertical2_api_event_result")
    if not isinstance(result, dict) or str(result.get("match_id")) != current_match_key:
        st.info("Configurá filtros y presioná 'Cargar datos' para ver métricas e insights.")
        return

    canonical_events = result.get("canonical_events", [])
    if result.get("used_fallback_events"):
        status_msg = result.get("events_status_message", "")
        detail = f" ({status_msg})" if status_msg else ""
        st.warning(f"Mostrando datos de fallback para eventos{detail}")

    team_filter = None if selected_team == "Todos" else selected_team
    player_filter = None if selected_player == "Todos" else selected_player
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
            title="Field Tilt",
            value=metrics.get("field_tilt_index"),
            label=str(metrics.get("field_tilt_label", "No aplica")),
            description="Dominio territorial del equipo en acciones del último tercio.",
        )
    with prop_col2:
        _render_proprietary_metric_card(
            title="Directness",
            value=metrics.get("directness_index"),
            label=str(metrics.get("directness_label", "No aplica")),
            description="Relación entre progresiones y volumen de pases.",
        )
    with prop_col3:
        _render_proprietary_metric_card(
            title="Amenaza progresiva",
            value=metrics.get("progressive_threat_index"),
            label=str(metrics.get("progressive_threat_label", "No aplica")),
            description="Peligro combinado por progresiones, último tercio, remate y xG.",
        )

    prop_row2_col1, prop_row2_col2 = st.columns(2)
    with prop_row2_col1:
        _render_proprietary_metric_card(
            title="Altura de recuperación",
            value=metrics.get("recovery_height_index"),
            label=str(metrics.get("recovery_height_label", "No aplica")),
            description="Promedio de altura donde se recupera el balón (x sobre 120m).",
        )
    with prop_row2_col2:
        _render_proprietary_metric_card(
            title="Calidad de remate",
            value=metrics.get("shot_quality_index"),
            label=str(metrics.get("shot_quality_label", "No aplica")),
            description="xG promedio por remate, expresado en escala 0-100.",
        )
    if selected_player and selected_player != "Todos":
        _render_proprietary_metric_card(
            title="Influencia del jugador",
            value=metrics.get("player_influence_score"),
            label=str(metrics.get("player_influence_label", "No aplica")),
            description="Aporte combinado del jugador en volumen, progresión, amenaza y recuperación.",
        )

    st.markdown("### Visualizaciones tácticas")
    map_tab, shots_tab, progressive_tab, recoveries_tab = st.tabs(
        ["Mapa de eventos", "Remates", "Progresiones", "Recuperaciones"]
    )
    with map_tab:
        st.plotly_chart(
            create_event_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
            use_container_width=True,
        )
    with shots_tab:
        st.plotly_chart(
            create_shot_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
            use_container_width=True,
        )
    with progressive_tab:
        st.plotly_chart(
            create_progressive_actions_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
            use_container_width=True,
        )
    with recoveries_tab:
        st.plotly_chart(
            create_recoveries_map(canonical_events, selected_team=team_filter, selected_player=player_filter),
            use_container_width=True,
        )

    st.markdown("#### Insights iniciales")
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No hay insights disponibles para esta selección.")

    st.markdown("### Análisis de jugador")
    if selected_player and selected_player != "Todos":
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
        st.plotly_chart(
            create_player_action_map(canonical_events, selected_player=selected_player),
            use_container_width=True,
        )
    else:
        st.info("Seleccioná un jugador específico para ver su mapa de acciones.")

    st.markdown("### Modelo canónico")
    with st.expander("Ver preview del Canonical Event Model (JSON)"):
        canonical_preview = canonical_events[:10]
        st.code(json.dumps(canonical_preview, indent=2, ensure_ascii=False), language="json")
