from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from src.services.open_event_metrics import filter_analytical_events
from src.utils.ui.theme import apply_plotly_dark_theme


def _apply_event_filters(
    canonical_events: list[dict[str, Any]],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in filter_analytical_events(canonical_events):
        if selected_team and event.get("team_name") != selected_team:
            continue
        if selected_player and event.get("player_name") != selected_player:
            continue
        events.append(event)
    return events


def _has_xy(event: dict[str, Any]) -> bool:
    return isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))


def _hover_text(event: dict[str, Any]) -> str:
    xg = float(event.get("xG", 0.0) or 0.0)
    return (
        f"Equipo: {event.get('team_name', 'N/D')}<br>"
        f"Jugador: {event.get('player_name', 'N/D')}<br>"
        f"Minuto: {event.get('minute', 0)}:{int(event.get('second', 0)):02d}<br>"
        f"Evento: {event.get('event_type', 'N/D')}<br>"
        f"Outcome: {event.get('outcome', 'N/D')}<br>"
        f"xG: {xg:.3f}"
    )


def create_pitch_figure(title: str | None = None) -> go.Figure:
    fig = go.Figure()
    # Marco del campo
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=120,
        y1=80,
        line=dict(color="#e5e7eb", width=2),
        fillcolor="#14532d",
        layer="below",
    )
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="#e5e7eb", width=2), layer="below")
    fig.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="#e5e7eb", width=2), layer="below")
    # Áreas y arcos
    fig.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="#e5e7eb", width=2), layer="below")
    fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="#e5e7eb", width=2), layer="below")
    fig.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="#e5e7eb", width=2), layer="below")
    fig.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="#e5e7eb", width=2), layer="below")
    # Arcos simples fuera de línea
    fig.add_shape(type="line", x0=-1.2, y0=36, x1=-1.2, y1=44, line=dict(color="#e5e7eb", width=3), layer="below")
    fig.add_shape(type="line", x0=121.2, y0=36, x1=121.2, y1=44, line=dict(color="#e5e7eb", width=3), layer="below")
    # Flecha de orientación
    fig.add_annotation(x=110, y=76, text="Ataque ->", showarrow=False, font=dict(color="#f8fafc", size=12))

    fig.update_xaxes(range=[-2, 122], visible=False)
    fig.update_yaxes(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title or "",
        height=500,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        showlegend=True,
    )
    apply_plotly_dark_theme(fig)
    fig.update_layout(paper_bgcolor="#0f131a", plot_bgcolor="#14532d")
    return fig


def create_event_map(
    canonical_events: list[dict[str, Any]],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> go.Figure:
    fig = create_pitch_figure("Mapa de Eventos")
    events = [event for event in _apply_event_filters(canonical_events, selected_team, selected_player) if _has_xy(event)]
    if not events:
        fig.add_annotation(x=60, y=40, text="Sin eventos con ubicación para mostrar", showarrow=False, font=dict(size=14))
        return fig

    style_map = {
        "Pass": ("#38bdf8", "circle", 7),
        "Shot": ("#ef4444", "diamond", 10),
        "Carry": ("#a78bfa", "triangle-up", 8),
        "Ball Recovery": ("#22c55e", "square", 9),
        "Interception": ("#f59e0b", "x", 9),
        "Duel": ("#fb7185", "cross", 9),
    }
    for event_type, (color, symbol, size) in style_map.items():
        subset = [event for event in events if event.get("event_type") == event_type]
        if not subset:
            continue
        fig.add_trace(
            go.Scatter(
                x=[event["x"] for event in subset],
                y=[event["y"] for event in subset],
                mode="markers",
                name=event_type,
                marker=dict(size=size, color=color, symbol=symbol, opacity=0.85, line=dict(width=1, color="#0f172a")),
                hovertemplate="%{text}<extra></extra>",
                text=[_hover_text(event) for event in subset],
            )
        )
    return fig


def create_shot_map(
    canonical_events: list[dict[str, Any]],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> go.Figure:
    fig = create_pitch_figure("Mapa de Remates")
    events = _apply_event_filters(canonical_events, selected_team, selected_player)
    shots = [event for event in events if event.get("event_type") == "Shot" and _has_xy(event)]
    if not shots:
        fig.add_annotation(x=60, y=40, text="No hay remates para esta selección", showarrow=False, font=dict(size=14))
        return fig

    sizes = [max(8.0, min(28.0, 8.0 + float(event.get("xG", 0.0) or 0.0) * 50.0)) for event in shots]
    fig.add_trace(
        go.Scatter(
            x=[event["x"] for event in shots],
            y=[event["y"] for event in shots],
            mode="markers",
            name="Remates",
            marker=dict(size=sizes, color="#ef4444", symbol="diamond", opacity=0.85, line=dict(width=1, color="#fee2e2")),
            hovertemplate="%{text}<extra></extra>",
            text=[_hover_text(event) for event in shots],
        )
    )
    return fig


def create_progressive_actions_map(
    canonical_events: list[dict[str, Any]],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> go.Figure:
    fig = create_pitch_figure("Acciones Progresivas")
    events = _apply_event_filters(canonical_events, selected_team, selected_player)
    progressive = [
        event
        for event in events
        if bool(event.get("progressive", False))
        and event.get("event_type") in {"Pass", "Carry"}
        and _has_xy(event)
        and isinstance(event.get("end_x"), (int, float))
        and isinstance(event.get("end_y"), (int, float))
    ]
    if not progressive:
        fig.add_annotation(x=60, y=40, text="No hay progresiones para esta selección", showarrow=False, font=dict(size=14))
        return fig

    for idx, event in enumerate(progressive[:800]):
        color = "#38bdf8" if event.get("event_type") == "Pass" else "#a78bfa"
        fig.add_trace(
            go.Scatter(
                x=[event["x"], event["end_x"]],
                y=[event["y"], event["end_y"]],
                mode="lines+markers",
                showlegend=idx == 0,
                name="Progresión",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                hovertemplate="%{text}<extra></extra>",
                text=[_hover_text(event), _hover_text(event)],
            )
        )
    return fig


def create_recoveries_map(
    canonical_events: list[dict[str, Any]],
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> go.Figure:
    fig = create_pitch_figure("Mapa de Recuperaciones")
    events = _apply_event_filters(canonical_events, selected_team, selected_player)
    recoveries = []
    for event in events:
        if not _has_xy(event):
            continue
        event_type = str(event.get("event_type", ""))
        outcome = str(event.get("outcome", "") or "").lower()
        if event_type in {"Ball Recovery", "Interception"}:
            recoveries.append(event)
        elif event_type == "Duel" and any(token in outcome for token in ("won", "success", "tackle")):
            recoveries.append(event)
    if not recoveries:
        fig.add_annotation(x=60, y=40, text="No hay recuperaciones para esta selección", showarrow=False, font=dict(size=14))
        return fig

    fig.add_trace(
        go.Scatter(
            x=[event["x"] for event in recoveries],
            y=[event["y"] for event in recoveries],
            mode="markers",
            name="Recuperaciones",
            marker=dict(size=9, color="#22c55e", symbol="square", opacity=0.9, line=dict(width=1, color="#052e16")),
            hovertemplate="%{text}<extra></extra>",
            text=[_hover_text(event) for event in recoveries],
        )
    )
    return fig


def create_player_action_map(canonical_events: list[dict[str, Any]], selected_player: str | None) -> go.Figure:
    if not selected_player or selected_player == "Todos":
        fig = create_pitch_figure("Mapa de Acciones del Jugador")
        fig.add_annotation(
            x=60,
            y=40,
            text="Seleccioná un jugador específico para ver su mapa de acciones.",
            showarrow=False,
            font=dict(size=13),
        )
        return fig
    return create_event_map(canonical_events, selected_player=selected_player)
