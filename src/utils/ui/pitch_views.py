from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from src.utils.ui.theme import apply_plotly_dark_theme


def _get_signal(normalized_payload: dict[str, Any], section: str, key: str) -> float:
    section_data = normalized_payload.get(section, {})
    signals = section_data.get("signals", {})
    try:
        return float(signals.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, alpha)):.2f})"


def _zone_alpha(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.18
    ratio = max(0.0, min(1.0, value / max_value))
    return 0.18 + ratio * 0.5


def _base_pitch_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color="#e5e7eb", width=2), fillcolor="#14532d")
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="#e5e7eb", width=2))
    fig.add_shape(type="circle", x0=43.5, y0=25, x1=61.5, y1=43, line=dict(color="#e5e7eb", width=2))
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="#e5e7eb", width=2))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#e5e7eb", width=2))
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="#e5e7eb", width=2))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#e5e7eb", width=2))
    fig.update_xaxes(range=[0, 105], visible=False)
    fig.update_yaxes(range=[0, 68], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    apply_plotly_dark_theme(fig)
    fig.update_layout(paper_bgcolor="#0f131a", plot_bgcolor="#14532d")
    return fig


def _view_content(view_key: str, normalized_payload: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    attack_signals = {
        "crosses": _get_signal(normalized_payload, "attack", "crosses"),
        "box_entries": _get_signal(normalized_payload, "attack", "box entries"),
        "final_third": _get_signal(normalized_payload, "attack", "final third"),
        "shots": _get_signal(normalized_payload, "attack", "shots"),
        "xg": _get_signal(normalized_payload, "attack", "xg"),
    }
    defense_signals = {
        "duels": _get_signal(normalized_payload, "defense", "duels"),
        "pressing": _get_signal(normalized_payload, "defense", "pressing"),
        "recoveries": _get_signal(normalized_payload, "defense", "recoveries"),
        "turnover": _get_signal(normalized_payload, "transitions", "turnover"),
        "regain": _get_signal(normalized_payload, "transitions", "regain"),
    }
    transition_signals = {
        "regain": _get_signal(normalized_payload, "transitions", "regain"),
        "counter": _get_signal(normalized_payload, "transitions", "counter"),
        "transition": _get_signal(normalized_payload, "transitions", "transition"),
        "turnover": _get_signal(normalized_payload, "transitions", "turnover"),
        "direct_attack": _get_signal(normalized_payload, "transitions", "direct attack"),
    }

    if view_key == "Attack":
        zones = [
            {"label": "Zonas de centros", "value": attack_signals["crosses"], "x0": 72, "x1": 105, "y0": 50, "y1": 68, "color": "#60a5fa"},
            {"label": "Dribbles exitosos", "value": (attack_signals["box_entries"] + attack_signals["final_third"]) / 2, "x0": 62, "x1": 90, "y0": 18, "y1": 50, "color": "#34d399"},
            {"label": "Recoveries último tercio", "value": defense_signals["recoveries"] + defense_signals["regain"], "x0": 70, "x1": 105, "y0": 18, "y1": 50, "color": "#fbbf24"},
            {"label": "Origen de remates", "value": attack_signals["shots"] + attack_signals["xg"], "x0": 86, "x1": 105, "y0": 24, "y1": 44, "color": "#f87171"},
        ]
        return (
            "Attack View",
            "Dónde genera ventaja el equipo en fase ofensiva.",
            zones,
        )

    if view_key == "Defense":
        zones = [
            {"label": "Duelos en tercio propio", "value": defense_signals["duels"], "x0": 0, "x1": 35, "y0": 18, "y1": 50, "color": "#60a5fa"},
            {"label": "Pérdidas peligrosas", "value": defense_signals["turnover"], "x0": 0, "x1": 52.5, "y0": 20, "y1": 48, "color": "#f87171"},
            {"label": "Fragilidad costado izq.", "value": max(0.0, defense_signals["turnover"] - defense_signals["pressing"] * 0.4), "x0": 0, "x1": 40, "y0": 0, "y1": 16, "color": "#fb7185"},
            {"label": "Fragilidad costado der.", "value": max(0.0, defense_signals["turnover"] - defense_signals["pressing"] * 0.4), "x0": 0, "x1": 40, "y0": 52, "y1": 68, "color": "#fb7185"},
        ]
        return (
            "Defense View",
            "Dónde sufre más el equipo cuando defiende su propio arco.",
            zones,
        )

    zones = [
        {"label": "Recoveries por zona", "value": transition_signals["regain"], "x0": 35, "x1": 70, "y0": 20, "y1": 48, "color": "#22c55e"},
        {"label": "Losses por zona", "value": transition_signals["turnover"], "x0": 20, "x1": 60, "y0": 20, "y1": 48, "color": "#ef4444"},
        {"label": "Salida de contra", "value": transition_signals["counter"] + transition_signals["direct_attack"], "x0": 60, "x1": 95, "y0": 18, "y1": 50, "color": "#38bdf8"},
        {"label": "Ritmo de transición", "value": transition_signals["transition"], "x0": 45, "x1": 80, "y0": 0, "y1": 68, "color": "#f59e0b"},
    ]
    return (
        "Transitions View",
        "Cómo responde el equipo al recuperar o perder el balón.",
        zones,
    )


def build_pitch_view_figure(view_key: str, normalized_payload: dict[str, Any]) -> dict[str, Any]:
    title, subtitle, zones = _view_content(view_key, normalized_payload)
    figure = _base_pitch_figure()
    max_zone_value = max((zone["value"] for zone in zones), default=0.0)

    for zone in zones:
        alpha = _zone_alpha(float(zone["value"]), float(max_zone_value))
        figure.add_shape(
            type="rect",
            x0=zone["x0"],
            y0=zone["y0"],
            x1=zone["x1"],
            y1=zone["y1"],
            line=dict(color=_hex_to_rgba(zone["color"], min(0.95, alpha + 0.2)), width=1),
            fillcolor=_hex_to_rgba(zone["color"], alpha),
        )
        figure.add_annotation(
            x=(zone["x0"] + zone["x1"]) / 2,
            y=(zone["y0"] + zone["y1"]) / 2,
            text=f"{zone['label']}<br>{int(round(zone['value']))}",
            showarrow=False,
            font=dict(size=10, color="#f8fafc"),
            align="center",
        )

    has_signal = any(float(zone["value"]) > 0 for zone in zones)
    if not has_signal:
        figure.add_annotation(
            x=52.5,
            y=34,
            text="Sin señales zonales suficientes",
            showarrow=False,
            font=dict(size=13, color="#e5e7eb"),
        )

    return {
        "title": title,
        "subtitle": subtitle,
        "figure": figure,
        "has_signal": has_signal,
    }
