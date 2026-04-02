import streamlit as st
import json
import plotly.graph_objects as go

from src.services.event_normalizer import normalize_event_data
from src.services.pdf_ingestion import ingest_pdf
from src.services.insight_generator import generate_match_insights
from src.services.proprietary_metrics import calculate_proprietary_metrics
from src.services.proprietary_metrics import get_metric_presentation
from src.utils.ui.pitch_views import build_pitch_view_figure
from src.utils.ui.theme import apply_plotly_dark_theme


def _build_tactical_radar(
    proprietary_metrics: dict,
    normalized_payload: dict,
) -> go.Figure:
    attack_signals = normalized_payload.get("attack", {}).get("signals", {})
    final_third_activity = int(
        round(
            40
            + 8 * float(attack_signals.get("final third", 0) or 0)
            + 8 * float(attack_signals.get("shots", 0) or 0)
            + 7 * float(attack_signals.get("box entries", 0) or 0)
        )
    )
    final_third_activity = max(0, min(100, final_third_activity))

    categories = [
        "Control Territorial",
        "Verticalidad",
        "Impacto Pressing",
        "Riesgo en Salida",
        "Actividad Último Tercio",
    ]
    values = [
        int(proprietary_metrics.get("field_tilt_index", {}).get("score", 0) or 0),
        int(proprietary_metrics.get("directness_index", {}).get("score", 0) or 0),
        int(proprietary_metrics.get("pressing_efficiency", {}).get("score", 0) or 0),
        int(proprietary_metrics.get("risk_exposure_score", {}).get("score", 0) or 0),
        final_third_activity,
    ]
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Perfil del partido",
            line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56, 189, 248, 0.25)",
        )
    )
    radar.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#d1d5db")),
            angularaxis=dict(tickfont=dict(color="#f3f4f6", size=11)),
            bgcolor="#111827",
        ),
        showlegend=False,
        height=360,
    )
    apply_plotly_dark_theme(radar)
    radar.update_layout(paper_bgcolor="#0f131a", plot_bgcolor="#0f131a")
    return radar


def render_vertical2() -> None:
    st.title("Vertical 2 — Data Analytics")
    st.caption("Resumen táctico automático desde reportes Wyscout, pensado para lectura rápida de analistas y scouts.")
    if st.button("Volver a Home", key="vertical2_back_home"):
        st.session_state.active_vertical = "home"
        st.rerun()

    st.subheader("Carga de Reporte")
    uploaded_pdf = st.file_uploader("Subir reporte Wyscout (.pdf)", type=["pdf"], key="vertical2_pdf_uploader")

    if uploaded_pdf is None:
        st.info("Sube un PDF Wyscout para generar métricas, cancha táctica e insights automáticos.")
        return

    with st.spinner("Procesando reporte PDF..."):
        raw_payload = ingest_pdf(uploaded_pdf)
        normalized_payload = normalize_event_data(raw_payload)
        proprietary_metrics = calculate_proprietary_metrics(normalized_payload, raw_payload)
        insights = generate_match_insights(normalized_payload, proprietary_metrics)

    st.subheader("Estado de Procesamiento")
    st.markdown(f"**Archivo:** {raw_payload.get('file_name', 'unknown.pdf')}")
    st.markdown(f"**Páginas detectadas:** {raw_payload.get('page_count', 0)}")

    if raw_payload.get("status") == "ok":
        st.success("Parsing completado. Datos listos para revisión preliminar.")
    else:
        st.warning("Se aplicó una lectura parcial del reporte. Mostramos una salida estable para mantener el análisis.")

    for message in raw_payload.get("messages", []):
        st.caption(message)

    st.subheader("KPIs del Partido")
    metric_order = [
        "field_tilt_index",
        "directness_index",
        "pressing_efficiency",
        "risk_exposure_score",
    ]
    metric_columns = st.columns(4)
    for idx, metric_key in enumerate(metric_order):
        metric_data = proprietary_metrics.get(metric_key, {})
        presentation = get_metric_presentation(metric_key, metric_data)
        with metric_columns[idx]:
            st.metric(presentation.get("label", "Métrica"), f"{metric_data.get('score', 0)}")
            st.caption(presentation.get("subtitle", "Métrica no disponible."))
            st.markdown(
                f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:{presentation.get('color', '#f59e0b')};color:#111827;font-weight:600;'>Nivel: {presentation.get('level', 'Medio')}</span>",
                unsafe_allow_html=True,
            )
            st.caption(presentation.get("interpretation", "Interpretación no disponible."))

    st.subheader("Radar Táctico")
    st.caption("Comparativa rápida de cinco dimensiones clave del comportamiento colectivo.")
    st.plotly_chart(_build_tactical_radar(proprietary_metrics, normalized_payload), use_container_width=True)

    st.subheader("Insights del Partido")
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("No se detectaron señales suficientes para redactar insights automáticos.")

    st.subheader("Cancha Principal")
    attack_tab, defense_tab, transitions_tab = st.tabs(["Attack", "Defense", "Transitions"])

    with attack_tab:
        attack_view = build_pitch_view_figure("Attack", normalized_payload)
        st.markdown(f"**{attack_view.get('title', 'Attack View')}**")
        st.caption(attack_view.get("subtitle", "Vista ofensiva del comportamiento del equipo."))
        if not attack_view.get("has_signal", True):
            st.info("Señales ofensivas limitadas en este reporte. Mostramos mapa base con información mínima.")
        st.plotly_chart(attack_view.get("figure"), use_container_width=True)

    with defense_tab:
        defense_view = build_pitch_view_figure("Defense", normalized_payload)
        st.markdown(f"**{defense_view.get('title', 'Defense View')}**")
        st.caption(defense_view.get("subtitle", "Vista defensiva del comportamiento del equipo."))
        if not defense_view.get("has_signal", True):
            st.info("Señales defensivas limitadas en este reporte. Mostramos mapa base con información mínima.")
        st.plotly_chart(defense_view.get("figure"), use_container_width=True)

    with transitions_tab:
        transitions_view = build_pitch_view_figure("Transitions", normalized_payload)
        st.markdown(f"**{transitions_view.get('title', 'Transitions View')}**")
        st.caption(transitions_view.get("subtitle", "Vista de transiciones del comportamiento del equipo."))
        if not transitions_view.get("has_signal", True):
            st.info("Señales de transición limitadas en este reporte. Mostramos mapa base con información mínima.")
        st.plotly_chart(transitions_view.get("figure"), use_container_width=True)

    preview = {
        "status": normalized_payload.get("status", "warning"),
        "match_info": normalized_payload.get("match_info", {}),
        "formations": normalized_payload.get("formations", []),
        "attack": normalized_payload.get("attack", {}),
        "defense": normalized_payload.get("defense", {}),
        "transitions": normalized_payload.get("transitions", {}),
        "build_up": normalized_payload.get("build_up", {}),
        "finishing": normalized_payload.get("finishing", {}),
        "proprietary_metrics": proprietary_metrics,
    }

    st.subheader("Preview del schema normalizado")
    st.code(json.dumps(preview, indent=2, ensure_ascii=False), language="json")
