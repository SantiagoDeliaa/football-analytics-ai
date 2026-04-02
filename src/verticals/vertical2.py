import streamlit as st
import json

from src.services.event_normalizer import normalize_event_data
from src.services.pdf_ingestion import ingest_pdf
from src.services.proprietary_metrics import calculate_proprietary_metrics
from src.services.proprietary_metrics import get_metric_presentation
from src.utils.ui.pitch_views import build_pitch_view_figure


def render_vertical2() -> None:
    st.title("Vertical 2 — Data Analytics")
    st.caption("Wyscout PDF ingestion MVP para convertir reportes en un schema analítico estable")
    if st.button("Volver a Home", key="vertical2_back_home"):
        st.session_state.active_vertical = "home"
        st.rerun()

    st.subheader("PDF Ingestion")
    uploaded_pdf = st.file_uploader("Subir reporte Wyscout (.pdf)", type=["pdf"], key="vertical2_pdf_uploader")

    if uploaded_pdf is None:
        st.info("Sube un reporte PDF para iniciar la extracción y normalización de datos.")
        return

    with st.spinner("Procesando reporte PDF..."):
        raw_payload = ingest_pdf(uploaded_pdf)
        normalized_payload = normalize_event_data(raw_payload)
        proprietary_metrics = calculate_proprietary_metrics(normalized_payload, raw_payload)

    st.subheader("Estado del procesamiento")
    st.markdown(f"**Archivo:** {raw_payload.get('file_name', 'unknown.pdf')}")
    st.markdown(f"**Páginas detectadas:** {raw_payload.get('page_count', 0)}")

    if raw_payload.get("status") == "ok":
        st.success("Parsing completado. Datos listos para revisión preliminar.")
    else:
        st.warning("Parsing parcial aplicado. Se utilizó un fallback robusto para mantener continuidad.")

    for message in raw_payload.get("messages", []):
        st.caption(message)

    st.subheader("Proprietary Metrics MVP")
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

    st.subheader("Cancha Principal")
    attack_tab, defense_tab, transitions_tab = st.tabs(["Attack", "Defense", "Transitions"])

    with attack_tab:
        attack_view = build_pitch_view_figure("Attack", normalized_payload)
        st.markdown(f"**{attack_view.get('title', 'Attack View')}**")
        st.caption(attack_view.get("subtitle", "Vista ofensiva del comportamiento del equipo."))
        if not attack_view.get("has_signal", True):
            st.info("Datos ofensivos limitados en este reporte. Mostramos una vista base de referencia.")
        st.plotly_chart(attack_view.get("figure"), use_container_width=True)

    with defense_tab:
        defense_view = build_pitch_view_figure("Defense", normalized_payload)
        st.markdown(f"**{defense_view.get('title', 'Defense View')}**")
        st.caption(defense_view.get("subtitle", "Vista defensiva del comportamiento del equipo."))
        if not defense_view.get("has_signal", True):
            st.info("Datos defensivos limitados en este reporte. Mostramos una vista base de referencia.")
        st.plotly_chart(defense_view.get("figure"), use_container_width=True)

    with transitions_tab:
        transitions_view = build_pitch_view_figure("Transitions", normalized_payload)
        st.markdown(f"**{transitions_view.get('title', 'Transitions View')}**")
        st.caption(transitions_view.get("subtitle", "Vista de transiciones del comportamiento del equipo."))
        if not transitions_view.get("has_signal", True):
            st.info("Datos de transición limitados en este reporte. Mostramos una vista base de referencia.")
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
