"""
Soccer Analytics AI - Versión con Estadísticas Tácticas
Interfaz Streamlit mejorada con análisis táctico completo
"""

import streamlit as st
from pathlib import Path
from src.models.load_model import load_roboflow_model
from src.controllers.process_video import process_video
from src.utils.config import INPUTS_DIR, OUTPUTS_DIR
from ultralytics import YOLO
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2

def draw_pitch_base(width_px: int, height_px: int) -> np.ndarray:
    pitch = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    pitch[:, :] = (20, 70, 20)
    line_color = (255, 255, 255)
    thickness = max(1, int(min(width_px, height_px) * 0.004))
    cv2.rectangle(pitch, (0, 0), (width_px - 1, height_px - 1), line_color, thickness)
    mid_x = width_px // 2
    cv2.line(pitch, (mid_x, 0), (mid_x, height_px - 1), line_color, thickness)
    circle_radius = int(height_px * 0.12)
    cv2.circle(pitch, (mid_x, height_px // 2), circle_radius, line_color, thickness)
    box_w = int(width_px * 0.17)
    box_h = int(height_px * 0.36)
    box_y1 = (height_px - box_h) // 2
    box_y2 = box_y1 + box_h
    cv2.rectangle(pitch, (0, box_y1), (box_w, box_y2), line_color, thickness)
    cv2.rectangle(pitch, (width_px - box_w, box_y1), (width_px - 1, box_y2), line_color, thickness)
    return pitch

def render_heatmap_overlay(heatmap_small: np.ndarray, out_w: int, out_h: int, flip_vertical: bool, use_log: bool) -> np.ndarray:
    heatmap = heatmap_small.astype(np.float32)
    if flip_vertical:
        heatmap = np.flipud(heatmap)
    if use_log:
        heatmap = np.log1p(heatmap)
    positive = heatmap[heatmap > 0]
    if positive.size > 0:
        p2 = float(np.percentile(positive, 2))
        p98 = float(np.percentile(positive, 98))
        if p98 <= p2:
            p98 = p2 + 1e-6
        heatmap = np.clip(heatmap, p2, p98)
        heatmap = (heatmap - p2) / (p98 - p2)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    else:
        heatmap_uint8 = np.zeros_like(heatmap, dtype=np.uint8)
    heatmap_up = cv2.resize(heatmap_uint8, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    heatmap_color = cv2.applyColorMap(heatmap_up, cmap)
    alpha = (heatmap_up.astype(np.float32) / 255.0) * 0.75
    alpha[heatmap_up < 10] = 0.0
    alpha_3 = np.dstack([alpha, alpha, alpha])
    pitch = draw_pitch_base(out_w, out_h).astype(np.float32)
    overlay = heatmap_color.astype(np.float32)
    blended = pitch * (1.0 - alpha_3) + overlay * alpha_3
    return blended.astype(np.uint8)

def draw_heatmap_legend(height_px: int, width_px: int = 70) -> np.ndarray:
    grad = np.linspace(1, 0, height_px, dtype=np.float32)
    grad_img = (grad[:, None] * 255).astype(np.uint8)
    grad_img = np.repeat(grad_img, width_px, axis=1)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    legend = cv2.applyColorMap(grad_img, cmap)
    text_color = (255, 255, 255)
    cv2.putText(legend, "Densidad", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    cv2.putText(legend, "relativa", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return legend

st.set_page_config(page_title="Soccer Analytics AI", layout="wide", initial_sidebar_state="expanded")

st.title("⚽ Soccer Analytics AI")
st.caption("Análisis táctico completo: Tracking, Formaciones y Métricas de Comportamiento Colectivo")

# === SIDEBAR CONFIGURATION ===
st.sidebar.header("⚙️ Configuración")

# 1. PLAYERS MODEL
st.sidebar.subheader("1. Jugadores")
player_source = st.sidebar.radio(
    "Modelo de Jugadores",
    ["YOLOv8 Genérico (COCO)", "Subir Modelo Custom (.pt)"],
    index=0
)

player_model = None
if player_source == "YOLOv8 Genérico (COCO)":
    model_size = st.sidebar.selectbox("Tamaño del modelo", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], index=2)
    with st.spinner(f"Cargando {model_size}..."):
        player_model = load_roboflow_model(model_size)
else:
    uploaded_player = st.sidebar.file_uploader("Subir modelo jugadores (.pt)", type=["pt"])
    if uploaded_player:
        p_path = Path("models") / uploaded_player.name
        p_path.parent.mkdir(exist_ok=True)
        with open(p_path, "wb") as f:
            f.write(uploaded_player.read())
        player_model = YOLO(p_path)
        st.sidebar.success(f"Cargado: {uploaded_player.name}")

# 2. BALL MODEL
st.sidebar.subheader("2. Pelota")
ball_source = st.sidebar.radio(
    "Modelo de Pelota",
    ["Heurística (Clase 'sports ball')", "Subir Modelo Custom (.pt)"],
    index=0
)

ball_model = None
if ball_source == "Subir Modelo Custom (.pt)":
    uploaded_ball = st.sidebar.file_uploader("Subir modelo pelota (.pt)", type=["pt"])
    if uploaded_ball:
        b_path = Path("models") / uploaded_ball.name
        b_path.parent.mkdir(exist_ok=True)
        with open(b_path, "wb") as f:
            f.write(uploaded_ball.read())
        ball_model = YOLO(b_path)
        st.sidebar.success(f"Cargado: {uploaded_ball.name}")

# 3. RADAR / PITCH
st.sidebar.subheader("3. Radar View")
enable_radar = st.sidebar.checkbox("Habilitar Radar", value=True)
enable_analytics = st.sidebar.checkbox("Habilitar Análisis Táctico", value=True,
                                       help="Calcula formaciones y métricas tácticas")

pitch_model = None
full_field_approx = False

if enable_radar:
    pitch_source = st.sidebar.radio(
        "Modelo de Campo",
        [
            "Homography.pt (32 keypoints) - ⭐ Recomendado",
            "Soccana Keypoint (29 keypoints)",
            "Aproximación Pantalla Completa (Experimental)"
        ],
        index=0,
        help="Homography.pt detecta más keypoints y tiene mayor tasa de éxito"
    )

    if pitch_source == "Homography.pt (32 keypoints) - ⭐ Recomendado":
        homography_path = Path("models/homography.pt")
        if homography_path.exists():
            with st.spinner("Cargando modelo Homography (32 keypoints)..."):
                pitch_model = YOLO(str(homography_path))
            st.sidebar.success("✅ Modelo Homography cargado (100% tasa éxito)")
        else:
            st.sidebar.error(f"❌ Modelo no encontrado: {homography_path}")
            st.sidebar.info("Asegúrate de tener el archivo models/homography.pt")

    elif pitch_source == "Soccana Keypoint (29 keypoints)":
        soccana_path = Path("models/soccana_keypoint/Model/weights/best.pt")
        if soccana_path.exists():
            with st.spinner("Cargando modelo Soccana_Keypoint (29 keypoints)..."):
                pitch_model = YOLO(str(soccana_path))
            st.sidebar.success("✅ Modelo Soccana cargado (~65% tasa éxito)")
        else:
            st.sidebar.error(f"❌ Modelo no encontrado")
            st.sidebar.info("Descarga con: python scripts/download_soccana_model.py")

    else:
        full_field_approx = True

# === MAIN AREA ===
# Initialize session state for stats
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

uploaded_video = st.file_uploader("📹 Arrastra un video aquí", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save input video
    input_path = INPUTS_DIR / uploaded_video.name
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    # Tabs for organization
    tabs = st.tabs(["🎬 Video", "📊 Estadísticas", "📈 Gráficos", "💾 Exportar", "🕵️ Scouting"])

    with tabs[0]:
        col_input, col_output = st.columns(2)

        with col_input:
            st.subheader("Video Original")
            st.video(str(input_path))

        with col_output:
            if st.session_state.video_processed:
                st.subheader("Video Procesado")
                output_filename = f"processed_{uploaded_video.name}"
                target_path = OUTPUTS_DIR / output_filename
                if target_path.exists():
                    st.video(str(target_path))

                    with open(target_path, "rb") as f:
                        st.download_button(
                            "⬇️ Descargar Video",
                            f,
                            file_name=output_filename,
                            mime="video/mp4"
                        )
            else:
                st.info("👉 Haz clic en 'Procesar Video' para iniciar")

    # Process Button
    if st.button("🚀 Procesar Video", type="primary", use_container_width=True):
        if player_model is None:
            st.error("❌ Debes cargar un modelo de jugadores")
        else:
            with st.spinner("Procesando video..."):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                output_filename = f"processed_{uploaded_video.name}"
                target_path = OUTPUTS_DIR / output_filename
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

                try:
                    status_placeholder.text("🎯 Iniciando análisis...")
                    progress_bar.progress(10)

                    # Process video
                    def _on_progress(ratio, frame, total):
                        percent = min(99, 10 + int(80 * ratio))
                        progress_bar.progress(percent)
                        status_placeholder.text(f"Procesando... {percent}% ({frame}/{total})")
                    process_video(
                        source_path=str(input_path),
                        target_path=str(target_path),
                        player_model=player_model,
                        ball_model=ball_model,
                        pitch_model=pitch_model,
                        conf=0.3,
                        detection_mode="players_and_ball",
                        img_size=512,
                        full_field_approx=full_field_approx,
                        progress_callback=_on_progress
                    )

                    progress_bar.progress(90)
                    status_placeholder.text("📊 Generando estadísticas...")

                    # Check if stats file exists
                    stats_path = target_path.parent / f"{target_path.stem}_stats.json"
                    if stats_path.exists():
                        with open(stats_path, 'r') as f:
                            st.session_state.stats = json.load(f)

                    progress_bar.progress(100)
                    status_placeholder.success("✅ ¡Procesamiento completado!")
                    st.session_state.video_processed = True
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())

    # Statistics Tab
    with tabs[1]:
        if st.session_state.stats:
            stats = st.session_state.stats

            st.subheader("📊 Análisis Táctico")

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Duración", f"{stats.get('duration_seconds', 0):.1f}s")
            with col2:
                st.metric("Frames Procesados", stats.get('total_frames', 0))
            with col3:
                fps = stats.get('total_frames', 0) / stats.get('duration_seconds', 1)
                st.metric("FPS", f"{fps:.1f}")

            st.divider()

            # Formations
            if 'formations' in stats:
                st.subheader("⚽ Formaciones Detectadas")

                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown("### 🟢 Team 1")
                    form1 = stats['formations'].get('team1', {}).get('most_common', 'N/A')
                    st.metric("Formación más común", form1, help="Basada en análisis temporal")

                with col_t2:
                    st.markdown("### 🔵 Team 2")
                    form2 = stats['formations'].get('team2', {}).get('most_common', 'N/A')
                    st.metric("Formación más común", form2, help="Basada en análisis temporal")

            st.divider()

            # Tactical Metrics
            if 'metrics' in stats:
                st.subheader("📈 Métricas Tácticas")

                metrics1 = stats['metrics'].get('team1', {})
                metrics2 = stats['metrics'].get('team2', {})

                # Comparison table
                comparison_data = {
                    'Métrica': ['Presión (m)', 'Amplitud (m)', 'Compactación (m²)'],
                    'Team 1': [
                        f"{metrics1.get('pressure_height', {}).get('mean', 0):.1f}",
                        f"{metrics1.get('offensive_width', {}).get('mean', 0):.1f}",
                        f"{metrics1.get('compactness', {}).get('mean', 0):.0f}"
                    ],
                    'Team 2': [
                        f"{metrics2.get('pressure_height', {}).get('mean', 0):.1f}",
                        f"{metrics2.get('offensive_width', {}).get('mean', 0):.1f}",
                        f"{metrics2.get('compactness', {}).get('mean', 0):.0f}"
                    ]
                }

                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)

                # Detailed metrics
                with st.expander("Ver métricas detalladas"):
                    col_det1, col_det2 = st.columns(2)

                    with col_det1:
                        st.markdown("#### Team 1")
                        if metrics1:
                            for metric_name, values in metrics1.items():
                                if isinstance(values, dict) and 'mean' in values:
                                    st.text(f"{metric_name}:")
                                    st.text(f"  Media: {values['mean']:.2f}")
                                    st.text(f"  Min-Max: {values['min']:.2f} - {values['max']:.2f}")

                    with col_det2:
                        st.markdown("#### Team 2")
                        if metrics2:
                            for metric_name, values in metrics2.items():
                                if isinstance(values, dict) and 'mean' in values:
                                    st.text(f"{metric_name}:")
                                    st.text(f"  Media: {values['mean']:.2f}")
                                    st.text(f"  Min-Max: {values['min']:.2f} - {values['max']:.2f}")

        else:
            st.info("📊 Las estadísticas aparecerán aquí después de procesar el video")

    # Charts Tab
    with tabs[2]:
        if st.session_state.stats and 'timeline' in st.session_state.stats:
            st.subheader("📈 Evolución Temporal")

            timeline = st.session_state.stats['timeline']

            # Pressure Height Chart
            if 'team1' in timeline and 'pressure_height' in timeline['team1']:
                frames1 = timeline['team1'].get('frame_number', [])
                pressure1 = timeline['team1'].get('pressure_height', [])

                frames2 = timeline.get('team2', {}).get('frame_number', [])
                pressure2 = timeline.get('team2', {}).get('pressure_height', [])

                fig_pressure = go.Figure()
                fig_pressure.add_trace(go.Scatter(x=frames1, y=pressure1, name='Team 1', line=dict(color='green')))
                fig_pressure.add_trace(go.Scatter(x=frames2, y=pressure2, name='Team 2', line=dict(color='blue')))
                fig_pressure.update_layout(
                    title='Altura de Presión (m)',
                    xaxis_title='Frame',
                    yaxis_title='Presión (m)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_pressure, use_container_width=True)

            # Compactness Chart
            if 'team1' in timeline and 'compactness' in timeline['team1']:
                compact1 = timeline['team1'].get('compactness', [])
                compact2 = timeline.get('team2', {}).get('compactness', [])

                fig_compact = go.Figure()
                fig_compact.add_trace(go.Scatter(x=frames1, y=compact1, name='Team 1', line=dict(color='green')))
                fig_compact.add_trace(go.Scatter(x=frames2, y=compact2, name='Team 2', line=dict(color='blue')))
                fig_compact.update_layout(
                    title='Compactación (m²)',
                    xaxis_title='Frame',
                    yaxis_title='Área (m²)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_compact, use_container_width=True)

            # Width Chart
            if 'team1' in timeline and 'offensive_width' in timeline['team1']:
                width1 = timeline['team1'].get('offensive_width', [])
                width2 = timeline.get('team2', {}).get('offensive_width', [])

                fig_width = go.Figure()
                fig_width.add_trace(go.Scatter(x=frames1, y=width1, name='Team 1', line=dict(color='green')))
                fig_width.add_trace(go.Scatter(x=frames2, y=width2, name='Team 2', line=dict(color='blue')))
                fig_width.update_layout(
                    title='Amplitud Ofensiva (m)',
                    xaxis_title='Frame',
                    yaxis_title='Amplitud (m)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_width, use_container_width=True)

        else:
            st.info("📈 Los gráficos aparecerán aquí después de procesar el video con análisis táctico")

    # Export Tab
    with tabs[3]:
        if st.session_state.stats:
            st.subheader("💾 Exportar Datos")

            col_json, col_csv = st.columns(2)

            with col_json:
                st.markdown("### JSON")
                json_str = json.dumps(st.session_state.stats, indent=2)
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json_str,
                    file_name="soccer_analytics_stats.json",
                    mime="application/json"
                )

            with col_csv:
                st.markdown("### CSV")
                if 'timeline' in st.session_state.stats:
                    timeline = st.session_state.stats['timeline']
                    if 'team1' in timeline:
                        df_export = pd.DataFrame(timeline['team1'])
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Descargar CSV (Team 1)",
                            data=csv,
                            file_name="team1_timeline.csv",
                            mime="text/csv"
                        )

        else:
            st.info("💾 Las opciones de exportación aparecerán aquí después de procesar el video")

    with tabs[4]:
        if st.session_state.stats and 'scouting_heatmaps' in st.session_state.stats:
            stats = st.session_state.stats
            st.subheader("🕵️ Scouting")
            with st.expander("Cómo interpretar estas métricas"):
                st.markdown("- Compactación: profundidad/ancho del bloque en metros")
                st.markdown("- Línea defensiva: altura promedio del último bloque")
                st.markdown("- Heatmap: distribución espacial de presencia")
            col_t1, col_t2 = st.columns(2)
            if 'timeline' in stats:
                timeline = stats['timeline']
                with col_t1:
                    st.markdown("### 🟢 Team 1")
                    frames = timeline.get('team1', {}).get('frame_number', [])
                    depth = timeline.get('team1', {}).get('block_depth_m', [])
                    width = timeline.get('team1', {}).get('block_width_m', [])
                    left = timeline.get('team1', {}).get('def_line_left_m', [])
                    right = timeline.get('team1', {}).get('def_line_right_m', [])
                    if len(frames) > 0:
                        fig_c = go.Figure()
                        fig_c.add_trace(go.Scatter(x=frames, y=depth, name='Profundidad (m)', line=dict(color='green')))
                        fig_c.add_trace(go.Scatter(x=frames, y=width, name='Ancho (m)', line=dict(color='darkgreen')))
                        fig_c.update_layout(title='Compactación', xaxis_title='Frame', yaxis_title='Metros', hovermode='x unified')
                        st.plotly_chart(fig_c, use_container_width=True)
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(x=frames, y=left, name='Línea Izquierda (m)', line=dict(color='blue')))
                        fig_d.add_trace(go.Scatter(x=frames, y=right, name='Línea Derecha (m)', line=dict(color='darkblue')))
                        fig_d.update_layout(title='Línea Defensiva', xaxis_title='Frame', yaxis_title='Metros', hovermode='x unified')
                        st.plotly_chart(fig_d, use_container_width=True)
                with col_t2:
                    st.markdown("### 🔵 Team 2")
                    frames = timeline.get('team2', {}).get('frame_number', [])
                    depth = timeline.get('team2', {}).get('block_depth_m', [])
                    width = timeline.get('team2', {}).get('block_width_m', [])
                    left = timeline.get('team2', {}).get('def_line_left_m', [])
                    right = timeline.get('team2', {}).get('def_line_right_m', [])
                    if len(frames) > 0:
                        fig_c = go.Figure()
                        fig_c.add_trace(go.Scatter(x=frames, y=depth, name='Profundidad (m)', line=dict(color='green')))
                        fig_c.add_trace(go.Scatter(x=frames, y=width, name='Ancho (m)', line=dict(color='darkgreen')))
                        fig_c.update_layout(title='Compactación', xaxis_title='Frame', yaxis_title='Metros', hovermode='x unified')
                        st.plotly_chart(fig_c, use_container_width=True)
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(x=frames, y=left, name='Línea Izquierda (m)', line=dict(color='blue')))
                        fig_d.add_trace(go.Scatter(x=frames, y=right, name='Línea Derecha (m)', line=dict(color='darkblue')))
                        fig_d.update_layout(title='Línea Defensiva', xaxis_title='Frame', yaxis_title='Metros', hovermode='x unified')
                        st.plotly_chart(fig_d, use_container_width=True)
            st.subheader("Heatmaps")
            flip_vertical = st.checkbox("Invertir eje vertical (Y)", value=True)
            use_log = st.checkbox("Usar escala logarítmica", value=False)
            col_h1, col_h2 = st.columns(2)
            heatmap_meta = stats.get('scouting_heatmaps', {})
            out_w = 840
            out_h = int(out_w * 68 / 105)
            legend = draw_heatmap_legend(out_h)
            with col_h1:
                h1 = heatmap_meta.get('team1', None)
                if h1 is not None:
                    heatmap = np.array(h1, dtype=np.float32)
                    rendered = render_heatmap_overlay(heatmap, out_w, out_h, flip_vertical, use_log)
                    combo = np.concatenate([rendered, legend], axis=1)
                    st.image(combo, caption="Heatmap Team 1", use_column_width=True)
                    st.caption(f"bins_shape={heatmap_meta.get('bins_shape')} | sample_rate={heatmap_meta.get('sample_rate')} | total_samples={heatmap_meta.get('total_samples')}")
            with col_h2:
                h2 = heatmap_meta.get('team2', None)
                if h2 is not None:
                    heatmap = np.array(h2, dtype=np.float32)
                    rendered = render_heatmap_overlay(heatmap, out_w, out_h, flip_vertical, use_log)
                    combo = np.concatenate([rendered, legend], axis=1)
                    st.image(combo, caption="Heatmap Team 2", use_column_width=True)
                    st.caption(f"bins_shape={heatmap_meta.get('bins_shape')} | sample_rate={heatmap_meta.get('sample_rate')} | total_samples={heatmap_meta.get('total_samples')}")
            with st.expander("Cómo interpretar el Heatmap"):
                st.markdown("- Colores más calientes = más presencia")
                st.markdown("- Muestra zonas de influencia por equipo")
                st.markdown("- Depende del tracking y homografía")
                st.markdown("- Densidad relativa al clip")
        else:
            st.warning("Scouting metrics not available for this video.")
else:
    st.info("👈 Sube un video para comenzar el análisis táctico")

# Footer
st.divider()
st.caption("Soccer Analytics AI - Sistema de Análisis Táctico Completo")
st.caption("Tracking + Formaciones + Métricas de Comportamiento Colectivo")
