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
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_HOT
    heatmap_color = cv2.applyColorMap(heatmap_up, cmap)
    alpha = (heatmap_up.astype(np.float32) / 255.0) * 0.75
    alpha[heatmap_up < 10] = 0.0
    alpha_3 = np.dstack([alpha, alpha, alpha])
    pitch = draw_pitch_base(out_w, out_h).astype(np.float32)
    overlay = heatmap_color.astype(np.float32)
    blended = pitch * (1.0 - alpha_3) + overlay * alpha_3
    return blended.astype(np.uint8)

def draw_heatmap_legend(height_px: int, width_px: int = 90) -> np.ndarray:
    grad = np.linspace(1, 0, height_px, dtype=np.float32)
    grad_img = (grad[:, None] * 255).astype(np.uint8)
    grad_img = np.repeat(grad_img, width_px, axis=1)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_HOT
    legend = cv2.applyColorMap(grad_img, cmap)
    text_color = (255, 255, 255)
    cv2.putText(legend, "Presencia", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    cv2.putText(legend, "(clip)", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    return legend

def format_mmss(seconds: float) -> str:
    minutes = int(seconds) // 60
    sec = int(seconds) % 60
    return f"{minutes:02d}:{sec:02d}"

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
enable_possession = st.sidebar.checkbox("Habilitar Análisis de Posesión", value=False,
                                        help="Posesión de pelota, velocidad y distancia por jugador")

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
    tabs = st.tabs(["🎬 Video", "📊 Estadísticas", "📈 Gráficos", "💾 Exportar", "🕵️ Scouting", "📘 Interpretación", "⚽ Posesión"])

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
                        progress_callback=_on_progress,
                        enable_possession=enable_possession
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
            st.markdown("### 🧠 Resumen automático del clip")
            metrics = stats.get('metrics', {})
            timeline = stats.get('timeline', {})
            duration_seconds = stats.get('duration_seconds', None)
            total_frames = stats.get('total_frames', None)
            fps = None
            if duration_seconds and total_frames:
                if duration_seconds > 0:
                    fps = total_frames / duration_seconds
            def get_valid_frames(team_key: str) -> int:
                team_metrics = metrics.get(team_key, {})
                valid_frames = team_metrics.get('valid_frames', None)
                if valid_frames is None:
                    valid_frames = len(timeline.get(team_key, {}).get('frame_number', []))
                return valid_frames
            def compute_confidence(team_key: str) -> tuple:
                valid_frames = get_valid_frames(team_key)
                if total_frames and total_frames > 0:
                    ratio = max(0.0, min(1.0, valid_frames / total_frames))
                    if ratio >= 0.70:
                        return "Alta", ratio
                    if ratio >= 0.40:
                        return "Media", ratio
                    return "Baja", ratio
                return None, None
            def build_summary(team_key: str, confidence_level: str) -> list:
                team_metrics = metrics.get(team_key, {})
                depth_mean = team_metrics.get('block_depth_m', {}).get('mean', None)
                width_mean = team_metrics.get('block_width_m', {}).get('mean', None)
                left_mean = team_metrics.get('def_line_left_m', {}).get('mean', None)
                right_mean = team_metrics.get('def_line_right_m', {}).get('mean', None)
                if depth_mean is None or width_mean is None or left_mean is None or right_mean is None:
                    return ["No hay suficientes datos"]
                bullets = []
                strong = confidence_level != "Baja"
                if depth_mean > 40:
                    bullets.append("Equipo muy largo (transición)" if strong else "Sugiere equipo muy largo (transición)")
                elif depth_mean < 30:
                    bullets.append("Bloque compacto en profundidad" if strong else "Podría indicar bloque compacto")
                if width_mean > 35:
                    bullets.append("Equipo abierto en amplitud" if strong else "Sugiere equipo abierto")
                elif width_mean < 25:
                    bullets.append("Equipo cerrado en amplitud" if strong else "Podría indicar equipo cerrado")
                line_avg = (left_mean + right_mean) / 2.0
                if line_avg >= 70:
                    bullets.append("Línea defensiva alta" if strong else "Sugiere línea defensiva alta")
                elif line_avg >= 50:
                    bullets.append("Bloque medio" if strong else "Podría indicar bloque medio")
                else:
                    bullets.append("Bloque bajo" if strong else "Podría indicar bloque bajo")
                if duration_seconds:
                    bullets.append(f"Duración del clip: {duration_seconds:.1f}s")
                return bullets
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**Team 1**")
                level1, ratio1 = compute_confidence('team1')
                if level1:
                    if level1 == "Alta":
                        st.success(f"Confianza: {level1} ({ratio1 * 100:.0f}%)")
                    elif level1 == "Media":
                        st.info(f"Confianza: {level1} ({ratio1 * 100:.0f}%)")
                    else:
                        st.warning(f"Confianza: {level1} ({ratio1 * 100:.0f}%)")
                    if level1 == "Baja":
                        st.warning("Interpretación limitada: pocos frames válidos...")
                        st.caption("⚠ Pocos frames válidos: posible jitter de homografía / tracking.")
                    elif level1 == "Media":
                        st.info("Interpretación moderada...")
                else:
                    st.info("Confianza: N/D")
                for item in build_summary('team1', level1):
                    st.markdown(f"- {item}")
            with col_s2:
                st.markdown("**Team 2**")
                level2, ratio2 = compute_confidence('team2')
                if level2:
                    if level2 == "Alta":
                        st.success(f"Confianza: {level2} ({ratio2 * 100:.0f}%)")
                    elif level2 == "Media":
                        st.info(f"Confianza: {level2} ({ratio2 * 100:.0f}%)")
                    else:
                        st.warning(f"Confianza: {level2} ({ratio2 * 100:.0f}%)")
                    if level2 == "Baja":
                        st.warning("Interpretación limitada: pocos frames válidos...")
                        st.caption("⚠ Pocos frames válidos: posible jitter de homografía / tracking.")
                    elif level2 == "Media":
                        st.info("Interpretación moderada...")
                else:
                    st.info("Confianza: N/D")
                for item in build_summary('team2', level2):
                    st.markdown(f"- {item}")
            st.caption("Qué mirar primero: (1) compactación, (2) línea defensiva, (3) heatmap.")
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
                        x_vals = frames
                        x_title = "Frame"
                        tickvals = None
                        ticktext = None
                        if fps:
                            x_vals = [f / fps for f in frames]
                            x_title = "Tiempo (mm:ss)"
                            if len(x_vals) > 1:
                                step = max(1, int(len(x_vals) / 6))
                                tickvals = x_vals[::step]
                                ticktext = [format_mmss(t) for t in tickvals]
                        st.markdown("#### Compactación ℹ️")
                        st.caption("Qué es: tamaño del bloque del equipo. Profundidad = largo (def→ata). Ancho = apertura lateral.")
                        st.caption("Cómo leer: picos = equipo se estira (transición). valles = equipo compacto (bloque).")
                        fig_c = go.Figure()
                        fig_c.add_trace(go.Scatter(x=x_vals, y=depth, name='Profundidad (m)', line=dict(color='green')))
                        fig_c.add_trace(go.Scatter(x=x_vals, y=width, name='Ancho (m)', line=dict(color='darkgreen')))
                        fig_c.update_layout(title='Compactación', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                        if tickvals:
                            fig_c.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                        st.plotly_chart(fig_c, use_container_width=True)
                        st.markdown("#### Línea defensiva ℹ️")
                        st.caption("Qué es: altura del último bloque en X (0–105m).")
                        st.caption("Cómo leer: sube = línea alta/presión; baja = repliegue/bloque bajo.")
                        st.caption("Nota: mostramos dos hipótesis (izq/der) por orientación broadcast.")
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(x=x_vals, y=left, name='Línea Izquierda (m)', line=dict(color='blue')))
                        fig_d.add_trace(go.Scatter(x=x_vals, y=right, name='Línea Derecha (m)', line=dict(color='darkblue')))
                        fig_d.update_layout(title='Línea Defensiva', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                        if tickvals:
                            fig_d.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                        st.plotly_chart(fig_d, use_container_width=True)
                with col_t2:
                    st.markdown("### 🔵 Team 2")
                    frames = timeline.get('team2', {}).get('frame_number', [])
                    depth = timeline.get('team2', {}).get('block_depth_m', [])
                    width = timeline.get('team2', {}).get('block_width_m', [])
                    left = timeline.get('team2', {}).get('def_line_left_m', [])
                    right = timeline.get('team2', {}).get('def_line_right_m', [])
                    if len(frames) > 0:
                        x_vals = frames
                        x_title = "Frame"
                        tickvals = None
                        ticktext = None
                        if fps:
                            x_vals = [f / fps for f in frames]
                            x_title = "Tiempo (mm:ss)"
                            if len(x_vals) > 1:
                                step = max(1, int(len(x_vals) / 6))
                                tickvals = x_vals[::step]
                                ticktext = [format_mmss(t) for t in tickvals]
                        st.markdown("#### Compactación ℹ️")
                        st.caption("Qué es: tamaño del bloque del equipo. Profundidad = largo (def→ata). Ancho = apertura lateral.")
                        st.caption("Cómo leer: picos = equipo se estira (transición). valles = equipo compacto (bloque).")
                        fig_c = go.Figure()
                        fig_c.add_trace(go.Scatter(x=x_vals, y=depth, name='Profundidad (m)', line=dict(color='green')))
                        fig_c.add_trace(go.Scatter(x=x_vals, y=width, name='Ancho (m)', line=dict(color='darkgreen')))
                        fig_c.update_layout(title='Compactación', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                        if tickvals:
                            fig_c.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                        st.plotly_chart(fig_c, use_container_width=True)
                        st.markdown("#### Línea defensiva ℹ️")
                        st.caption("Qué es: altura del último bloque en X (0–105m).")
                        st.caption("Cómo leer: sube = línea alta/presión; baja = repliegue/bloque bajo.")
                        st.caption("Nota: mostramos dos hipótesis (izq/der) por orientación broadcast.")
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Scatter(x=x_vals, y=left, name='Línea Izquierda (m)', line=dict(color='blue')))
                        fig_d.add_trace(go.Scatter(x=x_vals, y=right, name='Línea Derecha (m)', line=dict(color='darkblue')))
                        fig_d.update_layout(title='Línea Defensiva', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                        if tickvals:
                            fig_d.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                        st.plotly_chart(fig_d, use_container_width=True)
            st.subheader("Heatmaps")
            flip_vertical = st.checkbox("Invertir eje vertical (Y)", value=True)
            use_log = st.checkbox("Usar escala logarítmica", value=False)
            st.caption("Escala log: hace visibles zonas con poca presencia cuando hay una zona dominante.")
            col_h1, col_h2 = st.columns(2)
            heatmap_meta = stats.get('scouting_heatmaps', {})
            out_w = 840
            out_h = int(out_w * 68 / 105)
            legend = draw_heatmap_legend(out_h)
            with col_h1:
                h1 = heatmap_meta.get('team1', None)
                if h1 is not None:
                    st.markdown("#### Heatmap ℹ️")
                    st.caption("Intensidad = cantidad de presencia en este clip (no es ‘calor’, es frecuencia).")
                    heatmap = np.array(h1, dtype=np.float32)
                    rendered = render_heatmap_overlay(heatmap, out_w, out_h, flip_vertical, use_log)
                    combo = np.concatenate([rendered, legend], axis=1)
                    st.image(combo, caption="Heatmap Team 1", use_column_width=True)
                    st.caption(f"bins_shape={heatmap_meta.get('bins_shape')} | sample_rate={heatmap_meta.get('sample_rate')} | total_samples={heatmap_meta.get('total_samples')}")
            with col_h2:
                h2 = heatmap_meta.get('team2', None)
                if h2 is not None:
                    st.markdown("#### Heatmap ℹ️")
                    st.caption("Intensidad = cantidad de presencia en este clip (no es ‘calor’, es frecuencia).")
                    heatmap = np.array(h2, dtype=np.float32)
                    rendered = render_heatmap_overlay(heatmap, out_w, out_h, flip_vertical, use_log)
                    combo = np.concatenate([rendered, legend], axis=1)
                    st.image(combo, caption="Heatmap Team 2", use_column_width=True)
                    st.caption(f"bins_shape={heatmap_meta.get('bins_shape')} | sample_rate={heatmap_meta.get('sample_rate')} | total_samples={heatmap_meta.get('total_samples')}")
        else:
            st.warning("Scouting metrics not available for this video.")
    with tabs[5]:
        st.subheader("📘 Interpretación")
        st.markdown("### Compactación (profundidad/ancho)")
        st.markdown("- Qué mide: tamaño del bloque del equipo en el campo")
        st.markdown("- Cómo leerlo: si sube, el equipo se estira; si baja, se compacta")
        st.markdown("- Señales típicas: picos = transición; valles = bloque bajo")
        st.markdown("- Ejemplos: si Profundidad sube fuerte y Ancho sube → contragolpe / equipo estirado")
        st.markdown("- Cuándo desconfiar: tracking inestable o homografía saltando")
        st.markdown("### Línea defensiva")
        st.markdown("- Qué mide: posición en metros sobre el eje X del campo (0–105m)")
        st.markdown("- Cómo leerlo: sube = línea alta; baja = repliegue/bloque bajo")
        st.markdown("- Señales típicas: caída sostenida → bloque bajo")
        st.markdown("- Ejemplo: si Línea defensiva cae y se mantiene baja → bloque bajo")
        st.markdown("- Cuándo desconfiar: cambios bruscos por cámara o tracking")
        st.markdown("### Heatmap")
        st.markdown("- Qué mide: frecuencia acumulada de presencia por zona")
        st.markdown("- Cómo leerlo: zonas rojas = mayor presencia")
        st.markdown("- Señales típicas: concentración en banda o carril central")
        st.markdown("- Densidad relativa: se compara dentro del mismo clip")
        st.markdown("- Escala log: resalta zonas con poca presencia cuando hay una dominante")
        st.markdown("- Cuándo desconfiar: homografía imprecisa o poca muestra")

    # Possession Tab
    with tabs[6]:
        if st.session_state.stats and 'possession' in st.session_state.stats:
            stats = st.session_state.stats
            possession = stats['possession']
            speed_dist = stats.get('speed_distance', {})

            st.subheader("⚽ Posesión de Pelota")

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Team 1 Posesión", f"{possession['team1_possession_pct']:.1f}%")
            with col_p2:
                st.metric("Team 2 Posesión", f"{possession['team2_possession_pct']:.1f}%")
            with col_p3:
                st.metric("Frames Disputados", possession['contested_frames'])

            # Pie chart de posesión
            contested_pct = 100 - possession['team1_possession_pct'] - possession['team2_possession_pct']
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Team 1", "Team 2", "Sin posesión"],
                values=[
                    possession['team1_possession_pct'],
                    possession['team2_possession_pct'],
                    max(0, contested_pct),
                ],
                marker_colors=["#00FF00", "#00BFFF", "#888888"],
            )])
            fig_pie.update_layout(title="Distribución de Posesión")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Mejora L: Timeline de posesión
            timeline_data = possession.get('timeline', [])
            if timeline_data and len(timeline_data) > 10:
                st.subheader("Timeline de Posesión")
                # Agrupar en segmentos para visualización
                fps_val = stats.get('total_frames', len(timeline_data)) / max(0.1, stats.get('duration_seconds', 1))
                chunk_size = max(1, int(fps_val))  # 1 segundo por barra
                chunks_t1 = []
                chunks_t2 = []
                chunks_x = []
                for ci in range(0, len(timeline_data), chunk_size):
                    chunk = timeline_data[ci:ci+chunk_size]
                    t1_count = sum(1 for v in chunk if v == 'team1')
                    t2_count = sum(1 for v in chunk if v == 'team2')
                    total = max(1, t1_count + t2_count)
                    chunks_t1.append(t1_count / total * 100)
                    chunks_t2.append(t2_count / total * 100)
                    chunks_x.append(ci / max(1, fps_val))

                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Bar(x=chunks_x, y=chunks_t1, name='Team 1', marker_color='#00FF00'))
                fig_timeline.add_trace(go.Bar(x=chunks_x, y=chunks_t2, name='Team 2', marker_color='#00BFFF'))
                fig_timeline.update_layout(
                    barmode='stack',
                    title='Posesión por segundo',
                    xaxis_title='Tiempo (s)',
                    yaxis_title='%',
                    yaxis=dict(range=[0, 100]),
                    height=250,
                    margin=dict(t=40, b=30),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            # Mejora E: Estadísticas de pases
            passes_data = possession.get('passes', {})
            if passes_data and passes_data.get('total', 0) > 0:
                st.subheader("Pases y Pérdidas")
                col_pa1, col_pa2, col_pa3 = st.columns(3)
                with col_pa1:
                    st.metric("Pases Team 1", passes_data.get('team1_passes', 0))
                with col_pa2:
                    st.metric("Pases Team 2", passes_data.get('team2_passes', 0))
                with col_pa3:
                    st.metric("Pérdidas de Balón", passes_data.get('turnovers', 0))

            # Top poseedores
            if possession.get('top_possessors'):
                st.subheader("Jugadores con Más Posesión")
                df_poss = pd.DataFrame(possession['top_possessors'])
                df_poss.columns = ['ID Jugador', 'Frames', 'Equipo']
                st.dataframe(df_poss, use_container_width=True, hide_index=True)

            st.divider()

            # Velocidad y Distancia
            if speed_dist:
                st.subheader("Velocidad y Distancia")
                per_team = speed_dist.get('per_team', {})

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("### Team 1")
                    t1 = per_team.get('team1', {})
                    st.metric("Distancia Total", f"{t1.get('total_distance_m', 0):.0f} m")
                    st.metric("Dist. Prom. por Jugador", f"{t1.get('avg_distance_m', 0):.0f} m")
                    st.metric("Velocidad Máxima", f"{t1.get('max_speed_kmh', 0):.1f} km/h")
                    st.metric("Sprints Totales", t1.get('total_sprints', 0))
                with col_s2:
                    st.markdown("### Team 2")
                    t2 = per_team.get('team2', {})
                    st.metric("Distancia Total", f"{t2.get('total_distance_m', 0):.0f} m")
                    st.metric("Dist. Prom. por Jugador", f"{t2.get('avg_distance_m', 0):.0f} m")
                    st.metric("Velocidad Máxima", f"{t2.get('max_speed_kmh', 0):.1f} km/h")
                    st.metric("Sprints Totales", t2.get('total_sprints', 0))

                # Tabla por jugador con sprints y zonas de intensidad
                if 'per_player' in speed_dist:
                    with st.expander("Ver detalle por jugador"):
                        rows = []
                        for tid, data in speed_dist['per_player'].items():
                            row = {
                                'ID': tid,
                                'Equipo': data['team'],
                                'Distancia (m)': round(data['distance_m'], 1),
                                'Vel. Máx (km/h)': round(data.get('max_speed_kmh', 0), 1),
                                'Sprints': data.get('sprint_count', 0),
                                'Dist. Sprint (m)': round(data.get('sprint_distance_m', 0), 1),
                            }
                            zones = data.get('intensity_zones_m', {})
                            if zones:
                                row['Walking (m)'] = zones.get('walking', 0)
                                row['Jogging (m)'] = zones.get('jogging', 0)
                                row['Running (m)'] = zones.get('running', 0)
                                row['High Int. (m)'] = zones.get('high_intensity', 0)
                                row['Sprint (m)'] = zones.get('sprint', 0)
                            rows.append(row)
                        if rows:
                            df_speed = pd.DataFrame(rows).sort_values('Distancia (m)', ascending=False)
                            st.dataframe(df_speed, use_container_width=True, hide_index=True)
        else:
            if enable_possession:
                st.info("Los datos de posesión aparecerán aquí después de procesar el video.")
            else:
                st.warning("Activa 'Habilitar Análisis de Posesión' en el sidebar antes de procesar.")
else:
    st.info("👈 Sube un video para comenzar el análisis táctico")

# Footer
st.divider()
st.caption("Soccer Analytics AI - Sistema de Análisis Táctico Completo")
st.caption("Tracking + Formaciones + Métricas de Comportamiento Colectivo")
