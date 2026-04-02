import streamlit as st
from src.verticals.home import render_home
from src.verticals.vertical1 import render_vertical1
from src.verticals.vertical2 import render_vertical2

ROUTE_HOME = "home"
ROUTE_VERTICAL1 = "vertical1"
ROUTE_VERTICAL2 = "vertical2"

def _stop_execution() -> None:
    stop_fn = getattr(st, "stop", None)
    if callable(stop_fn):
        stop_fn()

if "active_vertical" not in st.session_state:
    st.session_state.active_vertical = ROUTE_HOME

route = st.session_state.active_vertical

if route == ROUTE_VERTICAL1:
    render_vertical1()
    _stop_execution()
elif route == ROUTE_VERTICAL2:
    st.set_page_config(page_title="Soccer Analytics Platform", layout="wide", initial_sidebar_state="expanded")
    render_vertical2()
    _stop_execution()
else:
    st.set_page_config(page_title="Soccer Analytics Platform", layout="wide", initial_sidebar_state="expanded")
    selected_route = render_home()
    if selected_route:
        st.session_state.active_vertical = selected_route
        st.rerun()
    _stop_execution()

# 2. BALL MODEL
st.sidebar.subheader("Modelo de pelota")
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
st.sidebar.subheader("Radar táctico")
enable_radar = st.sidebar.checkbox("Habilitar Radar", value=True)
enable_analytics = st.sidebar.checkbox("Habilitar Análisis Táctico", value=True,
                                       help="Calcula formaciones y métricas tácticas")
enable_possession = st.sidebar.checkbox("Habilitar Análisis de Posesión", value=False,
                                        help="Posesión de pelota, velocidad y distancia por jugador")
disable_inertia = st.sidebar.checkbox("Modo NO INERCIA (A/B)", value=False,
                                      help="No reutiliza homografía entre frames (max_inertia_frames=0)")

st.sidebar.subheader("Exportación")
export_profile_options = ["summary", "debug_sampled", "full"]
default_export_index = export_profile_options.index(EXPORT_PROFILE) if EXPORT_PROFILE in export_profile_options else 1
export_profile = st.sidebar.selectbox("Perfil de export", export_profile_options, index=default_export_index)
if export_profile != "summary":
    sample_stride = st.sidebar.number_input("Stride de muestreo", min_value=1, max_value=60, value=int(SAMPLE_STRIDE), step=1)
    topk_frames = st.sidebar.number_input("Top K frames", min_value=5, max_value=200, value=int(TOPK_FRAMES), step=5)
else:
    sample_stride = int(SAMPLE_STRIDE)
    topk_frames = int(TOPK_FRAMES)
if export_profile == "full":
    enable_compression = st.sidebar.checkbox("Comprimir JSON (.gz)", value=bool(ENABLE_COMPRESSION))
else:
    enable_compression = False

pitch_model = None
full_field_approx = False

if enable_radar:
    pitch_source = st.sidebar.radio(
        "Modelo de Campo",
        [
            "Homography.pt (32 keypoints) - Recomendado",
            "Soccana Keypoint (29 keypoints)",
            "Aproximación Pantalla Completa (Experimental)"
        ],
        index=0,
        help="Homography.pt detecta más keypoints y tiene mayor tasa de éxito"
    )

    if pitch_source == "Homography.pt (32 keypoints) - Recomendado":
        homography_path = Path("models/homography.pt")
        if homography_path.exists():
            with st.spinner("Cargando modelo Homography (32 keypoints)..."):
                pitch_model = YOLO(str(homography_path))
            st.sidebar.success("Modelo Homography cargado (100% tasa éxito)")
        else:
            st.sidebar.error(f"Modelo no encontrado: {homography_path}")
            st.sidebar.info("Asegúrate de tener el archivo models/homography.pt")

    elif pitch_source == "Soccana Keypoint (29 keypoints)":
        soccana_path = Path("models/soccana_keypoint/Model/weights/best.pt")
        if soccana_path.exists():
            with st.spinner("Cargando modelo Soccana_Keypoint (29 keypoints)..."):
                pitch_model = YOLO(str(soccana_path))
            st.sidebar.success("Modelo Soccana cargado (~65% tasa éxito)")
        else:
            st.sidebar.error("Modelo no encontrado")
            st.sidebar.info("Descarga con: python scripts/download_soccana_model.py")

    else:
        full_field_approx = True

# === MAIN AREA ===
# Initialize session state for stats
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

if st.session_state.stats:
    render_section_title("Session Overview")
    _stats = st.session_state.stats
    _duration = _stats.get("duration_seconds", 0.0)
    _frames = _stats.get("total_frames", 0)
    _fps = _frames / max(1e-6, float(_duration if _duration else 0.0)) if _frames else 0.0
    _col_a, _col_b, _col_c = st.columns(3)
    with _col_a:
        render_status_card("Duración analizada", f"{_duration:.1f} s")
    with _col_b:
        render_status_card("Frames procesados", str(_frames))
    with _col_c:
        render_status_card("Rendimiento", f"{_fps:.1f} FPS")

uploaded_video = st.file_uploader("Cargar video", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save input video
    input_path = INPUTS_DIR / uploaded_video.name
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    # Tabs for organization
    tabs = st.tabs(["Video", "Estadísticas", "Gráficos", "Exportar", "Scouting", "Guía", "Posesión"])

    with tabs[0]:
        col_input, col_output = st.columns(2)

        with col_input:
            render_section_title("Video Source")
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
                            "Descargar video procesado",
                            f,
                            file_name=output_filename,
                            mime="video/mp4"
                        )
            else:
                render_status_card("Estado del procesamiento", "Pendiente de ejecución")
                st.info("Ejecuta el procesamiento para habilitar métricas y exportes.")

    # Process Button
    _radar_mode = "Activado" if enable_radar else "Desactivado"
    _analytics_mode = "Activado" if enable_analytics else "Desactivado"
    _possession_mode = "Activado" if enable_possession else "Desactivado"
    _status_a, _status_b, _status_c = st.columns(3)
    with _status_a:
        render_status_card("Radar táctico", _radar_mode)
    with _status_b:
        render_status_card("Análisis táctico", _analytics_mode)
    with _status_c:
        render_status_card("Posesión", _possession_mode)
    if st.button("Procesar video", type="primary", use_container_width=True):
        if player_model is None:
            st.error("Debes cargar un modelo de jugadores.")
        else:
            with st.spinner("Procesando video..."):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                output_filename = f"processed_{uploaded_video.name}"
                target_path = OUTPUTS_DIR / output_filename
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

                try:
                    status_placeholder.text("Inicializando análisis...")
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
                        enable_possession=enable_possession,
                        disable_inertia=disable_inertia,
                        export_profile=export_profile,
                        sample_stride=sample_stride,
                        topk_frames=topk_frames,
                        enable_compression=enable_compression
                    )

                    progress_bar.progress(90)
                    status_placeholder.text("Generando estadísticas...")

                    # Check if stats file exists
                    stats_path = target_path.parent / f"{target_path.stem}_stats.json"
                    if stats_path.exists():
                        with open(stats_path, 'r') as f:
                            st.session_state.stats = json.load(f)

                    progress_bar.progress(100)
                    status_placeholder.success("Procesamiento completado.")
                    st.session_state.video_processed = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())

    # Statistics Tab
    with tabs[1]:
        if st.session_state.stats:
            stats = st.session_state.stats

            render_section_title("Tactical Overview")
            st.subheader("Análisis táctico")
            if stats.get('health_summary', {}).get('demo_mode') == "degraded":
                st.warning("Demo degradado: métricas aproximadas (homografía no estable).")

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
                st.subheader("Formaciones detectadas")

                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown("### Team 1")
                    raw_form1 = stats['formations'].get('team1', {}).get('most_common', 'N/A')
                    display_form1, approx1 = map_formation_display(raw_form1)
                    st.metric("Formación más común", display_form1, help="Basada en análisis temporal")
                    if approx1:
                        st.caption("Aprox. por detección incompleta o fase mixta.")

                with col_t2:
                    st.markdown("### Team 2")
                    raw_form2 = stats['formations'].get('team2', {}).get('most_common', 'N/A')
                    display_form2, approx2 = map_formation_display(raw_form2)
                    st.metric("Formación más común", display_form2, help="Basada en análisis temporal")
                    if approx2:
                        st.caption("Aprox. por detección incompleta o fase mixta.")

            st.divider()

            # Tactical Metrics
            if 'metrics' in stats:
                st.subheader("Métricas tácticas")

                metrics1 = stats['metrics'].get('team1', {})
                metrics2 = stats['metrics'].get('team2', {})

                # Comparison table
                comparison_data = {
                    'Métrica': ['Presión (m)', 'Amplitud (m)', 'Compactación (m²)'],
                    'Team 1': [
                        format_metric_mean(metrics1.get('pressure_height'), 1),
                        format_metric_mean(metrics1.get('offensive_width'), 1),
                        format_metric_mean(metrics1.get('compactness'), 0)
                    ],
                    'Team 2': [
                        format_metric_mean(metrics2.get('pressure_height'), 1),
                        format_metric_mean(metrics2.get('offensive_width'), 1),
                        format_metric_mean(metrics2.get('compactness'), 0)
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
                                    mean_text, min_text, max_text = format_metric_range(values, 2)
                                    st.text(f"{metric_name}:")
                                    st.text(f"  Media: {mean_text}")
                                    st.text(f"  Min-Max: {min_text} - {max_text}")

                    with col_det2:
                        st.markdown("#### Team 2")
                        if metrics2:
                            for metric_name, values in metrics2.items():
                                if isinstance(values, dict) and 'mean' in values:
                                    mean_text, min_text, max_text = format_metric_range(values, 2)
                                    st.text(f"{metric_name}:")
                                    st.text(f"  Media: {mean_text}")
                                    st.text(f"  Min-Max: {min_text} - {max_text}")

        else:
            st.info("Las estadísticas aparecerán aquí después de procesar el video.")

    # Charts Tab
    with tabs[2]:
        if st.session_state.stats and 'timeline' in st.session_state.stats:
            render_section_title("Temporal Analysis")
            st.subheader("Evolución temporal")

            timeline = st.session_state.stats['timeline']

            # Pressure Height Chart
            if 'team1' in timeline and 'pressure_height' in timeline['team1']:
                frames1_raw = timeline['team1'].get('frame_number', [])
                pressure1_raw = timeline['team1'].get('pressure_height', [])

                frames2_raw = timeline.get('team2', {}).get('frame_number', [])
                pressure2_raw = timeline.get('team2', {}).get('pressure_height', [])

                frames1, pressure1 = filter_series(frames1_raw, pressure1_raw)
                frames2, pressure2 = filter_series(frames2_raw, pressure2_raw)
                if frames1 or frames2:
                    fig_pressure = go.Figure()
                    fig_pressure.add_trace(go.Scatter(x=frames1, y=pressure1, name='Team 1', line=dict(color='green')))
                    fig_pressure.add_trace(go.Scatter(x=frames2, y=pressure2, name='Team 2', line=dict(color='blue')))
                    fig_pressure.update_layout(
                        title='Altura de Presión (m)',
                        xaxis_title='Frame',
                        yaxis_title='Presión (m)',
                        hovermode='x unified'
                    )
                    apply_plotly_dark_theme(fig_pressure)
                    st.plotly_chart(fig_pressure, use_container_width=True)
                else:
                    st.info("Sin datos suficientes en este tramo")

            # Compactness Chart
            if 'team1' in timeline and 'compactness' in timeline['team1']:
                frames1_raw = timeline['team1'].get('frame_number', [])
                compact1_raw = timeline['team1'].get('compactness', [])
                frames2_raw = timeline.get('team2', {}).get('frame_number', [])
                compact2_raw = timeline.get('team2', {}).get('compactness', [])

                frames1, compact1 = filter_series(frames1_raw, compact1_raw)
                frames2, compact2 = filter_series(frames2_raw, compact2_raw)
                if frames1 or frames2:
                    fig_compact = go.Figure()
                    fig_compact.add_trace(go.Scatter(x=frames1, y=compact1, name='Team 1', line=dict(color='green')))
                    fig_compact.add_trace(go.Scatter(x=frames2, y=compact2, name='Team 2', line=dict(color='blue')))
                    fig_compact.update_layout(
                        title='Compactación (m²)',
                        xaxis_title='Frame',
                        yaxis_title='Área (m²)',
                        hovermode='x unified'
                    )
                    apply_plotly_dark_theme(fig_compact)
                    st.plotly_chart(fig_compact, use_container_width=True)
                else:
                    st.info("Sin datos suficientes en este tramo")

            # Width Chart
            if 'team1' in timeline and 'offensive_width' in timeline['team1']:
                frames1_raw = timeline['team1'].get('frame_number', [])
                width1_raw = timeline['team1'].get('offensive_width', [])
                frames2_raw = timeline.get('team2', {}).get('frame_number', [])
                width2_raw = timeline.get('team2', {}).get('offensive_width', [])

                frames1, width1 = filter_series(frames1_raw, width1_raw)
                frames2, width2 = filter_series(frames2_raw, width2_raw)
                if frames1 or frames2:
                    fig_width = go.Figure()
                    fig_width.add_trace(go.Scatter(x=frames1, y=width1, name='Team 1', line=dict(color='green')))
                    fig_width.add_trace(go.Scatter(x=frames2, y=width2, name='Team 2', line=dict(color='blue')))
                    fig_width.update_layout(
                        title='Amplitud Ofensiva (m)',
                        xaxis_title='Frame',
                        yaxis_title='Amplitud (m)',
                        hovermode='x unified'
                    )
                    apply_plotly_dark_theme(fig_width)
                    st.plotly_chart(fig_width, use_container_width=True)
                else:
                    st.info("Sin datos suficientes en este tramo")

        else:
            st.info("Los gráficos aparecerán aquí después de procesar el video con análisis táctico.")

    # Export Tab
    with tabs[3]:
        if st.session_state.stats:
            render_section_title("Data Exports")
            st.subheader("Exportar datos")
            exp_a, exp_b, exp_c = st.columns(3)
            with exp_a:
                render_status_card("Formato JSON", "Disponible")
            with exp_b:
                render_status_card("Formato CSV", "Disponible")
            with exp_c:
                render_status_card("Formato PDF", "Disponible")

            col_json, col_csv = st.columns(2)

            with col_json:
                st.markdown("### JSON")
                json_str = json.dumps(st.session_state.stats, indent=2)
                st.download_button(
                    label="Descargar JSON",
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
                            label="Descargar CSV (Team 1)",
                            data=csv,
                            file_name="team1_timeline.csv",
                            mime="text/csv"
                        )

            st.divider()
            st.markdown("### PDF")
            pdf_bytes = generate_scouting_pdf(
                st.session_state.stats,
                uploaded_video.name if uploaded_video else None,
                use_log=st.session_state.get("heatmap_use_log", False),
                flip_vertical=st.session_state.get("heatmap_flip_vertical", True)
            )
            st.download_button(
                label="Exportar PDF",
                data=pdf_bytes,
                file_name="reporte_scouting.pdf",
                mime="application/pdf"
            )

        else:
            st.info("Las opciones de exportación aparecerán aquí después de procesar el video.")

    with tabs[4]:
        if st.session_state.stats and 'scouting_heatmaps' in st.session_state.stats:
            stats = st.session_state.stats
            render_section_title("Scouting Analysis")
            st.subheader("Scouting")
            st.markdown("### Resumen automático del clip")
            metrics = stats.get('metrics', {})
            timeline = stats.get('timeline', {})
            health_summary = stats.get('health_summary', {})
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
                    valid_frames = health_summary.get('valid_frames', None)
                if valid_frames is None:
                    valid_frames = len(timeline.get(team_key, {}).get('frame_number', []))
                return valid_frames
            def compute_confidence(team_key: str) -> tuple:
                score = 100.0
                fallback_ratio = health_summary.get("fallback_ratio", None)
                invalid_formation_ratio = health_summary.get("invalid_formation_ratio", None)
                p95_reproj_error_m = health_summary.get("p95_reproj_error_m", None)
                avg_short_tracks_ratio = health_summary.get("avg_short_tracks_ratio", None)
                invalid_ratio = health_summary.get("invalid_ratio", None)
                if fallback_ratio is not None and fallback_ratio > 0.05:
                    score -= 20
                if invalid_formation_ratio is not None and invalid_formation_ratio > 0.10:
                    score -= 20
                if p95_reproj_error_m is not None and p95_reproj_error_m > 1.5:
                    score -= 20
                if avg_short_tracks_ratio is not None and avg_short_tracks_ratio > 0.4:
                    score -= 15
                if invalid_ratio is not None and invalid_ratio > 0.20:
                    score -= 15
                score = max(0.0, min(100.0, score))
                if score >= 80:
                    return "Alta", score
                if score >= 60:
                    return "Media", score
                return "Baja", score
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
                        st.success(f"Confianza: {level1} ({ratio1:.0f}/100)")
                    elif level1 == "Media":
                        st.info(f"Confianza: {level1} ({ratio1:.0f}/100)")
                    else:
                        st.warning(f"Confianza: {level1} ({ratio1:.0f}/100)")
                    if level1 == "Baja":
                        st.warning("Interpretación limitada: pocos frames válidos...")
                        st.caption("Pocos frames válidos: posible jitter de homografía / tracking.")
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
                        st.success(f"Confianza: {level2} ({ratio2:.0f}/100)")
                    elif level2 == "Media":
                        st.info(f"Confianza: {level2} ({ratio2:.0f}/100)")
                    else:
                        st.warning(f"Confianza: {level2} ({ratio2:.0f}/100)")
                    if level2 == "Baja":
                        st.warning("Interpretación limitada: pocos frames válidos...")
                        st.caption("Pocos frames válidos: posible jitter de homografía / tracking.")
                    elif level2 == "Media":
                        st.info("Interpretación moderada...")
                else:
                    st.info("Confianza: N/D")
                for item in build_summary('team2', level2):
                    st.markdown(f"- {item}")
            fallback_ratio = health_summary.get("fallback_ratio", None)
            invalid_formation_ratio = health_summary.get("invalid_formation_ratio", None)
            p95_reproj_error_m = health_summary.get("p95_reproj_error_m", None)
            p95_churn_ratio = health_summary.get("p95_churn_ratio", None)
            churn_warn_ratio = health_summary.get("churn_warn_ratio", None)
            p95_max_speed_mps = health_summary.get("p95_max_speed_mps", None)
            speed_violation_ratio = health_summary.get("speed_violation_ratio", None)
            p95_max_jump_m = health_summary.get("p95_max_jump_m", None)
            jump_violation_ratio = health_summary.get("jump_violation_ratio", None)
            if (
                fallback_ratio is not None
                or invalid_formation_ratio is not None
                or p95_reproj_error_m is not None
                or p95_churn_ratio is not None
                or p95_max_speed_mps is not None
                or p95_max_jump_m is not None
            ):
                st.markdown("**Indicadores de salud**")
                if fallback_ratio is not None:
                    st.markdown(f"- Fallback ratio: {fallback_ratio * 100:.1f}%")
                if invalid_formation_ratio is not None:
                    st.markdown(f"- Invalid formation ratio: {invalid_formation_ratio * 100:.1f}%")
                if p95_reproj_error_m is not None:
                    st.markdown(f"- P95 reproj error (m): {p95_reproj_error_m:.2f}")
                if p95_churn_ratio is not None:
                    st.markdown(f"- P95 churn ratio: {p95_churn_ratio * 100:.1f}%")
                if churn_warn_ratio is not None:
                    st.markdown(f"- Churn warn ratio: {churn_warn_ratio * 100:.1f}%")
                if p95_max_speed_mps is not None:
                    st.markdown(f"- P95 max speed (m/s): {p95_max_speed_mps:.2f}")
                if speed_violation_ratio is not None:
                    st.markdown(f"- Speed violations: {speed_violation_ratio * 100:.1f}%")
                if p95_max_jump_m is not None:
                    st.markdown(f"- P95 max jump (m): {p95_max_jump_m:.2f}")
                if jump_violation_ratio is not None:
                    st.markdown(f"- Jump violations: {jump_violation_ratio * 100:.1f}%")
            st.caption("Qué mirar primero: (1) compactación, (2) línea defensiva, (3) heatmap.")
            col_t1, col_t2 = st.columns(2)
            if 'timeline' in stats:
                timeline = stats['timeline']
                with col_t1:
                    st.markdown("### Team 1")
                    frames = timeline.get('team1', {}).get('frame_number', [])
                    depth = timeline.get('team1', {}).get('block_depth_m', [])
                    width = timeline.get('team1', {}).get('block_width_m', [])
                    left = timeline.get('team1', {}).get('def_line_left_m', [])
                    right = timeline.get('team1', {}).get('def_line_right_m', [])
                    if len(frames) > 0:
                        depth_x, depth_vals = filter_series(frames, depth)
                        width_x, width_vals = filter_series(frames, width)
                        left_x, left_vals = filter_series(frames, left)
                        right_x, right_vals = filter_series(frames, right)
                        x_vals = depth_x or width_x or left_x or right_x
                        x_title = "Frame"
                        tickvals = None
                        ticktext = None
                        if fps:
                            if depth_x:
                                depth_x = [f / fps for f in depth_x]
                            if width_x:
                                width_x = [f / fps for f in width_x]
                            if left_x:
                                left_x = [f / fps for f in left_x]
                            if right_x:
                                right_x = [f / fps for f in right_x]
                            x_vals = depth_x or width_x or left_x or right_x
                            x_title = "Tiempo (mm:ss)"
                            if len(x_vals) > 1:
                                step = max(1, int(len(x_vals) / 6))
                                tickvals = x_vals[::step]
                                ticktext = [format_mmss(t) for t in tickvals]
                        st.markdown("#### Compactación")
                        st.caption("Qué es: tamaño del bloque del equipo. Profundidad = largo (def→ata). Ancho = apertura lateral.")
                        st.caption("Cómo leer: picos = equipo se estira (transición). valles = equipo compacto (bloque).")
                        if depth_x or width_x:
                            fig_c = go.Figure()
                            if depth_x:
                                fig_c.add_trace(go.Scatter(x=depth_x, y=depth_vals, name='Profundidad (m)', line=dict(color='green')))
                            if width_x:
                                fig_c.add_trace(go.Scatter(x=width_x, y=width_vals, name='Ancho (m)', line=dict(color='darkgreen')))
                            fig_c.update_layout(title='Compactación', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                            apply_plotly_dark_theme(fig_c)
                            if tickvals:
                                fig_c.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                            st.plotly_chart(fig_c, use_container_width=True)
                        else:
                            st.info("Sin datos suficientes en este tramo")
                        st.markdown("#### Línea defensiva")
                        st.caption("Qué es: altura del último bloque en X (0–105m).")
                        st.caption("Cómo leer: sube = línea alta/presión; baja = repliegue/bloque bajo.")
                        st.caption("Nota: mostramos dos hipótesis (izq/der) por orientación broadcast.")
                        if left_x or right_x:
                            fig_d = go.Figure()
                            if left_x:
                                fig_d.add_trace(go.Scatter(x=left_x, y=left_vals, name='Línea Izquierda (m)', line=dict(color='blue')))
                            if right_x:
                                fig_d.add_trace(go.Scatter(x=right_x, y=right_vals, name='Línea Derecha (m)', line=dict(color='darkblue')))
                            fig_d.update_layout(title='Línea Defensiva', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                            apply_plotly_dark_theme(fig_d)
                            if tickvals:
                                fig_d.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                            st.plotly_chart(fig_d, use_container_width=True)
                        else:
                            st.info("Sin datos suficientes en este tramo")
                with col_t2:
                    st.markdown("### Team 2")
                    frames = timeline.get('team2', {}).get('frame_number', [])
                    depth = timeline.get('team2', {}).get('block_depth_m', [])
                    width = timeline.get('team2', {}).get('block_width_m', [])
                    left = timeline.get('team2', {}).get('def_line_left_m', [])
                    right = timeline.get('team2', {}).get('def_line_right_m', [])
                    if len(frames) > 0:
                        depth_x, depth_vals = filter_series(frames, depth)
                        width_x, width_vals = filter_series(frames, width)
                        left_x, left_vals = filter_series(frames, left)
                        right_x, right_vals = filter_series(frames, right)
                        x_vals = depth_x or width_x or left_x or right_x
                        x_title = "Frame"
                        tickvals = None
                        ticktext = None
                        if fps:
                            if depth_x:
                                depth_x = [f / fps for f in depth_x]
                            if width_x:
                                width_x = [f / fps for f in width_x]
                            if left_x:
                                left_x = [f / fps for f in left_x]
                            if right_x:
                                right_x = [f / fps for f in right_x]
                            x_vals = depth_x or width_x or left_x or right_x
                            x_title = "Tiempo (mm:ss)"
                            if len(x_vals) > 1:
                                step = max(1, int(len(x_vals) / 6))
                                tickvals = x_vals[::step]
                                ticktext = [format_mmss(t) for t in tickvals]
                        st.markdown("#### Compactación")
                        st.caption("Qué es: tamaño del bloque del equipo. Profundidad = largo (def→ata). Ancho = apertura lateral.")
                        st.caption("Cómo leer: picos = equipo se estira (transición). valles = equipo compacto (bloque).")
                        if depth_x or width_x:
                            fig_c = go.Figure()
                            if depth_x:
                                fig_c.add_trace(go.Scatter(x=depth_x, y=depth_vals, name='Profundidad (m)', line=dict(color='green')))
                            if width_x:
                                fig_c.add_trace(go.Scatter(x=width_x, y=width_vals, name='Ancho (m)', line=dict(color='darkgreen')))
                            fig_c.update_layout(title='Compactación', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                            apply_plotly_dark_theme(fig_c)
                            if tickvals:
                                fig_c.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                            st.plotly_chart(fig_c, use_container_width=True)
                        else:
                            st.info("Sin datos suficientes en este tramo")
                        st.markdown("#### Línea defensiva")
                        st.caption("Qué es: altura del último bloque en X (0–105m).")
                        st.caption("Cómo leer: sube = línea alta/presión; baja = repliegue/bloque bajo.")
                        st.caption("Nota: mostramos dos hipótesis (izq/der) por orientación broadcast.")
                        if left_x or right_x:
                            fig_d = go.Figure()
                            if left_x:
                                fig_d.add_trace(go.Scatter(x=left_x, y=left_vals, name='Línea Izquierda (m)', line=dict(color='blue')))
                            if right_x:
                                fig_d.add_trace(go.Scatter(x=right_x, y=right_vals, name='Línea Derecha (m)', line=dict(color='darkblue')))
                            fig_d.update_layout(title='Línea Defensiva', xaxis_title=x_title, yaxis_title='Metros', hovermode='x unified')
                            apply_plotly_dark_theme(fig_d)
                            if tickvals:
                                fig_d.update_xaxes(tickvals=tickvals, ticktext=ticktext)
                            st.plotly_chart(fig_d, use_container_width=True)
                        else:
                            st.info("Sin datos suficientes en este tramo")
            st.subheader("Heatmaps")
            st.caption("Zonas calientes = mayor permanencia/ocupación en el clip.")
            flip_vertical = st.checkbox("Invertir eje vertical (Y)", value=True)
            use_log = st.checkbox("Usar escala logarítmica", value=False)
            st.session_state["heatmap_flip_vertical"] = flip_vertical
            st.session_state["heatmap_use_log"] = use_log
            st.caption("Escala log: hace visibles zonas con poca presencia cuando hay una zona dominante.")
            col_h1, col_h2 = st.columns(2)
            heatmap_meta = stats.get('scouting_heatmaps', {})
            out_w = 840
            out_h = int(out_w * 68 / 105)
            legend = draw_heatmap_legend(out_h)
            with col_h1:
                h1 = heatmap_meta.get('team1', None)
                st.markdown("#### Heatmap")
                st.caption("Intensidad = cantidad de presencia en este clip (no es ‘calor’, es frecuencia).")
                if isinstance(h1, dict):
                    h1 = h1.get("downsampled", None)
                approx = False
                if h1 is not None:
                    heatmap = np.array(h1, dtype=np.float32)
                else:
                    heatmap = build_centroid_heatmap(stats.get('homography_telemetry', {}), "team1")
                    if heatmap is not None:
                        approx = True
                if heatmap is None:
                    st.info("Sin datos suficientes en este tramo")
                    st.image(draw_pitch_base(out_w, out_h), caption="Heatmap Team 1", use_column_width=True)
                else:
                    rendered = render_heatmap_overlay(heatmap, out_w, out_h, flip_vertical, use_log)
                    combo = np.concatenate([rendered, legend], axis=1)
                    caption = "Heatmap Team 1" + (" (aprox)" if approx else "")
                    st.image(combo, caption=caption, use_column_width=True)
                if isinstance(heatmap_meta.get('team1'), dict):
                    st.caption(f"bins_shape={heatmap_meta.get('bins_shape')} | sample_rate={heatmap_meta.get('sample_rate')} | total_samples={heatmap_meta.get('total_samples')}")
            with col_h2:
                h2 = heatmap_meta.get('team2', None)
                st.markdown("#### Heatmap")
                st.caption("Intensidad = cantidad de presencia en este clip (no es ‘calor’, es frecuencia).")
                if isinstance(h2, dict):
                    h2 = h2.get("downsampled", None)
                approx = False
                if h2 is not None:
                    heatmap = np.array(h2, dtype=np.float32)
                else:
                    heatmap = build_centroid_heatmap(stats.get('homography_telemetry', {}), "team2")
                    if heatmap is not None:
                        approx = True
                if heatmap is None:
                    st.info("Sin datos suficientes en este tramo")
                    st.image(draw_pitch_base(out_w, out_h), caption="Heatmap Team 2", use_column_width=True)
                else:
                    rendered = render_heatmap_overlay(heatmap, out_w, out_h, flip_vertical, use_log)
                    combo = np.concatenate([rendered, legend], axis=1)
                    caption = "Heatmap Team 2" + (" (aprox)" if approx else "")
                    st.image(combo, caption=caption, use_column_width=True)
                if isinstance(heatmap_meta.get('team2'), dict):
                    st.caption(f"bins_shape={heatmap_meta.get('bins_shape')} | sample_rate={heatmap_meta.get('sample_rate')} | total_samples={heatmap_meta.get('total_samples')}")
        else:
            st.warning("Scouting metrics not available for this video.")
    with tabs[5]:
        render_section_title("Interpretation Guide")
        st.subheader("Guía de interpretación")
        st.markdown(INTERPRETATION_MARKDOWN)

    # Possession Tab
    with tabs[6]:
        if st.session_state.stats and 'possession' in st.session_state.stats:
            stats = st.session_state.stats
            possession = stats['possession']
            speed_dist = stats.get('speed_distance', {})

            render_section_title("Possession & Physical Output")
            st.subheader("Posesión de pelota")

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
            apply_plotly_dark_theme(fig_pie)
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
                apply_plotly_dark_theme(fig_timeline)
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
                    t1_max, t1_capped = cap_speed_display(t1.get('max_speed_kmh', 0))
                    st.metric("Velocidad Máxima", f"{t1_max:.1f} km/h")
                    if t1_capped:
                        st.caption("Velocidad cappeada por plausibilidad (beta / tracking).")
                    st.metric("Sprints Totales", t1.get('total_sprints', 0))
                with col_s2:
                    st.markdown("### Team 2")
                    t2 = per_team.get('team2', {})
                    st.metric("Distancia Total", f"{t2.get('total_distance_m', 0):.0f} m")
                    st.metric("Dist. Prom. por Jugador", f"{t2.get('avg_distance_m', 0):.0f} m")
                    t2_max, t2_capped = cap_speed_display(t2.get('max_speed_kmh', 0))
                    st.metric("Velocidad Máxima", f"{t2_max:.1f} km/h")
                    if t2_capped:
                        st.caption("Velocidad cappeada por plausibilidad (beta / tracking).")
                    st.metric("Sprints Totales", t2.get('total_sprints', 0))

                # Tabla por jugador con sprints y zonas de intensidad
                if 'per_player' in speed_dist:
                    with st.expander("Ver detalle por jugador"):
                        rows = []
                        for tid, data in speed_dist['per_player'].items():
                            max_speed, capped = cap_speed_display(data.get('max_speed_kmh', 0))
                            row = {
                                'ID': tid,
                                'Equipo': data['team'],
                                'Distancia (m)': round(data['distance_m'], 1),
                                'Vel. Máx (km/h)': round(max_speed, 1),
                                'Cappeada': "Sí" if capped else "No",
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
    render_status_card("Estado de sesión", "Sin video cargado")
    st.info("Carga un video para comenzar el análisis táctico.")

# Footer
st.divider()
st.markdown(
    """
    <div class="platform-footer">
        <span><strong>FTI Platform</strong></span>
        <span class="footer-pill">Premium Tactical Analytics</span>
        <span>Tracking avanzado · Scouting táctico · Exportes ejecutivos</span>
    </div>
    """,
    unsafe_allow_html=True,
)
