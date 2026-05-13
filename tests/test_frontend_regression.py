import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class SessionState(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeUploadedFile:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload

    def read(self):
        return self._payload


class StopExecution(Exception):
    pass


class FakeProgress:
    def __init__(self, recorder):
        self.recorder = recorder

    def progress(self, value):
        self.recorder.progress_values.append(value)


class FakePlaceholder:
    def __init__(self, recorder):
        self.recorder = recorder

    def text(self, value):
        self.recorder.placeholder_texts.append(str(value))

    def success(self, value):
        self.recorder.success_messages.append(str(value))


class FakeSidebar:
    def __init__(self, recorder, config):
        self.recorder = recorder
        self.config = config

    def header(self, text):
        self.recorder.sidebar_headers.append(str(text))

    def subheader(self, text):
        self.recorder.sidebar_subheaders.append(str(text))

    def radio(self, label, options, index=0, **kwargs):
        value = self.config.get("radio", {}).get(label, options[index])
        self.recorder.sidebar_radios.append((label, list(options), value))
        return value

    def selectbox(self, label, options, index=0, **kwargs):
        value = self.config.get("selectbox", {}).get(label, options[index])
        self.recorder.sidebar_selectboxes.append((label, list(options), value))
        return value

    def number_input(self, label, min_value=None, max_value=None, value=0, step=1, **kwargs):
        selected = self.config.get("number_input", {}).get(label, value)
        self.recorder.sidebar_number_inputs.append((label, selected))
        return selected

    def checkbox(self, label, value=False, **kwargs):
        selected = self.config.get("checkbox", {}).get(label, value)
        self.recorder.sidebar_checkboxes.append((label, selected))
        return selected

    def file_uploader(self, label, type=None, **kwargs):
        self.recorder.sidebar_file_uploaders.append(label)
        return self.config.get("file_uploader", {}).get(label, None)

    def button(self, label, **kwargs):
        return self.config.get("button", {}).get(label, False)

    def success(self, text):
        self.recorder.success_messages.append(str(text))

    def error(self, text):
        self.recorder.error_messages.append(str(text))

    def info(self, text):
        self.recorder.info_messages.append(str(text))


class StreamlitRecorder:
    def __init__(self, config):
        self.config = config
        self.session_state = SessionState(config.get("session_state", {}))
        self.page_config_calls = 0
        self.markdowns = []
        self.captions = []
        self.headers = []
        self.subheaders = []
        self.metrics = []
        self.file_uploaders = []
        self.tabs_labels = []
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []
        self.success_messages = []
        self.placeholder_texts = []
        self.progress_values = []
        self.plotly_calls = 0
        self.plotly_keys = []
        self.text_inputs = []
        self.json_payloads = []
        self.expander_labels = []
        self.dataframe_calls = 0
        self.download_buttons = []
        self.sidebar_headers = []
        self.sidebar_subheaders = []
        self.sidebar_radios = []
        self.sidebar_selectboxes = []
        self.sidebar_number_inputs = []
        self.sidebar_checkboxes = []
        self.sidebar_file_uploaders = []
        self.sidebar = FakeSidebar(self, config.get("sidebar", {}))

    def cache_resource(self, fn=None, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    def set_page_config(self, **kwargs):
        self.page_config_calls += 1

    def markdown(self, text, **kwargs):
        self.markdowns.append(str(text))

    def caption(self, text):
        self.captions.append(str(text))

    def header(self, text):
        self.headers.append(str(text))

    def title(self, text):
        self.headers.append(str(text))

    def subheader(self, text):
        self.subheaders.append(str(text))

    def metric(self, label, value, **kwargs):
        self.metrics.append((str(label), str(value)))

    def file_uploader(self, label, type=None, **kwargs):
        self.file_uploaders.append(label)
        return self.config.get("uploaded_video", None)

    def tabs(self, labels):
        self.tabs_labels = list(labels)
        return [FakeContext() for _ in labels]

    def columns(self, spec):
        if isinstance(spec, int):
            count = spec
        else:
            count = len(spec)
        return [FakeContext() for _ in range(count)]

    def video(self, *args, **kwargs):
        return None

    def download_button(self, label, data, **kwargs):
        self.download_buttons.append(label)
        return False

    def button(self, label, **kwargs):
        button_config = self.config.get("button", {})
        if isinstance(button_config, dict):
            if label in button_config:
                return button_config[label]
        return self.config.get("button_clicked", False)

    def text_input(self, label, value="", **kwargs):
        selected = self.config.get("text_input", {}).get(label, value)
        self.text_inputs.append((label, selected))
        return selected

    def checkbox(self, label, value=False, **kwargs):
        return self.config.get("checkbox", {}).get(label, value)

    def spinner(self, text):
        return FakeContext()

    def empty(self):
        return FakePlaceholder(self)

    def progress(self, value):
        self.progress_values.append(value)
        return FakeProgress(self)

    def rerun(self):
        return None

    def stop(self):
        raise StopExecution()

    def divider(self):
        return None

    def dataframe(self, *args, **kwargs):
        self.dataframe_calls += 1

    def plotly_chart(self, *args, **kwargs):
        self.plotly_calls += 1
        self.plotly_keys.append(kwargs.get("key"))

    def image(self, *args, **kwargs):
        return None

    def info(self, text):
        self.info_messages.append(str(text))

    def warning(self, text):
        self.warning_messages.append(str(text))

    def error(self, text):
        self.error_messages.append(str(text))

    def success(self, text):
        self.success_messages.append(str(text))

    def expander(self, label, **kwargs):
        self.expander_labels.append(str(label))
        return FakeContext()

    def text(self, value):
        self.markdowns.append(str(value))

    def code(self, value, **kwargs):
        self.markdowns.append(str(value))

    def json(self, value, **kwargs):
        self.json_payloads.append(value)


def make_streamlit_module(config):
    recorder = StreamlitRecorder(config)
    module = ModuleType("streamlit")
    module.session_state = recorder.session_state
    module.sidebar = recorder.sidebar
    module.cache_resource = recorder.cache_resource
    module.set_page_config = recorder.set_page_config
    module.markdown = recorder.markdown
    module.caption = recorder.caption
    module.header = recorder.header
    module.title = recorder.title
    module.subheader = recorder.subheader
    module.metric = recorder.metric
    module.file_uploader = recorder.file_uploader
    module.tabs = recorder.tabs
    module.columns = recorder.columns
    module.video = recorder.video
    module.download_button = recorder.download_button
    module.button = recorder.button
    module.text_input = recorder.text_input
    module.checkbox = recorder.checkbox
    module.spinner = recorder.spinner
    module.empty = recorder.empty
    module.progress = recorder.progress
    module.rerun = recorder.rerun
    module.stop = recorder.stop
    module.divider = recorder.divider
    module.dataframe = recorder.dataframe
    module.plotly_chart = recorder.plotly_chart
    module.image = recorder.image
    module.info = recorder.info
    module.warning = recorder.warning
    module.error = recorder.error
    module.success = recorder.success
    module.expander = recorder.expander
    module.text = recorder.text
    module.code = recorder.code
    module.json = recorder.json
    return module, recorder


def make_ultralytics_module():
    module = ModuleType("ultralytics")

    class DummyYOLO:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    module.YOLO = DummyYOLO
    return module


def make_process_video_module():
    module = ModuleType("src.controllers.process_video")

    def process_video(*args, **kwargs):
        return None

    module.process_video = process_video
    return module


def make_reportlab_modules():
    reportlab = ModuleType("reportlab")
    lib = ModuleType("reportlab.lib")
    pagesizes = ModuleType("reportlab.lib.pagesizes")
    pagesizes.A4 = (595.0, 842.0)
    colors = ModuleType("reportlab.lib.colors")
    colors.lightgrey = "lightgrey"
    colors.black = "black"
    colors.grey = "grey"
    utils = ModuleType("reportlab.lib.utils")

    class DummyImageReader:
        def __init__(self, _):
            pass

        def getSize(self):
            return (100.0, 100.0)

    utils.ImageReader = DummyImageReader

    pdfgen = ModuleType("reportlab.pdfgen")
    canvas_module = ModuleType("reportlab.pdfgen.canvas")

    class DummyCanvas:
        def __init__(self, *args, **kwargs):
            pass

        def showPage(self):
            return None

        def setFont(self, *args, **kwargs):
            return None

        def drawString(self, *args, **kwargs):
            return None

        def drawCentredString(self, *args, **kwargs):
            return None

        def drawImage(self, *args, **kwargs):
            return None

        def save(self):
            return None

    canvas_module.Canvas = DummyCanvas
    pdfgen.canvas = canvas_module

    platypus = ModuleType("reportlab.platypus")

    class DummyTable:
        def __init__(self, *args, **kwargs):
            pass

        def setStyle(self, *args, **kwargs):
            return None

        def wrap(self, *args, **kwargs):
            return (100.0, 40.0)

        def drawOn(self, *args, **kwargs):
            return None

    class DummyTableStyle:
        def __init__(self, *args, **kwargs):
            pass

    platypus.Table = DummyTable
    platypus.TableStyle = DummyTableStyle

    return {
        "reportlab": reportlab,
        "reportlab.lib": lib,
        "reportlab.lib.pagesizes": pagesizes,
        "reportlab.lib.colors": colors,
        "reportlab.lib.utils": utils,
        "reportlab.pdfgen": pdfgen,
        "reportlab.pdfgen.canvas": canvas_module,
        "reportlab.platypus": platypus,
    }


def run_app(monkeypatch, config):
    session_state = dict(config.get("session_state", {}))
    session_state.setdefault("active_vertical", "vertical1")
    config = dict(config)
    config["session_state"] = session_state
    st_module, recorder = make_streamlit_module(config)
    monkeypatch.setitem(sys.modules, "streamlit", st_module)
    monkeypatch.setitem(sys.modules, "ultralytics", make_ultralytics_module())
    monkeypatch.setitem(sys.modules, "src.controllers.process_video", make_process_video_module())
    for module_name, module_obj in make_reportlab_modules().items():
        monkeypatch.setitem(sys.modules, module_name, module_obj)
    for name in [
        "app",
        "src.models.load_model",
        "src.utils.ui.theme",
        "src.utils.ui.ai_coach_panel",
        "src.verticals.home",
        "src.verticals.vertical1",
        "src.verticals.vertical2",
        "src.verticals.vertical2_api_event",
        "src.verticals.vertical1_legacy",
    ]:
        if name in sys.modules:
            del sys.modules[name]
    try:
        importlib.import_module("app")
    except StopExecution:
        pass
    return recorder


def load_vertical2_api_event(monkeypatch, config=None):
    config = config or {"session_state": {}}
    session_state = dict(config.get("session_state", {}))
    config = dict(config)
    config["session_state"] = session_state
    st_module, recorder = make_streamlit_module(config)
    monkeypatch.setitem(sys.modules, "streamlit", st_module)
    if "src.utils.ui.ai_coach_panel" in sys.modules:
        del sys.modules["src.utils.ui.ai_coach_panel"]
    if "src.verticals.vertical2_api_event" in sys.modules:
        del sys.modules["src.verticals.vertical2_api_event"]
    module = importlib.import_module("src.verticals.vertical2_api_event")
    return module, recorder


def build_full_stats():
    return {
        "duration_seconds": 12.0,
        "total_frames": 360,
        "formations": {
            "team1": {"most_common": "4-4-2"},
            "team2": {"most_common": "4-3-3"},
        },
        "metrics": {
            "team1": {
                "pressure_height": {"mean": 35.0, "min": 20.0, "max": 50.0},
                "offensive_width": {"mean": 42.0, "min": 30.0, "max": 55.0},
                "compactness": {"mean": 900.0, "min": 700.0, "max": 1200.0},
                "block_depth_m": {"mean": 32.0, "min": 25.0, "max": 39.0},
                "block_width_m": {"mean": 30.0, "min": 22.0, "max": 37.0},
                "def_line_left_m": {"mean": 54.0, "min": 40.0, "max": 66.0},
                "def_line_right_m": {"mean": 56.0, "min": 42.0, "max": 68.0},
            },
            "team2": {
                "pressure_height": {"mean": 38.0, "min": 21.0, "max": 53.0},
                "offensive_width": {"mean": 40.0, "min": 29.0, "max": 52.0},
                "compactness": {"mean": 880.0, "min": 690.0, "max": 1100.0},
                "block_depth_m": {"mean": 31.0, "min": 24.0, "max": 38.0},
                "block_width_m": {"mean": 29.0, "min": 21.0, "max": 36.0},
                "def_line_left_m": {"mean": 53.0, "min": 39.0, "max": 65.0},
                "def_line_right_m": {"mean": 55.0, "min": 41.0, "max": 67.0},
            },
        },
        "timeline": {
            "team1": {
                "frame_number": [1, 2, 3, 4, 5, 6],
                "pressure_height": [30, 32, 34, 35, 36, 38],
                "compactness": [900, 890, 870, 860, 855, 845],
                "offensive_width": [40, 41, 42, 43, 42, 41],
                "block_depth_m": [31, 32, 33, 34, 33, 32],
                "block_width_m": [29, 30, 30, 31, 30, 29],
                "def_line_left_m": [50, 52, 54, 55, 56, 57],
                "def_line_right_m": [52, 54, 56, 57, 58, 59],
            },
            "team2": {
                "frame_number": [1, 2, 3, 4, 5, 6],
                "pressure_height": [40, 39, 38, 37, 36, 35],
                "compactness": [850, 860, 870, 875, 885, 890],
                "offensive_width": [38, 39, 40, 40, 41, 42],
                "block_depth_m": [30, 31, 31, 32, 32, 33],
                "block_width_m": [28, 28, 29, 29, 30, 30],
                "def_line_left_m": [49, 50, 51, 52, 53, 54],
                "def_line_right_m": [51, 52, 53, 54, 55, 56],
            },
        },
        "scouting_heatmaps": {
            "team1": {"downsampled": [[1.0, 2.0], [3.0, 4.0]]},
            "team2": {"downsampled": [[1.0, 1.0], [1.0, 1.0]]},
            "bins_shape": [26, 17],
            "sample_rate": 10,
            "total_samples": 120,
        },
        "health_summary": {
            "fallback_ratio": 0.1,
            "invalid_formation_ratio": 0.1,
            "p95_reproj_error_m": 1.0,
            "p95_churn_ratio": 0.2,
            "churn_warn_ratio": 0.1,
            "p95_max_speed_mps": 8.0,
            "speed_violation_ratio": 0.02,
            "p95_max_jump_m": 1.2,
            "jump_violation_ratio": 0.01,
            "demo_mode": "stable",
        },
        "possession": {
            "team1_possession_pct": 54.0,
            "team2_possession_pct": 40.0,
            "contested_frames": 20,
            "timeline": ["team1"] * 30 + ["team2"] * 30,
            "passes": {"total": 10, "team1_passes": 6, "team2_passes": 4, "turnovers": 3},
            "top_possessors": [[10, 20, "team1"], [8, 18, "team2"]],
        },
        "speed_distance": {
            "per_team": {
                "team1": {"total_distance_m": 1000, "avg_distance_m": 100, "max_speed_kmh": 29, "total_sprints": 8},
                "team2": {"total_distance_m": 980, "avg_distance_m": 98, "max_speed_kmh": 30, "total_sprints": 9},
            },
            "per_player": {
                "p1": {
                    "team": "team1",
                    "distance_m": 120.5,
                    "max_speed_kmh": 31.2,
                    "sprint_count": 2,
                    "sprint_distance_m": 20.3,
                    "intensity_zones_m": {"walking": 40, "jogging": 30, "running": 25, "high_intensity": 15, "sprint": 10},
                }
            },
        },
        "homography_telemetry": {
            "team1_centroid_x": [10, 20, 30],
            "team1_centroid_y": [10, 20, 30],
            "team2_centroid_x": [80, 70, 60],
            "team2_centroid_y": [50, 40, 30],
            "homography_mode": ["track", "inertia"],
        },
    }


def test_smoke_app_imports_without_video(monkeypatch):
    recorder = run_app(monkeypatch, {"uploaded_video": None, "session_state": {}})
    assert recorder.page_config_calls == 1
    assert "Cargar video" in recorder.file_uploaders
    assert any("Sin video cargado" in item for item in recorder.markdowns)


def test_router_home_does_not_execute_vertical1_legacy_block(monkeypatch):
    recorder = run_app(monkeypatch, {"uploaded_video": None, "session_state": {"active_vertical": "home"}})
    assert any("tip-title" in item for item in recorder.markdowns)
    assert any("tip-inline-tip" in item for item in recorder.markdowns)
    assert any("TIP" in item for item in recorder.markdowns)
    assert not any("Selecciona una vertical para continuar" in item for item in recorder.markdowns)
    assert recorder.sidebar_subheaders == []
    assert recorder.file_uploaders == []


def test_router_vertical2_does_not_execute_vertical1_legacy_block(monkeypatch):
    recorder = run_app(monkeypatch, {"uploaded_video": None, "session_state": {"active_vertical": "vertical2"}})
    assert "Data Analytics" in recorder.headers
    assert any("Sube un PDF Wyscout" in msg for msg in recorder.info_messages)
    assert "Subir reporte Wyscout (.pdf)" in recorder.file_uploaders
    assert recorder.tabs_labels == ["API Event Data", "Load PDF"]
    assert "API Event Data" in recorder.subheaders
    assert recorder.sidebar_subheaders == []


def test_home_click_navigates_to_computer_vision(monkeypatch):
    cv_label = "Computer Vision\nTracking and tactical metrics from broadcast video"
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": None,
            "session_state": {"active_vertical": "home"},
            "button": {cv_label: True},
        },
    )
    assert recorder.session_state.active_vertical == "vertical1"


def test_home_click_navigates_to_data_analytics(monkeypatch):
    cv_label = "Computer Vision\nTracking and tactical metrics from broadcast video"
    da_label = "Data Analytics\nTactical insights and proprietary metrics from event data and reports"
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": None,
            "session_state": {"active_vertical": "home"},
            "button": {cv_label: False, da_label: True},
        },
    )
    assert recorder.session_state.active_vertical == "vertical2"


def test_home_branding_renders_inline_tip_and_highlighted_initials(monkeypatch):
    recorder = run_app(monkeypatch, {"uploaded_video": None, "session_state": {"active_vertical": "home"}})
    assert any(
        '<h1 class="tip-title"><span class="tip-accent">T</span>actical <span class="tip-accent">I</span>ntelligence <span class="tip-accent">P</span>latform <span class="tip-inline-tip">(<span class="tip-accent">TIP</span>)</span></h1>'
        in item
        for item in recorder.markdowns
    )


def test_vertical2_back_to_home_sets_route(monkeypatch):
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": None,
            "session_state": {"active_vertical": "vertical2"},
            "button": {"Volver a Home": True},
        },
    )
    assert recorder.session_state.active_vertical == "home"


def test_vertical2_pdf_upload_renders_normalized_schema_preview(monkeypatch):
    uploaded = FakeUploadedFile("wyscout_report.pdf", b"dummy bytes")
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": uploaded,
            "session_state": {"active_vertical": "vertical2"},
        },
    )
    assert any("Archivo:** wyscout_report.pdf" in item for item in recorder.markdowns)
    metric_labels = [label for label, _ in recorder.metrics]
    assert metric_labels == [
        "Control del Juego",
        "Velocidad de Ataque",
        "Impacto del Pressing",
        "Riesgo en Salida",
    ]
    assert any(level in item for item in recorder.markdowns for level in ["Nivel: Alto", "Nivel: Medio", "Nivel: Bajo"])
    assert any(
        interpretation in item
        for item in recorder.captions
        for interpretation in [
            "El equipo dominó territorialmente el partido.",
            "El equipo tuvo control parcial del territorio.",
            "El equipo tuvo poca presencia en campo rival.",
        ]
    )
    assert "Radar Táctico" in recorder.subheaders
    assert "Insights del Partido" in recorder.subheaders
    assert recorder.tabs_labels == ["Attack", "Defense", "Transitions"]
    assert recorder.plotly_calls == 4
    assert any("Preview del schema normalizado" in item for item in recorder.subheaders)
    assert any('"match_info"' in item for item in recorder.markdowns)
    assert any('"proprietary_metrics"' in item for item in recorder.markdowns)


def test_smoke_tabs_exist_when_video_loaded(monkeypatch):
    uploaded = FakeUploadedFile("demo.mp4", b"video")
    recorder = run_app(monkeypatch, {"uploaded_video": uploaded, "session_state": {}})
    assert recorder.tabs_labels == ["Video", "Estadísticas", "Gráficos", "Exportar", "Scouting", "Interpretación", "Posesión"]


def test_regression_video_loaded_not_processed_shows_safe_state(monkeypatch):
    uploaded = FakeUploadedFile("demo.mp4", b"video")
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": uploaded,
            "session_state": {"stats": None, "video_processed": False},
        },
    )
    assert any("Ejecuta el procesamiento para habilitar métricas y exportes." in msg for msg in recorder.info_messages)
    assert any("Pendiente de ejecución" in item for item in recorder.markdowns)


def test_regression_processed_with_stats_renders_core_views(monkeypatch):
    uploaded = FakeUploadedFile("demo.mp4", b"video")
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": uploaded,
            "session_state": {"stats": build_full_stats(), "video_processed": True},
        },
    )
    assert "Exportar datos" in recorder.subheaders
    assert "Scouting" in recorder.subheaders
    assert "Interpretación" in recorder.subheaders
    assert "Posesión de pelota" in recorder.subheaders
    assert recorder.plotly_calls >= 2
    assert recorder.dataframe_calls >= 2


def test_regression_passes_section_visible_when_available(monkeypatch):
    uploaded = FakeUploadedFile("demo.mp4", b"video")
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": uploaded,
            "session_state": {"stats": build_full_stats(), "video_processed": True},
        },
    )
    assert "Pases y Pérdidas" in recorder.subheaders


def test_regression_passes_section_hidden_when_empty(monkeypatch):
    uploaded = FakeUploadedFile("demo.mp4", b"video")
    stats = build_full_stats()
    stats["possession"]["passes"] = {"total": 0, "team1_passes": 0, "team2_passes": 0, "turnovers": 0}
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": uploaded,
            "session_state": {"stats": stats, "video_processed": True},
        },
    )
    assert "Pases y Pérdidas" not in recorder.subheaders


def test_regression_partial_stats_do_not_break_render(monkeypatch):
    uploaded = FakeUploadedFile("demo.mp4", b"video")
    partial_stats = {
        "duration_seconds": 4.0,
        "total_frames": 100,
        "metrics": {"team1": {}, "team2": {}},
        "timeline": {"team1": {"frame_number": [1, 2]}, "team2": {"frame_number": [1, 2]}},
    }
    recorder = run_app(
        monkeypatch,
        {
            "uploaded_video": uploaded,
            "session_state": {"stats": partial_stats, "video_processed": True},
        },
    )
    assert "Análisis táctico" in recorder.subheaders
    assert not any("Traceback" in item for item in recorder.markdowns)


def test_event_normalizer_returns_stable_schema_keys_on_fallback():
    from src.services.event_normalizer import normalize_event_data

    normalized = normalize_event_data(
        {
            "status": "warning",
            "file_name": "empty_report.pdf",
            "page_count": 0,
            "raw_text": "",
        }
    )
    assert set(normalized.keys()) >= {
        "match_info",
        "team_summary",
        "formations",
        "attack",
        "defense",
        "transitions",
        "build_up",
        "finishing",
    }
    assert normalized["status"] == "warning"


def test_proprietary_metrics_returns_four_scores_in_valid_range():
    from src.services.proprietary_metrics import calculate_proprietary_metrics

    normalized = {
        "attack": {"signals": {"final third": 3, "box entries": 2, "shots": 4}},
        "defense": {"signals": {"pressing": 2, "recoveries": 3}},
        "transitions": {"signals": {"direct attack": 1, "counter": 2, "regain": 3, "turnover": 1}},
        "build_up": {"signals": {"progression": 4, "possession": 2}},
        "finishing": {"signals": {"on target": 2}},
        "meta": {"sections_detected": {"attack": 2, "build_up": 2}},
    }
    raw = {"raw_text": "Possession: 62%"}
    metrics = calculate_proprietary_metrics(normalized, raw)
    assert set(metrics.keys()) == {
        "field_tilt_index",
        "directness_index",
        "pressing_efficiency",
        "risk_exposure_score",
    }
    for metric in metrics.values():
        assert 0 <= metric["score"] <= 100
        assert metric["label"]
        assert metric["description"]
        assert metric["category"] in {"Low", "Medium", "High"}


def test_proprietary_metric_presentation_returns_spanish_copy_and_color():
    from src.services.proprietary_metrics import get_metric_presentation

    presentation = get_metric_presentation("field_tilt_index", {"category": "High"})
    assert presentation["label"] == "Control del Juego"
    assert presentation["level"] == "Alto"
    assert presentation["color"] == "#22c55e"
    assert presentation["interpretation"] == "El equipo dominó territorialmente el partido."


def test_pitch_view_builder_returns_figure_and_metadata():
    from src.utils.ui.pitch_views import build_pitch_view_figure

    normalized = {
        "attack": {"signals": {"crosses": 2, "box entries": 2, "final third": 3, "shots": 1, "xg": 1}},
        "defense": {"signals": {"duels": 4, "pressing": 2, "recoveries": 1}},
        "transitions": {"signals": {"turnover": 2, "regain": 3, "counter": 1, "direct attack": 1, "transition": 2}},
    }
    view = build_pitch_view_figure("Transitions", normalized)
    assert view["title"] == "Transitions View"
    assert isinstance(view["subtitle"], str) and view["subtitle"]
    assert view["figure"] is not None
    assert isinstance(view["has_signal"], bool)


def test_insight_generator_returns_readable_insights():
    from src.services.insight_generator import generate_match_insights

    normalized = {
        "attack": {"signals": {"final third": 3, "crosses": 1, "box entries": 2, "shots": 2}},
        "defense": {"signals": {"recoveries": 2}},
        "transitions": {"signals": {"regain": 2, "turnover": 1}},
    }
    metrics = {
        "field_tilt_index": {"score": 78},
        "directness_index": {"score": 64},
        "pressing_efficiency": {"score": 71},
        "risk_exposure_score": {"score": 32},
    }
    insights = generate_match_insights(normalized, metrics)
    assert 3 <= len(insights) <= 5
    assert all(isinstance(item, str) and item for item in insights)


def test_component_apply_plotly_dark_theme_sets_expected_layout(monkeypatch):
    st_module, _ = make_streamlit_module({"session_state": {}})
    monkeypatch.setitem(sys.modules, "streamlit", st_module)
    if "src.utils.ui.theme" in sys.modules:
        del sys.modules["src.utils.ui.theme"]
    from src.utils.ui.theme import apply_plotly_dark_theme

    fig = go.Figure()
    apply_plotly_dark_theme(fig)
    assert fig.layout.template is not None
    assert fig.layout.paper_bgcolor == "#0f131a"
    assert fig.layout.plot_bgcolor == "#141b24"


def test_open_event_visualizations_render_visible_traces_and_keep_pitch_below():
    from src.services.open_event_visualizations import create_event_map
    from src.services.open_event_visualizations import create_progressive_actions_map

    canonical_events = [
        {
            "event_id": "1",
            "match_id": "m1",
            "team_id": "t1",
            "team_name": "Argentina",
            "player_id": "p1",
            "player_name": "Lionel Messi",
            "minute": 12,
            "second": 8,
            "event_type": "Pass",
            "x": 42.0,
            "y": 30.0,
            "end_x": 65.0,
            "end_y": 34.0,
            "outcome": "Complete",
            "progressive": True,
            "under_pressure": False,
            "xG": 0.0,
            "xA": 0.0,
        },
        {
            "event_id": "2",
            "match_id": "m1",
            "team_id": "t1",
            "team_name": "Argentina",
            "player_id": "p2",
            "player_name": "Julian Alvarez",
            "minute": 25,
            "second": 14,
            "event_type": "Shot",
            "x": 102.0,
            "y": 36.0,
            "end_x": None,
            "end_y": None,
            "outcome": "Goal",
            "progressive": False,
            "under_pressure": True,
            "xG": 0.34,
            "xA": 0.0,
        },
    ]

    event_map = create_event_map(canonical_events, selected_team="Argentina")
    assert len(event_map.data) >= 2
    assert all(getattr(shape, "layer", None) == "below" for shape in event_map.layout.shapes)

    progressive_map = create_progressive_actions_map(canonical_events, selected_team="Argentina")
    assert len(progressive_map.data) >= 1
    assert progressive_map.data[0].mode == "lines+markers"


def test_api_event_dashboard_hides_technical_information_by_default(monkeypatch):
    st_module, recorder = make_streamlit_module({"session_state": {}})
    monkeypatch.setitem(sys.modules, "streamlit", st_module)
    if "src.verticals.vertical2_api_event" in sys.modules:
        del sys.modules["src.verticals.vertical2_api_event"]
    from src.verticals.vertical2_api_event import _render_common_event_dashboard

    result = {
        "competition_name": "UEFA Euro",
        "match_label": "Argentina vs Francia",
        "raw_payload": [{"id": "raw-1"}],
        "canonical_events": [
            {
                "event_id": "1",
                "match_id": "m1",
                "team_id": "t1",
                "team_name": "Argentina",
                "player_id": "p1",
                "player_name": "Lionel Messi",
                "minute": 10,
                "second": 5,
                "event_type": "Pass",
                "x": 42.0,
                "y": 30.0,
                "end_x": 61.0,
                "end_y": 34.0,
                "outcome": "Complete",
                "progressive": True,
                "under_pressure": False,
                "xG": 0.0,
                "xA": 0.0,
            }
        ],
    }

    _render_common_event_dashboard(result, selected_team="Todos", selected_player="Todos", show_technical_info=False)

    assert "Información técnica" not in recorder.markdowns
    assert not any("Canonical Event Model" in item for item in recorder.markdowns)
    assert all(key is not None for key in recorder.plotly_keys)
    assert len(recorder.plotly_keys) == len(set(recorder.plotly_keys))


def test_api_event_dashboard_shows_ai_coach_warning_when_key_is_missing(monkeypatch):
    module, recorder = load_vertical2_api_event(monkeypatch, {"session_state": {}})

    monkeypatch.setattr(
        sys.modules["src.utils.ui.ai_coach_panel"],
        "get_ai_coach_config_status",
        lambda: {
            "configured": False,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "Falta configurar AI_COACH_API_KEY en el entorno.",
        },
    )

    result = {
        "provider": module.STORAGE_PROVIDER_STATSBOMB,
        "match_id": "m1",
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
        "match_label": "Argentina vs Francia",
        "raw_payload": [{"id": "raw-1"}],
        "canonical_events": [
            {
                "event_id": "1",
                "match_id": "m1",
                "team_id": "t1",
                "team_name": "Argentina",
                "player_id": "p1",
                "player_name": "Lionel Messi",
                "minute": 10,
                "second": 5,
                "event_type": "Pass",
                "x": 42.0,
                "y": 30.0,
                "end_x": 61.0,
                "end_y": 34.0,
                "outcome": "Complete",
                "progressive": True,
                "under_pressure": False,
                "xG": 0.0,
                "xA": 0.0,
            }
        ],
    }

    module._render_common_event_dashboard(result, selected_team="Todos", selected_player="Todos", show_technical_info=False)

    assert any("AI Tactical Coach" in item for item in recorder.markdowns)
    assert any("Falta configurar AI_COACH_API_KEY para activar el AI Tactical Coach." in item for item in recorder.warning_messages)
    assert all("AI Coach configurado." not in item for item in recorder.captions)


def test_api_event_dashboard_can_generate_ai_coach_diagnosis_and_debug_context(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {},
            "button": {"Generar diagnóstico táctico": True},
        },
    )
    import src.utils.ui.ai_coach_panel as ai_coach_panel

    monkeypatch.setattr(
        ai_coach_panel,
        "get_ai_coach_config_status",
        lambda: {
            "configured": True,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "AI Tactical Coach configurado correctamente.",
        },
    )
    monkeypatch.setattr(
        ai_coach_panel,
        "generate_tactical_diagnosis",
        lambda match_context: {
            "ok": True,
            "diagnosis": "Diagnóstico táctico de prueba.",
            "error": "",
        },
    )
    monkeypatch.setattr(
        ai_coach_panel,
        "answer_coach_question",
        lambda match_context, user_question, conversation_history=None: {
            "ok": True,
            "answer": "Respuesta de prueba.",
            "error": "",
        },
    )

    result = {
        "provider": module.STORAGE_PROVIDER_STATSBOMB,
        "match_id": "m1",
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
        "match_label": "Argentina vs Francia",
        "raw_payload": [{"id": "raw-1"}],
        "canonical_events": [
            {
                "event_id": "1",
                "match_id": "m1",
                "team_id": "t1",
                "team_name": "Argentina",
                "player_id": "p1",
                "player_name": "Lionel Messi",
                "minute": 10,
                "second": 5,
                "event_type": "Pass",
                "x": 42.0,
                "y": 30.0,
                "end_x": 61.0,
                "end_y": 34.0,
                "outcome": "Complete",
                "progressive": True,
                "under_pressure": False,
                "xG": 0.1,
                "xA": 0.0,
            }
        ],
    }

    module._render_common_event_dashboard(result, selected_team="Todos", selected_player="Todos", show_technical_info=True)

    assert any("Diagnóstico táctico de prueba." in item for item in recorder.markdowns)
    assert "Match Context enviado al AI Coach" in recorder.expander_labels
    assert recorder.json_payloads


def test_ai_coach_chat_uses_safe_input_clear_pattern(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {},
            "button": {"Preguntar": True},
            "text_input": {"Preguntale algo al AI Coach...": "¿Cómo estuvo el equipo?"},
        },
    )
    import src.utils.ui.ai_coach_panel as ai_coach_panel

    monkeypatch.setattr(
        ai_coach_panel,
        "get_ai_coach_config_status",
        lambda: {
            "configured": True,
            "api_key_configured": True,
            "model_configured": True,
            "base_url_configured": True,
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
            "message": "AI Tactical Coach configurado correctamente.",
        },
    )
    monkeypatch.setattr(
        ai_coach_panel,
        "answer_coach_question",
        lambda match_context, user_question, conversation_history=None: {
            "ok": True,
            "answer": "Respuesta contextual de prueba.",
            "error": "",
        },
    )

    result = {
        "provider": module.STORAGE_PROVIDER_STATSBOMB,
        "match_id": "m1",
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
        "match_label": "Argentina vs Francia",
        "raw_payload": [{"id": "raw-1"}],
        "canonical_events": [
            {
                "event_id": "1",
                "match_id": "m1",
                "team_id": "t1",
                "team_name": "Argentina",
                "player_id": "p1",
                "player_name": "Lionel Messi",
                "minute": 10,
                "second": 5,
                "event_type": "Pass",
                "x": 42.0,
                "y": 30.0,
                "end_x": 61.0,
                "end_y": 34.0,
                "outcome": "Complete",
                "progressive": True,
                "under_pressure": False,
                "xG": 0.1,
                "xA": 0.0,
            }
        ],
    }

    module._render_common_event_dashboard(result, selected_team="Todos", selected_player="Todos", show_technical_info=False)

    chat_key = ai_coach_panel.build_ai_coach_state_key(
        "ai_coach_chat",
        provider=module.STORAGE_PROVIDER_STATSBOMB,
        match_id="m1",
        selected_team="Todos",
        selected_player="Todos",
    )
    clear_key = ai_coach_panel.build_ai_coach_state_key(
        "ai_coach_clear_input",
        provider=module.STORAGE_PROVIDER_STATSBOMB,
        match_id="m1",
        selected_team="Todos",
        selected_player="Todos",
    )

    assert recorder.session_state[chat_key][-2:] == [
        {"role": "user", "content": "¿Cómo estuvo el equipo?"},
        {"role": "assistant", "content": "Respuesta contextual de prueba."},
    ]
    assert recorder.session_state[clear_key] is True


def test_api_event_config_debug_status_hides_secret_values(monkeypatch):
    module, recorder = load_vertical2_api_event(monkeypatch, {"session_state": {}})

    monkeypatch.setattr(
        module,
        "get_api_football_config_status",
        lambda: {
            "configured": True,
            "message": "API-Football configurado correctamente.",
        },
    )
    monkeypatch.setattr(
        module,
        "get_ai_coach_config_status",
        lambda: {
            "configured": False,
            "api_key_configured": False,
            "model_configured": True,
            "base_url_configured": True,
            "model": "gpt-4o-mini",
            "base_url": "https://example.com/v1/chat/completions",
            "message": "Falta configurar AI_COACH_API_KEY en el entorno.",
        },
    )

    module._render_environment_config_status(show_technical_info=True)

    assert not any("Estado seguro de configuración" in item for item in recorder.captions)
    assert "Estado técnico de variables de entorno" in recorder.expander_labels
    assert recorder.json_payloads
    serialized = str(recorder.json_payloads[-1])
    assert "demo-key" not in serialized
    assert "secret" not in serialized


def test_api_event_config_status_is_hidden_from_main_view(monkeypatch):
    module, recorder = load_vertical2_api_event(monkeypatch, {"session_state": {}})

    monkeypatch.setattr(
        module,
        "get_api_football_config_status",
        lambda: {
            "configured": True,
            "message": "API-Football configurado correctamente.",
        },
    )
    monkeypatch.setattr(
        module,
        "get_ai_coach_config_status",
        lambda: {
            "configured": True,
            "api_key_configured": True,
            "model_configured": True,
            "base_url_configured": True,
            "model": "gpt-4o-mini",
            "base_url": "https://example.com/v1/chat/completions",
            "message": "AI Tactical Coach configurado correctamente.",
        },
    )

    module._render_environment_config_status(show_technical_info=False)

    assert not recorder.captions
    assert "Estado técnico de variables de entorno" not in recorder.expander_labels


def test_api_football_provider_sections_show_technical_tables_only_in_debug(monkeypatch):
    st_module, recorder = make_streamlit_module({"session_state": {}})
    monkeypatch.setitem(sys.modules, "streamlit", st_module)
    if "src.verticals.vertical2_api_event" in sys.modules:
        del sys.modules["src.verticals.vertical2_api_event"]
    from src.verticals.vertical2_api_event import _render_api_football_provider_sections

    result = {
        "raw_payload": {
            "events": [
                {
                    "time": {"elapsed": 12},
                    "team": {"name": "Argentina"},
                    "player": {"name": "Lionel Messi"},
                    "type": "Goal",
                    "detail": "Normal Goal",
                    "comments": "",
                }
            ],
            "lineups": [
                {
                    "team": {"name": "Argentina"},
                    "formation": "4-3-3",
                    "startXI": [{"player": {"name": "Lionel Messi"}}],
                    "substitutes": [{"player": {"name": "Julian Alvarez"}}],
                }
            ],
            "statistics": [
                {
                    "team": {"name": "Argentina"},
                    "statistics": [{"type": "Shots on Goal", "value": 5}],
                }
            ],
            "players": [
                {
                    "team": {"name": "Argentina"},
                    "players": [{"player": {"name": "Lionel Messi", "age": 36, "pos": "F", "number": 10}}],
                }
            ],
        },
        "canonical_events": [],
    }

    _render_api_football_provider_sections(result, show_technical_info=False)
    assert recorder.dataframe_calls == 0

    recorder.dataframe_calls = 0
    _render_api_football_provider_sections(result, show_technical_info=True)
    assert recorder.dataframe_calls >= 1


def test_statsbomb_provider_loads_dashboard_from_local_history(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {},
            "button": {"Cargar desde historial local": True},
        },
    )

    competition = {
        "competition_id": 55,
        "season_id": 43,
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "display_name": "UEFA Euro - 2020",
    }
    match = {
        "match_id": "3775648",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
        "display_name": "Argentina vs Francia — 2022-12-18",
    }
    canonical_events = [
        {
            "event_id": "1",
            "match_id": "3775648",
            "team_id": "779",
            "team_name": "Argentina",
            "player_id": "p1",
            "player_name": "Lionel Messi",
            "minute": 10,
            "second": 5,
            "event_type": "Pass",
            "x": 42.0,
            "y": 30.0,
            "end_x": 61.0,
            "end_y": 34.0,
            "outcome": "Complete",
            "progressive": True,
            "under_pressure": False,
            "xG": 0.0,
            "xA": 0.0,
        }
    ]

    monkeypatch.setattr(module, "_cached_competitions", lambda: [competition])
    monkeypatch.setattr(module, "_cached_matches", lambda competition_id, season_id: [match])
    monkeypatch.setattr(module, "has_processed_match", lambda provider, match_id: True)
    monkeypatch.setattr(
        module,
        "load_processed_match_payloads",
        lambda provider, match_id: {
            "raw_events": [{"id": "raw-1"}],
            "canonical_events": canonical_events,
            "metrics": {"total_events": 1},
        },
    )
    monkeypatch.setattr(module, "get_processed_matches", lambda limit=20: [])

    module._render_statsbomb_provider(show_technical_info=False)

    stored = recorder.session_state[module.SESSION_RESULT_KEY]
    assert stored["provider"] == module.STORAGE_PROVIDER_STATSBOMB
    assert stored["loaded_from_local"] is True
    assert stored["match_id"] == "3775648"
    assert ("Eventos analizados", "1") in recorder.metrics


def test_statsbomb_provider_warns_on_persistence_failure_but_keeps_dashboard(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {},
            "button": {"Cargar datos": True},
        },
    )

    competition = {
        "competition_id": 55,
        "season_id": 43,
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "display_name": "UEFA Euro - 2020",
    }
    match = {
        "match_id": "3775648",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
        "display_name": "Argentina vs Francia — 2022-12-18",
    }
    raw_events = [
        {
            "id": "evt-1",
            "team": {"id": 779, "name": "Argentina"},
            "player": {"id": 10, "name": "Lionel Messi"},
            "minute": 10,
            "second": 5,
            "type": {"name": "Pass"},
            "location": [42.0, 30.0],
            "pass": {"end_location": [61.0, 34.0]},
        }
    ]

    monkeypatch.setattr(module, "_cached_competitions", lambda: [competition])
    monkeypatch.setattr(module, "_cached_matches", lambda competition_id, season_id: [match])
    monkeypatch.setattr(module, "_cached_events", lambda match_id: raw_events)
    monkeypatch.setattr(module, "has_processed_match", lambda provider, match_id: False)
    monkeypatch.setattr(module, "get_processed_matches", lambda limit=20: [])

    def _raise_on_save(*args, **kwargs):
        raise RuntimeError("sqlite unavailable")

    monkeypatch.setattr(module, "save_processed_match", _raise_on_save)

    module._render_statsbomb_provider(show_technical_info=False)

    assert any("No se pudo guardar el partido en historial local" in item for item in recorder.warning_messages)
    assert ("Eventos analizados", "1") in recorder.metrics
    assert recorder.plotly_calls >= 1


def test_api_football_provider_translates_plan_error_in_ui(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {},
            "button": {"Buscar partidos": True},
        },
    )

    country = {"name": "Argentina", "display_name": "Argentina"}
    league = {
        "league_id": 130,
        "league_name": "Copa Argentina",
        "display_name": "Copa Argentina (Argentina)",
        "seasons": [{"year": 2024}, {"year": 2023}],
        "current_season": 2024,
    }
    error_status = {
        "status": "error",
        "message": "API-Football devolvió errores en la respuesta.",
        "errors": ["plan: Free plans do not have access to this season, try from 2022 to 2024."],
    }

    monkeypatch.setattr(module, "get_api_football_api_key", lambda: "test-key")
    monkeypatch.setattr(module, "_cached_api_countries", lambda: [country])
    monkeypatch.setattr(module, "_cached_api_leagues", lambda country=None, season=None, search=None: [league])
    monkeypatch.setattr(module, "_cached_api_fixtures", lambda league_id, season, last=None: [])
    monkeypatch.setattr(module, "get_api_football_status", lambda: error_status)
    monkeypatch.setattr(module, "get_processed_matches", lambda limit=20: [])

    module._render_api_football_provider(show_technical_info=False)

    assert any("Tu plan actual no tiene acceso a la temporada seleccionada" in item for item in recorder.warning_messages)


def test_api_football_dashboard_without_coordinates_shows_message_and_skips_maps(monkeypatch):
    module, recorder = load_vertical2_api_event(monkeypatch, {"session_state": {}})

    result = {
        "provider": module.STORAGE_PROVIDER_API_FOOTBALL,
        "match_id": "12345",
        "competition_name": "Liga Profesional",
        "match_label": "River Plate vs Boca Juniors",
        "raw_payload": {"events": []},
        "canonical_events": [
            {
                "event_id": "12345-0-goal-15",
                "match_id": "12345",
                "team_id": "435",
                "team_name": "River Plate",
                "player_id": "unknown-player",
                "player_name": "Jugador desconocido",
                "minute": 15,
                "second": 0,
                "event_type": "Goal",
                "x": None,
                "y": None,
                "end_x": None,
                "end_y": None,
                "outcome": "Normal Goal",
                "progressive": False,
                "under_pressure": False,
                "xG": 0.0,
                "xA": 0.0,
            }
        ],
    }

    module._render_common_event_dashboard(
        result,
        selected_team="Todos",
        selected_player="Todos",
        show_technical_info=False,
    )

    assert any("Este provider no entrega coordenadas de eventos para este partido" in item for item in recorder.info_messages)
    assert recorder.plotly_calls == 0


def test_statsbomb_provider_ignores_stale_session_result_from_other_provider(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {
                "vertical2_api_event_result": {
                    "provider": "api_football",
                    "match_id": "old-fixture",
                    "canonical_events": [{"event_id": "stale"}],
                    "raw_payload": {"events": [{"id": "stale"}]},
                }
            }
        },
    )

    competition = {
        "competition_id": 55,
        "season_id": 43,
        "competition_name": "UEFA Euro",
        "season_name": "2020",
        "display_name": "UEFA Euro - 2020",
    }
    match = {
        "match_id": "3775648",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
        "display_name": "Argentina vs Francia — 2022-12-18",
    }

    monkeypatch.setattr(module, "_cached_competitions", lambda: [competition])
    monkeypatch.setattr(module, "_cached_matches", lambda competition_id, season_id: [match])
    monkeypatch.setattr(module, "has_processed_match", lambda provider, match_id: False)
    monkeypatch.setattr(module, "get_processed_matches", lambda limit=20: [])

    module._render_statsbomb_provider(show_technical_info=False)

    assert any("Configurá filtros y presioná 'Cargar datos' para ver métricas e insights." in item for item in recorder.info_messages)
    assert ("Eventos analizados", "1") not in recorder.metrics


def test_component_build_centroid_heatmap_handles_empty_and_valid():
    from src.utils.ui.heatmap_render import build_centroid_heatmap

    assert build_centroid_heatmap({}, "team1") is None
    telemetry = {"team1_centroid_x": [10, 20, None, 120], "team1_centroid_y": [15, 30, 22, 10]}
    heatmap = build_centroid_heatmap(telemetry, "team1")
    assert heatmap is not None
    assert isinstance(heatmap, np.ndarray)
    assert float(np.sum(heatmap)) == 2.0
