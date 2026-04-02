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
        return FakeContext()

    def text(self, value):
        self.markdowns.append(str(value))

    def code(self, value, **kwargs):
        self.markdowns.append(str(value))


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
        "src.verticals.home",
        "src.verticals.vertical1",
        "src.verticals.vertical2",
        "src.verticals.vertical1_legacy",
    ]:
        if name in sys.modules:
            del sys.modules[name]
    try:
        importlib.import_module("app")
    except StopExecution:
        pass
    return recorder


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
    assert "Vertical 2 — Data Analytics" in recorder.headers
    assert any("Sube un reporte PDF" in msg for msg in recorder.info_messages)
    assert "Subir reporte Wyscout (.pdf)" in recorder.file_uploaders
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
    assert any("Preview del schema normalizado" in item for item in recorder.subheaders)
    assert any('"match_info"' in item for item in recorder.markdowns)


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


def test_component_build_centroid_heatmap_handles_empty_and_valid():
    from src.utils.ui.heatmap_render import build_centroid_heatmap

    assert build_centroid_heatmap({}, "team1") is None
    telemetry = {"team1_centroid_x": [10, 20, None, 120], "team1_centroid_y": [15, 30, 22, 10]}
    heatmap = build_centroid_heatmap(telemetry, "team1")
    assert heatmap is not None
    assert isinstance(heatmap, np.ndarray)
    assert float(np.sum(heatmap)) == 2.0
