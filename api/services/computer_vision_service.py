from __future__ import annotations

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status

from api.schemas import ComputerVisionConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api"


def analyze_video(
    *,
    source_mode: str,
    config: ComputerVisionConfig,
    upload_file: UploadFile | None = None,
    soccernet_path: str | None = None,
) -> dict[str, Any]:
    try:
        source_path, video_name, cleanup_dir = prepare_job_source(
            source_mode=source_mode,
            upload_file=upload_file,
            soccernet_path=soccernet_path,
            config=config,
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _build_mock_result(video_name="clip-demo.mp4", config=config, message=str(exc))

    return analyze_video_from_source(
        source_path=source_path,
        video_name=video_name,
        config=config,
        cleanup_dir=cleanup_dir,
    )


def analyze_video_from_source(
    *,
    source_path: Path,
    video_name: str,
    config: ComputerVisionConfig,
    cleanup_dir: Path | None,
) -> dict[str, Any]:
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        target_path = OUTPUT_ROOT / f"{Path(source_path).stem}_processed.mp4"
        from src.controllers.process_video import process_video

        player_model = _load_player_model(config.model_name)
        pitch_model = _load_pitch_model() if config.enable_radar else None
        full_field_approx = bool(config.full_field_approx or (config.enable_radar and pitch_model is None))

        process_video(
            source_path=str(source_path),
            target_path=str(target_path),
            player_model=player_model,
            ball_model=None,
            pitch_model=pitch_model,
            conf=float(config.confidence),
            detection_mode="players_and_ball",
            img_size=int(config.image_size),
            full_field_approx=full_field_approx,
            enable_possession=bool(config.enable_possession),
            disable_inertia=bool(config.disable_inertia),
            export_profile="full",
            sample_stride=int(config.sample_stride),
            topk_frames=int(config.topk_frames),
            enable_compression=bool(config.enable_compression),
        )
        stats_path = target_path.parent / f"{target_path.stem}_stats.json"
        if not stats_path.exists():
            raise RuntimeError("El pipeline terminó sin generar el JSON de estadísticas.")

        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
        return _map_stats_to_response(stats_payload, video_name)
    except Exception as exc:
        return _build_mock_result(video_name=video_name, config=config, message=str(exc))
    finally:
        if cleanup_dir and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def prepare_job_source(
    *,
    source_mode: str,
    upload_file: UploadFile | None,
    soccernet_path: str | None,
    config: ComputerVisionConfig,
) -> tuple[Path, str, Path | None]:
    if source_mode == "upload":
        if upload_file is None or not upload_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debés subir un archivo de video.",
            )
        temp_dir = Path(tempfile.mkdtemp(prefix="cv-upload-"))
        source_path = temp_dir / upload_file.filename
        source_path.write_bytes(upload_file.file.read())
        if config.segment_mode:
            from src.controllers.clip_video_simple import clip_video_simple

            clipped_path = temp_dir / f"segment_{upload_file.filename}"
            clip_video_simple(
                str(source_path),
                str(clipped_path),
                float(config.start_seconds),
                float(config.duration_seconds),
            )
            source_path = clipped_path
        return source_path, upload_file.filename, temp_dir

    if source_mode == "soccernet":
        if not soccernet_path or len(soccernet_path.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debés enviar `soccernet_path`.",
            )
        resolved = Path(soccernet_path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el video: {resolved}",
            )
        if config.segment_mode:
            from src.controllers.clip_video_simple import clip_video_simple

            temp_dir = Path(tempfile.mkdtemp(prefix="cv-segment-"))
            clipped_path = temp_dir / resolved.name
            clip_video_simple(
                str(resolved),
                str(clipped_path),
                float(config.start_seconds),
                float(config.duration_seconds),
            )
            return clipped_path, resolved.name, temp_dir
        return resolved, resolved.name, None

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"source_mode no soportado: {source_mode}",
    )


@lru_cache(maxsize=4)
def _load_player_model(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


@lru_cache(maxsize=2)
def _load_pitch_model():
    from ultralytics import YOLO

    homography_path = MODEL_ROOT / "homography.pt"
    if homography_path.exists():
        return YOLO(str(homography_path))
    return None


def _map_stats_to_response(stats_payload: dict[str, Any], video_name: str) -> dict[str, Any]:
    health_summary = stats_payload.get("health_summary", {}) or {}
    quality_control = stats_payload.get("quality_control", {}) or {}
    timeline = stats_payload.get("timeline", {}) or {}
    metrics = stats_payload.get("metrics", {}) or {}
    possession = stats_payload.get("possession") or None

    return {
        "source": "api",
        "status_message": None,
        "video_name": video_name,
        "duration_seconds": float(stats_payload.get("duration_seconds", 0.0) or 0.0),
        "total_frames": int(stats_payload.get("total_frames", 0) or 0),
        "fps": float(stats_payload.get("fps", 0.0) or 0.0),
        "health_summary": health_summary,
        "formations": {
            "team1": {"most_common": _safe_nested(stats_payload, "formations", "team1", "most_common", default="Unknown")},
            "team2": {"most_common": _safe_nested(stats_payload, "formations", "team2", "most_common", default="Unknown")},
        },
        "metrics": metrics,
        "timeline": {
            "pressure_height": _metric_timeline(timeline, "pressure_height"),
            "compactness": _metric_timeline(timeline, "compactness"),
            "offensive_width": _metric_timeline(timeline, "offensive_width"),
        },
        "possession": _map_possession(possession),
        "scouting": _map_scouting(metrics, quality_control),
        "exports": {"json": True, "csv": True, "pdf": False},
        "warnings": list(quality_control.get("warnings") or []),
        "interpretation": _build_interpretation(health_summary),
    }


def _metric_timeline(timeline: dict[str, Any], metric_key: str) -> dict[str, Any]:
    team1 = timeline.get("team1", {}) or {}
    team2 = timeline.get("team2", {}) or {}
    frames = list(team1.get("frame_number") or team2.get("frame_number") or [])

    return {
        "frames": frames,
        "team1": list(team1.get(metric_key) or []),
        "team2": list(team2.get(metric_key) or []),
    }


def _map_possession(possession: dict[str, Any] | None) -> dict[str, Any] | None:
    if not possession:
        return None

    return {
        "team1_pct": round(float(possession.get("team1_possession_pct", 0.0) or 0.0)),
        "team2_pct": round(float(possession.get("team2_possession_pct", 0.0) or 0.0)),
        "contested_pct": round(
            (
                float(possession.get("contested_frames", 0.0) or 0.0)
                / max(1.0, float(possession.get("total_frames_analyzed", possession.get("total_possession_frames", 0.0)) or 1.0))
            )
            * 100.0
        ),
        "unknown_pct": None if possession.get("demo_ready", True) else 100,
    }


def _map_scouting(metrics: dict[str, Any], quality_control: dict[str, Any]) -> dict[str, Any]:
    team1_metrics = metrics.get("team1", {}) or {}
    team2_metrics = metrics.get("team2", {}) or {}

    return {
        "confidence": {
            "team1": {
                "label": str(quality_control.get("confidence_grade_team1") or "Media"),
                "score": _confidence_score(str(quality_control.get("confidence_grade_team1") or "Media")),
            },
            "team2": {
                "label": str(quality_control.get("confidence_grade_team2") or "Media"),
                "score": _confidence_score(str(quality_control.get("confidence_grade_team2") or "Media")),
            },
        },
        "bullets": {
            "team1": _build_team_bullets(team1_metrics),
            "team2": _build_team_bullets(team2_metrics),
        },
    }


def _build_team_bullets(team_metrics: dict[str, Any]) -> list[str]:
    pressure = _metric_mean(team_metrics, "pressure_height")
    width = _metric_mean(team_metrics, "offensive_width")
    compactness = _metric_mean(team_metrics, "compactness")

    bullets: list[str] = []
    if pressure is not None:
        if pressure >= 40:
            bullets.append("Bloque con tendencia a presionar en campo rival.")
        else:
            bullets.append("Bloque de presión más contenido y menos adelantado.")
    if width is not None:
        if width >= 34:
            bullets.append("Mantiene buena amplitud ofensiva para atacar por fuera.")
        else:
            bullets.append("Ataca con amplitud limitada y más juego interior.")
    if compactness is not None:
        if compactness <= 850:
            bullets.append("Muestra una estructura bastante compacta entre líneas.")
        else:
            bullets.append("La estructura es más extendida y con mayor superficie ocupada.")

    return bullets[:3] or ["No hay suficiente señal para resumir este equipo."]


def _build_interpretation(health_summary: dict[str, Any]) -> list[str]:
    insights = [
        "La lectura táctica debe empezar por homografía y tracking antes de usar métricas avanzadas.",
        "Los gráficos temporales ayudan a detectar cambios de estructura mejor que un promedio aislado.",
    ]
    if float(health_summary.get("fallback_ratio", 0.0) or 0.0) > 0.2:
        insights.append("El clip tuvo una proporción alta de fallback; tomá las conclusiones con cautela.")
    else:
        insights.append("La estabilidad general del pipeline permite una lectura táctica razonable del recorte.")
    return insights[:3]


def _metric_mean(team_metrics: dict[str, Any], key: str) -> float | None:
    metric = team_metrics.get(key)
    if isinstance(metric, dict) and isinstance(metric.get("mean"), (int, float)):
        return float(metric.get("mean"))
    return None


def _safe_nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _confidence_score(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "alta":
        return 82
    if normalized == "baja":
        return 48
    return 68


def _build_mock_result(video_name: str, config: ComputerVisionConfig, message: str) -> dict[str, Any]:
    duration = float(config.duration_seconds if config.segment_mode else 18.4)
    total_frames = int(round(duration * 25))
    return {
        "source": "mock",
        "status_message": f"No se pudo completar el pipeline real de video: {message}",
        "video_name": video_name,
        "duration_seconds": duration,
        "total_frames": total_frames,
        "fps": 25.0,
        "health_summary": {
            "demo_mode": "degraded",
            "fallback_ratio": 0.18,
            "p95_reproj_error_m": 1.4,
            "p95_churn_ratio": 0.32,
            "p95_max_speed_mps": 8.4,
            "ball_detected_ratio": 0.42,
        },
        "formations": {
            "team1": {"most_common": "4-3-3"},
            "team2": {"most_common": "4-4-2"},
        },
        "metrics": {
            "team1": {
                "pressure_height": {"mean": 42.1, "min": 35.2, "max": 51.8},
                "offensive_width": {"mean": 34.6, "min": 28.4, "max": 42.2},
                "compactness": {"mean": 812.0, "min": 744.0, "max": 930.0},
                "block_depth_m": {"mean": 37.4, "min": 30.1, "max": 44.8},
                "block_width_m": {"mean": 33.8, "min": 27.3, "max": 40.4},
                "def_line_left_m": {"mean": 63.2, "min": 54.7, "max": 70.4},
                "def_line_right_m": {"mean": 65.7, "min": 58.9, "max": 72.3},
                "valid_frames": 332,
            },
            "team2": {
                "pressure_height": {"mean": 35.8, "min": 29.4, "max": 44.1},
                "offensive_width": {"mean": 29.1, "min": 22.5, "max": 36.8},
                "compactness": {"mean": 748.0, "min": 681.0, "max": 840.0},
                "block_depth_m": {"mean": 31.8, "min": 27.6, "max": 38.9},
                "block_width_m": {"mean": 29.4, "min": 24.3, "max": 35.8},
                "def_line_left_m": {"mean": 54.8, "min": 49.2, "max": 61.7},
                "def_line_right_m": {"mean": 57.4, "min": 51.3, "max": 62.8},
                "valid_frames": 320,
            },
        },
        "timeline": {
            "pressure_height": {
                "frames": [0, 60, 120, 180, 240, 300, 360, 420],
                "team1": [38, 40, 43, 45, 41, 44, 46, 42],
                "team2": [30, 31, 34, 36, 35, 37, 39, 36],
            },
            "compactness": {
                "frames": [0, 60, 120, 180, 240, 300, 360, 420],
                "team1": [790, 810, 820, 835, 800, 815, 830, 808],
                "team2": [710, 725, 742, 760, 748, 771, 755, 740],
            },
            "offensive_width": {
                "frames": [0, 60, 120, 180, 240, 300, 360, 420],
                "team1": [31, 33, 35, 36, 34, 35, 37, 34],
                "team2": [26, 27, 29, 31, 30, 31, 32, 29],
            },
        },
        "possession": {
            "team1_pct": 54,
            "team2_pct": 39,
            "contested_pct": 7,
            "unknown_pct": None,
        },
        "scouting": {
            "confidence": {
                "team1": {"label": "Alta", "score": 84},
                "team2": {"label": "Media", "score": 68},
            },
            "bullets": {
                "team1": [
                    "Bloque medio-alto con tendencia a recuperar adelantado.",
                    "Buena amplitud ofensiva para atacar por fuera.",
                    "La línea defensiva se sostiene por encima del bloque medio.",
                ],
                "team2": [
                    "Bloque medio con menor altura de presión.",
                    "Compactación razonable, pero menos agresiva tras pérdida.",
                    "Fase ofensiva más contenida y menos ancha.",
                ],
            },
        },
        "exports": {"json": True, "csv": True, "pdf": False},
        "warnings": [
            "Demo degradado: las métricas son aproximadas si la homografía no es estable.",
            "La posesión depende de señal de balón consistente.",
        ],
        "interpretation": [
            "La lectura táctica debe empezar por homografía y tracking antes de interpretar métricas avanzadas.",
            "Si el clip presenta zoom extremo o pocas líneas de campo, la confiabilidad baja.",
            "Usá los gráficos temporales para detectar cambios de estructura y no sólo promedios.",
        ],
    }
