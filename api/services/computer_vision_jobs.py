from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status

from api.schemas import ComputerVisionConfig
from api.services.computer_vision_service import OUTPUT_ROOT
from api.services.computer_vision_service import analyze_video_from_source
from api.services.computer_vision_service import build_artifact_url
from api.services.computer_vision_service import prepare_job_source
from api.services.computer_vision_service import prepare_runtime_assets
from src.services.storage.computer_vision_repository import build_config_hash
from src.services.storage.computer_vision_repository import build_processing_id
from src.services.storage.computer_vision_repository import build_source_fingerprint
from src.services.storage.computer_vision_repository import delete_processed_video
from src.services.storage.computer_vision_repository import get_processed_videos
from src.services.storage.computer_vision_repository import load_processed_video_payloads
from src.services.storage.computer_vision_repository import save_processed_video


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create_job(
    *,
    source_mode: str,
    config: ComputerVisionConfig,
    upload_file: UploadFile | None = None,
    player_model_file: UploadFile | None = None,
    ball_model_file: UploadFile | None = None,
    soccernet_path: str | None = None,
) -> dict[str, Any]:
    try:
        source_path, video_name, cleanup_dir, source_details = prepare_job_source(
            source_mode=source_mode,
            upload_file=upload_file,
            soccernet_path=soccernet_path,
            config=config,
        )
        runtime_assets, cleanup_dir = prepare_runtime_assets(
            config=config,
            cleanup_dir=cleanup_dir,
            player_model_file=player_model_file,
            ball_model_file=ball_model_file,
        )
    except HTTPException:
        raise

    job_id = str(uuid.uuid4())
    now = _now_iso()
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "video_name": video_name,
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job_data

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job_id": job_id,
            "source_path": source_path,
            "video_name": video_name,
            "cleanup_dir": cleanup_dir,
            "config": config,
            "runtime_assets": runtime_assets,
            "source_details": source_details,
        },
        daemon=True,
    )
    thread.start()
    return job_data


def get_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job no encontrado: {job_id}",
            )
        return dict(job)


def _run_job(
    *,
    job_id: str,
    source_path: Path,
    video_name: str,
    cleanup_dir: Path | None,
    config: ComputerVisionConfig,
    runtime_assets: dict[str, Path | None],
    source_details: dict[str, Any],
) -> None:
    _update_job(job_id, status="running", error=None)
    config_payload = config.model_dump(mode="python")
    source_mode = str(source_details.get("source_mode") or "upload")
    source_label = _build_source_label(
        source_mode=source_mode,
        source_label=str(source_details.get("source_label") or video_name),
        config_payload=config_payload,
    )
    source_fingerprint = build_source_fingerprint(
        source_mode=source_mode,
        source_path=source_path,
        source_label=source_label,
        reference_path=source_details.get("reference_path"),
    )
    config_hash = build_config_hash(config_payload)
    processing_id = build_processing_id(source_fingerprint, config_hash)
    persisted_processing_id: str | None = None

    try:
        result = analyze_video_from_source(
            source_path=source_path,
            video_name=video_name,
            config=config,
            cleanup_dir=cleanup_dir,
            artifact_suffix=job_id,
            runtime_assets=runtime_assets,
        )
        if str(result.get("source") or "") == "api":
            save_status = save_processed_video(
                processing_id=processing_id,
                job_id=job_id,
                source_mode=source_mode,
                source_label=source_label,
                video_name=video_name,
                source_fingerprint=source_fingerprint,
                config_hash=config_hash,
                status="completed",
                result_payload=result,
                stats_path=_artifact_path_from_result(result, "stats_json_url"),
                video_path=_artifact_path_from_result(result, "video_url"),
            )
            if save_status.get("ok"):
                persisted_processing_id = processing_id
            else:
                _append_result_warning(
                    result,
                    str(save_status.get("message") or "No se pudo guardar el procesamiento en historial local."),
                )
        _update_job(
            job_id,
            status="completed",
            result=result,
            error=None,
            processing_id=persisted_processing_id,
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc))


def list_history(limit: int = 20) -> dict[str, Any]:
    rows = get_processed_videos(limit=limit)
    return {"items": [_history_item_from_row(row) for row in rows]}


def load_history_entry(processing_id: str) -> dict[str, Any]:
    payloads = load_processed_video_payloads(processing_id)
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró el procesamiento guardado {processing_id}.",
        )

    metadata = payloads.get("metadata", {}) or {}
    result = payloads.get("result", {}) or {}
    _refresh_result_artifacts(result, metadata)
    return {
        "metadata": _history_item_from_row(metadata),
        "result": result,
    }


def delete_history_entry(processing_id: str) -> dict[str, Any]:
    return delete_processed_video(processing_id=processing_id)


def _update_job(job_id: str, **patch: Any) -> None:
    with _JOBS_LOCK:
        current = _JOBS.get(job_id)
        if not current:
            return
        current.update(patch)
        current["updated_at"] = _now_iso()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_item_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "processing_id": str(row.get("processing_id") or ""),
        "job_id": str(row.get("job_id") or ""),
        "source_mode": str(row.get("source_mode") or ""),
        "source_label": str(row.get("source_label") or ""),
        "video_name": str(row.get("video_name") or ""),
        "status": str(row.get("status") or ""),
        "created_at": str(row.get("created_at") or ""),
        "updated_at": str(row.get("updated_at") or ""),
        "video_url": build_artifact_url(row.get("video_path")),
        "stats_json_url": build_artifact_url(row.get("stats_path")),
    }


def _artifact_path_from_result(result: dict[str, Any], artifact_key: str) -> Path | None:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    artifact_url = str(artifacts.get(artifact_key) or "").strip()
    if not artifact_url:
        return None

    artifact_name = artifact_url.rstrip("/").split("/")[-1].strip()
    if not artifact_name:
        return None
    return OUTPUT_ROOT / artifact_name


def _append_result_warning(result: dict[str, Any], message: str) -> None:
    warnings = list(result.get("warnings") or [])
    warnings.append(message)
    result["warnings"] = warnings


def _build_source_label(*, source_mode: str, source_label: str, config_payload: dict[str, Any]) -> str:
    base_label = source_label.strip() or "video"
    if not bool(config_payload.get("segment_mode")):
        return base_label

    start_seconds = int(config_payload.get("start_seconds") or 0)
    duration_seconds = int(config_payload.get("duration_seconds") or 0)
    segment_label = f"segmento {start_seconds}s-{start_seconds + duration_seconds}s"
    if source_mode == "soccernet":
        return f"{base_label} ({segment_label})"
    return f"{base_label} ({segment_label})"


def _refresh_result_artifacts(result: dict[str, Any], metadata: dict[str, Any]) -> None:
    artifacts = dict(result.get("artifacts") or {})
    artifacts["video_url"] = build_artifact_url(metadata.get("video_path"))
    artifacts["stats_json_url"] = build_artifact_url(metadata.get("stats_path"))
    artifacts["pdf_url"] = artifacts.get("pdf_url")
    result["artifacts"] = artifacts
