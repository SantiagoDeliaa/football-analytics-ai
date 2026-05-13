from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status

from api.schemas import ComputerVisionConfig
from api.services.computer_vision_service import analyze_video_from_source
from api.services.computer_vision_service import prepare_job_source


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def create_job(
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
) -> None:
    _update_job(job_id, status="running", error=None)
    try:
        result = analyze_video_from_source(
            source_path=source_path,
            video_name=video_name,
            config=config,
            cleanup_dir=cleanup_dir,
        )
        _update_job(job_id, status="completed", result=result, error=None)
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc))


def _update_job(job_id: str, **patch: Any) -> None:
    with _JOBS_LOCK:
        current = _JOBS.get(job_id)
        if not current:
            return
        current.update(patch)
        current["updated_at"] = _now_iso()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
