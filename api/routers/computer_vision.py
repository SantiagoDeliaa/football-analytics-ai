from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from api.schemas import ComputerVisionConfig
from api.schemas import ComputerVisionHistoryDetail
from api.schemas import ComputerVisionHistoryResponse
from api.schemas import ComputerVisionJobResponse
from api.schemas import ComputerVisionResultResponse
from api.schemas import DeleteComputerVisionHistoryResponse
from api.services.computer_vision_jobs import create_job
from api.services.computer_vision_jobs import delete_history_entry
from api.services.computer_vision_jobs import get_job
from api.services.computer_vision_jobs import list_history
from api.services.computer_vision_jobs import load_history_entry
from api.services.computer_vision_service import analyze_video


router = APIRouter(prefix="/api/v1/computer-vision", tags=["computer-vision"])


@router.post("/analyze", response_model=ComputerVisionResultResponse)
def post_analyze(
    source_mode: str = Form(...),
    config: str = Form(...),
    file: UploadFile | None = File(default=None),
    player_model_file: UploadFile | None = File(default=None),
    ball_model_file: UploadFile | None = File(default=None),
    soccernet_path: str | None = Form(default=None),
) -> dict:
    try:
        parsed_config = ComputerVisionConfig.model_validate(json.loads(config))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Config inválida: {exc}",
        ) from exc

    return analyze_video(
        source_mode=source_mode,
        config=parsed_config,
        upload_file=file,
        player_model_file=player_model_file,
        ball_model_file=ball_model_file,
        soccernet_path=soccernet_path,
    )


@router.post("/jobs", response_model=ComputerVisionJobResponse)
def post_job(
    source_mode: str = Form(...),
    config: str = Form(...),
    file: UploadFile | None = File(default=None),
    player_model_file: UploadFile | None = File(default=None),
    ball_model_file: UploadFile | None = File(default=None),
    soccernet_path: str | None = Form(default=None),
) -> dict:
    try:
        parsed_config = ComputerVisionConfig.model_validate(json.loads(config))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Config inválida: {exc}",
        ) from exc

    return create_job(
        source_mode=source_mode,
        config=parsed_config,
        upload_file=file,
        player_model_file=player_model_file,
        ball_model_file=ball_model_file,
        soccernet_path=soccernet_path,
    )


@router.get("/history", response_model=ComputerVisionHistoryResponse)
def get_history(limit: int = 20) -> dict:
    return list_history(limit=limit)


@router.get("/history/{processing_id}", response_model=ComputerVisionHistoryDetail)
def get_history_entry(processing_id: str) -> dict:
    return load_history_entry(processing_id)


@router.delete("/history/{processing_id}", response_model=DeleteComputerVisionHistoryResponse)
def delete_history(processing_id: str) -> dict:
    return delete_history_entry(processing_id)


@router.get("/jobs/{job_id}", response_model=ComputerVisionJobResponse)
def get_job_status(job_id: str) -> dict:
    return get_job(job_id)
