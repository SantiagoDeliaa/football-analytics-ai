from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CompetitionResponse(BaseModel):
    competition_id: int | None = None
    season_id: int | None = None
    competition_name: str
    season_name: str
    country_name: str | None = None
    display_name: str


class MatchResponse(BaseModel):
    match_id: int | None = None
    home_team: str
    away_team: str
    match_date: str
    competition: str | None = None
    season: str | None = None
    display_name: str


class EventDataAnalyzeRequest(BaseModel):
    provider: str
    match_id: int | str
    team: str | None = None
    player: str | None = None
    competition_name: str | None = None
    season_name: str | None = None
    match_label: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    match_date: str | None = None


class EventDataResultResponse(BaseModel):
    match_id: str
    competition_name: str
    match_label: str
    canonical_events: list[dict[str, Any]]
    metrics: dict[str, Any]
    insights: list[str]
    raw_events_count: int
    used_fallback_events: bool
    events_status_message: str


class ProcessedMatchSummaryResponse(BaseModel):
    provider: str
    match_id: str
    competition_name: str
    season_name: str
    home_team: str
    away_team: str
    match_date: str
    created_at: str
    updated_at: str


class PdfAnalysisResponse(BaseModel):
    normalized_payload: dict[str, Any]
    metrics: dict[str, Any]
    insights: list[str]
    ingestion: dict[str, Any]


class ComputerVisionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: Literal["yolov8n.pt", "yolov8s.pt"] = "yolov8n.pt"
    confidence: float = Field(default=0.25, ge=0.1, le=0.95)
    image_size: Literal[640, 720, 960] = 640
    only_person: bool = True
    segment_mode: bool = False
    start_seconds: int = Field(default=0, ge=0)
    duration_seconds: int = Field(default=10, ge=1)
    full_field_approx: bool = False
    enable_radar: bool = True
    enable_analytics: bool = True
    enable_possession: bool = True
    disable_inertia: bool = False
    export_profile: Literal["summary", "debug_sampled", "full"] = "debug_sampled"
    sample_stride: int = Field(default=10, ge=1)
    topk_frames: int = Field(default=20, ge=1)
    enable_compression: bool = True


class ComputerVisionResultResponse(BaseModel):
    source: Literal["api", "mock"]
    status_message: str | None = None
    video_name: str
    duration_seconds: float
    total_frames: int
    fps: float
    health_summary: dict[str, Any]
    formations: dict[str, Any]
    metrics: dict[str, Any]
    timeline: dict[str, Any]
    possession: dict[str, Any] | None = None
    scouting: dict[str, Any]
    exports: dict[str, bool]
    warnings: list[str]
    interpretation: list[str]


class ComputerVisionJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    video_name: str
    result: dict[str, Any] | None = None
    error: str | None = None
