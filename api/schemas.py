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


class ApiFootballCountryResponse(BaseModel):
    name: str
    code: str = ""
    flag: str = ""
    display_name: str


class ApiFootballLeagueResponse(BaseModel):
    league_id: int | None = None
    league_name: str
    country_name: str
    type: str = ""
    logo: str = ""
    seasons: list[int] = Field(default_factory=list)
    current_season: int | None = None
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


class CoachConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EventDataResultResponse(BaseModel):
    provider: str
    match_id: str
    competition_name: str
    season_name: str = ""
    match_label: str
    home_team: str = ""
    away_team: str = ""
    match_date: str = ""
    canonical_events: list[dict[str, Any]]
    metrics: dict[str, Any]
    insights: list[str]
    raw_events_count: int
    raw_payload: Any | None = None
    used_fallback_events: bool
    events_status_message: str


class EventDataCoachRequest(EventDataAnalyzeRequest):
    pass


class EventDataCoachQuestionRequest(EventDataCoachRequest):
    question: str
    conversation_history: list[CoachConversationMessage] = Field(default_factory=list)


class CoachDiagnosisResponse(BaseModel):
    ok: bool
    diagnosis: str
    error: str
    suggested_questions: list[str] = Field(default_factory=list)


class CoachAnswerResponse(BaseModel):
    ok: bool
    answer: str
    error: str
    suggested_questions: list[str] = Field(default_factory=list)


class CoachConfigStatusResponse(BaseModel):
    configured: bool
    api_key_configured: bool = False
    model_configured: bool = False
    base_url_configured: bool = False
    model: str
    base_url: str
    message: str


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


class DeleteProcessedMatchResponse(BaseModel):
    ok: bool
    message: str


class PdfAnalysisResponse(BaseModel):
    normalized_payload: dict[str, Any]
    metrics: dict[str, Any]
    insights: list[str]
    ingestion: dict[str, Any]


class ComputerVisionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: Literal["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"] = "yolov8n.pt"
    player_model_source: Literal["builtin", "custom"] = "builtin"
    ball_model_source: Literal["heuristic", "custom"] = "heuristic"
    pitch_source: Literal["homography", "soccana", "full_field_approx"] = "homography"
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
    quality_control: dict[str, Any] | None = None
    speed_distance: dict[str, Any] | None = None
    scouting_heatmaps: dict[str, Any] | None = None
    homography_telemetry: dict[str, Any] | None = None
    artifacts: dict[str, str | None] | None = None
    warnings: list[str]
    interpretation: list[str]


class ComputerVisionJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    video_name: str
    processing_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ComputerVisionHistoryItem(BaseModel):
    processing_id: str
    job_id: str | None = None
    source_mode: str
    source_label: str
    video_name: str
    status: str
    created_at: str
    updated_at: str
    video_url: str | None = None
    stats_json_url: str | None = None


class ComputerVisionHistoryResponse(BaseModel):
    items: list[ComputerVisionHistoryItem]


class ComputerVisionHistoryDetail(BaseModel):
    metadata: ComputerVisionHistoryItem
    result: ComputerVisionResultResponse


class DeleteComputerVisionHistoryResponse(BaseModel):
    ok: bool
    message: str
