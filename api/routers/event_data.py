from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from api.schemas import CompetitionResponse
from api.schemas import EventDataAnalyzeRequest
from api.schemas import EventDataResultResponse
from api.schemas import MatchResponse
from api.schemas import ProcessedMatchSummaryResponse
from api.schemas import PdfAnalysisResponse
from api.services.event_data_service import analyze_match
from api.services.event_data_service import analyze_pdf_report
from api.services.event_data_service import list_processed_history
from api.services.event_data_service import list_competitions
from api.services.event_data_service import list_matches
from api.services.event_data_service import load_processed_history_entry


router = APIRouter(prefix="/api/v1/event-data", tags=["event-data"])


@router.get("/competitions", response_model=list[CompetitionResponse])
def get_competitions(provider: str = "StatsBomb Open Data") -> list[dict]:
    return list_competitions(provider=provider)


@router.get("/matches", response_model=list[MatchResponse])
def get_matches(
    competition_id: int | str,
    season_id: int | str,
    provider: str = "StatsBomb Open Data",
) -> list[dict]:
    return list_matches(competition_id=competition_id, season_id=season_id, provider=provider)


@router.post("/analyze", response_model=EventDataResultResponse)
def post_analyze(payload: EventDataAnalyzeRequest) -> dict:
    return analyze_match(payload.model_dump())


@router.post("/pdf", response_model=PdfAnalysisResponse)
def post_pdf(file: UploadFile = File(...)) -> dict:
    return analyze_pdf_report(file)


@router.get("/history", response_model=list[ProcessedMatchSummaryResponse])
def get_history(limit: int = 20) -> list[dict]:
    return list_processed_history(limit=limit)


@router.get("/history/{provider}/{match_id}", response_model=EventDataResultResponse)
def get_history_entry(provider: str, match_id: str) -> dict:
    return load_processed_history_entry(provider=provider, match_id=match_id)
