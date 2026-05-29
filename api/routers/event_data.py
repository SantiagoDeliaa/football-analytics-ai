from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from api.schemas import ApiFootballCountryResponse
from api.schemas import ApiFootballLeagueResponse
from api.schemas import CompetitionResponse
from api.schemas import CoachAnswerResponse
from api.schemas import CoachConfigStatusResponse
from api.schemas import CoachDiagnosisResponse
from api.schemas import DeleteProcessedMatchResponse
from api.schemas import EventDataAnalyzeRequest
from api.schemas import EventDataCoachQuestionRequest
from api.schemas import EventDataCoachRequest
from api.schemas import EventDataResultResponse
from api.schemas import MatchResponse
from api.schemas import ProcessedMatchSummaryResponse
from api.schemas import PdfAnalysisResponse
from api.schemas import SportmonksMatchCenterResponse
from api.services.event_data_service import answer_ai_coach_from_payload
from api.services.event_data_service import analyze_match
from api.services.event_data_service import analyze_pdf_report
from api.services.event_data_service import delete_processed_history_entry
from api.services.event_data_service import generate_ai_coach_diagnosis
from api.services.event_data_service import get_ai_coach_status
from api.services.event_data_service import get_sportmonks_match_center_payload
from api.services.event_data_service import list_api_football_countries
from api.services.event_data_service import list_api_football_fixtures
from api.services.event_data_service import list_api_football_leagues
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


@router.get("/api-football/countries", response_model=list[ApiFootballCountryResponse])
def get_api_football_countries_endpoint() -> list[dict]:
    return list_api_football_countries()


@router.get("/api-football/leagues", response_model=list[ApiFootballLeagueResponse])
def get_api_football_leagues_endpoint(
    country: str,
    season: int | str | None = None,
    search: str | None = None,
) -> list[dict]:
    return list_api_football_leagues(country=country, season=season, search=search)


@router.get("/api-football/fixtures", response_model=list[MatchResponse])
def get_api_football_fixtures_endpoint(league_id: int | str, season: int | str) -> list[dict]:
    return list_api_football_fixtures(league_id=league_id, season=season)


@router.get(
    "/providers/sportmonks/matches/{match_id}/match-center",
    response_model=SportmonksMatchCenterResponse,
)
def get_sportmonks_match_center_endpoint(match_id: str) -> dict:
    return get_sportmonks_match_center_payload(match_id)


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


@router.delete("/history/{provider}/{match_id}", response_model=DeleteProcessedMatchResponse)
def delete_history_entry(provider: str, match_id: str) -> dict:
    return delete_processed_history_entry(provider=provider, match_id=match_id)


@router.get("/coach/status", response_model=CoachConfigStatusResponse)
def get_coach_status() -> dict:
    return get_ai_coach_status()


@router.post("/coach/diagnosis", response_model=CoachDiagnosisResponse)
def post_coach_diagnosis(payload: EventDataCoachRequest) -> dict:
    return generate_ai_coach_diagnosis(payload.model_dump())


@router.post("/coach/question", response_model=CoachAnswerResponse)
def post_coach_question(payload: EventDataCoachQuestionRequest) -> dict:
    return answer_ai_coach_from_payload(payload.model_dump())
