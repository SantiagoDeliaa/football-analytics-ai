from __future__ import annotations

import sys
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.main import app


client = TestClient(app)


def test_event_data_competitions_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.list_competitions",
        lambda provider="StatsBomb Open Data": [
            {
                "competition_id": 43,
                "season_id": 106,
                "competition_name": "FIFA World Cup",
                "season_name": "2022",
                "country_name": "World",
                "display_name": "FIFA World Cup - 2022",
            }
        ],
    )

    response = client.get("/api/v1/event-data/competitions")

    assert response.status_code == 200
    assert response.json()[0]["competition_name"] == "FIFA World Cup"


def test_event_data_competitions_supports_api_football(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.list_competitions",
        lambda provider="StatsBomb Open Data": [
            {
                "competition_id": 128,
                "season_id": 2024,
                "competition_name": "Liga Profesional",
                "season_name": "2024",
                "country_name": "Argentina",
                "display_name": "Liga Profesional (Argentina) - 2024",
            }
        ],
    )

    response = client.get("/api/v1/event-data/competitions?provider=API-Football")

    assert response.status_code == 200
    assert response.json()[0]["competition_name"] == "Liga Profesional"


def test_event_data_api_football_countries_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.list_api_football_countries",
        lambda: [
            {
                "name": "Argentina",
                "code": "AR",
                "flag": "https://example.com/ar.png",
                "display_name": "Argentina",
            }
        ],
    )

    response = client.get("/api/v1/event-data/api-football/countries")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "Argentina"


def test_event_data_api_football_leagues_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.list_api_football_leagues",
        lambda country, season=None, search=None: [
            {
                "league_id": 128,
                "league_name": "Liga Profesional",
                "country_name": country,
                "type": "League",
                "logo": "https://example.com/league.png",
                "seasons": [2024, 2023],
                "current_season": 2024,
                "display_name": "Liga Profesional (Argentina)",
            }
        ],
    )

    response = client.get("/api/v1/event-data/api-football/leagues?country=Argentina")

    assert response.status_code == 200
    assert response.json()[0]["league_name"] == "Liga Profesional"
    assert response.json()[0]["seasons"] == [2024, 2023]


def test_event_data_api_football_fixtures_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.list_api_football_fixtures",
        lambda league_id, season: [
            {
                "match_id": 555,
                "home_team": "River Plate",
                "away_team": "Boca Juniors",
                "match_date": "2024-05-12",
                "competition": "Liga Profesional",
                "season": str(season),
                "display_name": "River Plate vs Boca Juniors — 2024-05-12",
            }
        ],
    )

    response = client.get("/api/v1/event-data/api-football/fixtures?league_id=128&season=2024")

    assert response.status_code == 200
    assert response.json()[0]["match_id"] == 555


def test_event_data_analyze_returns_400_for_invalid_provider(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.analyze_match",
        lambda payload: (_ for _ in ()).throw(HTTPException(status_code=400, detail="Provider no soportado")),
    )

    response = client.post(
        "/api/v1/event-data/analyze",
        json={
            "provider": "",
            "match_id": 1,
        },
    )

    assert response.status_code == 400


def test_event_data_analyze_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.analyze_match",
        lambda payload: {
            "provider": "StatsBomb Open Data",
            "match_id": "3869685",
            "competition_name": "FIFA World Cup",
            "season_name": "2022",
            "match_label": "Argentina vs Francia — 2022-12-18",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
            "canonical_events": [],
            "metrics": {
                "total_events": 100,
                "total_passes": 50,
                "total_shots": 8,
                "progressive_actions": 12,
                "final_third_actions": 20,
                "recoveries": 7,
                "total_under_pressure": 10,
                "total_xg": 1.6,
            },
            "insights": ["Insight demo"],
            "raw_events_count": 120,
            "used_fallback_events": False,
            "events_status_message": "",
        },
    )

    response = client.post(
        "/api/v1/event-data/analyze",
        json={
            "provider": "StatsBomb Open Data",
            "match_id": 3869685,
            "team": "Argentina",
            "player": "Todos",
        },
    )

    assert response.status_code == 200
    assert response.json()["match_id"] == "3869685"


def test_event_data_analyze_api_football_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.analyze_match",
        lambda payload: {
            "provider": "API-Football",
            "match_id": "555",
            "competition_name": "Liga Profesional",
            "season_name": "2024",
            "match_label": "River Plate vs Boca Juniors — 2024-05-12",
            "home_team": "River Plate",
            "away_team": "Boca Juniors",
            "match_date": "2024-05-12",
            "canonical_events": [],
            "metrics": {
                "total_events": 24,
                "total_passes": 10,
                "total_shots": 4,
                "progressive_actions": 3,
                "final_third_actions": 6,
                "recoveries": 5,
                "total_under_pressure": 2,
                "total_xg": 0.0,
            },
            "insights": ["Insight API-Football"],
            "raw_events_count": 24,
            "used_fallback_events": False,
            "events_status_message": "Datos cargados correctamente desde API-Football.",
        },
    )

    response = client.post(
        "/api/v1/event-data/analyze",
        json={
            "provider": "API-Football",
            "match_id": 555,
            "team": "River Plate",
            "player": "Todos",
        },
    )

    assert response.status_code == 200
    assert response.json()["competition_name"] == "Liga Profesional"


def test_event_data_service_analyze_api_football_sets_provider_without_nameerror(monkeypatch):
    from api.services import event_data_service

    monkeypatch.setattr("api.services.event_data_service._ensure_api_football_configured", lambda: None)
    monkeypatch.setattr("api.services.event_data_service.get_api_football_fixture_events", lambda match_id: [])
    monkeypatch.setattr("api.services.event_data_service.get_api_football_fixture_lineups", lambda match_id: [])
    monkeypatch.setattr("api.services.event_data_service.get_api_football_fixture_statistics", lambda match_id: [])
    monkeypatch.setattr("api.services.event_data_service.get_api_football_fixture_players", lambda match_id: [])
    monkeypatch.setattr("api.services.event_data_service._raise_for_api_football_error_if_needed", lambda items: None)
    monkeypatch.setattr(
        "api.services.event_data_service.normalize_api_football_events_to_canonical",
        lambda raw_events, fixture_id: [],
    )
    monkeypatch.setattr(
        "api.services.event_data_service.calculate_open_event_metrics",
        lambda canonical_events, selected_team, selected_player: {
            "total_events": 0,
            "total_passes": 0,
            "total_shots": 0,
            "progressive_actions": 0,
            "final_third_actions": 0,
            "recoveries": 0,
            "total_under_pressure": 0,
            "total_xg": 0.0,
        },
    )
    monkeypatch.setattr(
        "api.services.event_data_service.generate_open_event_insights",
        lambda metrics, selected_team=None, selected_player=None: ["Insight API-Football"],
    )
    monkeypatch.setattr("api.services.event_data_service.save_processed_match", lambda **kwargs: None)
    monkeypatch.setattr(
        "api.services.event_data_service.get_api_football_status",
        lambda: {"message": "Datos cargados correctamente desde API-Football."},
    )

    result = event_data_service.analyze_match(
        {
            "provider": "API-Football",
            "match_id": "1158466",
            "competition_name": "Copa de la Liga Profesional",
            "season_name": "2024",
            "home_team": "Instituto Cordoba",
            "away_team": "Deportivo Riestra",
            "match_date": "2024-01-25",
        }
    )

    assert result["provider"] == "API-Football"
    assert result["match_id"] == "1158466"


def test_event_data_service_analyze_statsbomb_preserves_statsbomb_provider(monkeypatch):
    from api.services import event_data_service

    monkeypatch.setattr("api.services.event_data_service.get_match_events", lambda match_id: [])
    monkeypatch.setattr(
        "api.services.event_data_service.normalize_events_to_canonical",
        lambda raw_events, match_id: [],
    )
    monkeypatch.setattr(
        "api.services.event_data_service.calculate_open_event_metrics",
        lambda canonical_events, selected_team, selected_player: {
            "total_events": 0,
            "total_passes": 0,
            "total_shots": 0,
            "progressive_actions": 0,
            "final_third_actions": 0,
            "recoveries": 0,
            "total_under_pressure": 0,
            "total_xg": 0.0,
        },
    )
    monkeypatch.setattr(
        "api.services.event_data_service.generate_open_event_insights",
        lambda metrics, selected_team=None, selected_player=None: ["Insight StatsBomb"],
    )
    monkeypatch.setattr("api.services.event_data_service.save_processed_match", lambda **kwargs: None)
    monkeypatch.setattr(
        "api.services.event_data_service.get_ingestion_status",
        lambda: {"events": {"source": "api", "message": "Datos cargados correctamente desde StatsBomb."}},
    )

    result = event_data_service.analyze_match(
        {
            "provider": "StatsBomb Open Data",
            "match_id": "3869685",
            "competition_name": "FIFA World Cup",
            "season_name": "2022",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
        }
    )

    assert result["provider"] == "StatsBomb Open Data"
    assert result["match_id"] == "3869685"


def test_event_data_pdf_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.analyze_pdf_report",
        lambda file: {
            "normalized_payload": {"match_info": {"file_name": "report.pdf"}},
            "metrics": {
                "total_events": 12,
                "total_passes": 4,
                "total_shots": 2,
                "progressive_actions": 3,
                "final_third_actions": 3,
                "recoveries": 1,
                "total_under_pressure": 2,
                "total_xg": 0.3,
            },
            "insights": ["Insight PDF"],
            "ingestion": {
                "status": "ok",
                "parser": "pypdf",
                "page_count": 1,
                "bytes_size": 1024,
                "messages": ["Extracción completada con pypdf."],
            },
        },
    )

    response = client.post(
        "/api/v1/event-data/pdf",
        files={"file": ("report.pdf", b"fake pdf content", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["normalized_payload"]["match_info"]["file_name"] == "report.pdf"
    assert response.json()["ingestion"]["parser"] == "pypdf"


def test_event_data_pdf_returns_400_when_extraction_fails(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.analyze_pdf_report",
        lambda file: (_ for _ in ()).throw(
            HTTPException(status_code=400, detail="No se pudo extraer texto utilizable del PDF.")
        ),
    )

    response = client.post(
        "/api/v1/event-data/pdf",
        files={"file": ("report.pdf", b"fake pdf content", "application/pdf")},
    )

    assert response.status_code == 400


def test_event_data_history_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.list_processed_history",
        lambda limit=20: [
            {
                "provider": "StatsBomb Open Data",
                "match_id": "3869685",
                "competition_name": "FIFA World Cup",
                "season_name": "2022",
                "home_team": "Argentina",
                "away_team": "Francia",
                "match_date": "2022-12-18",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:05+00:00",
            }
        ],
    )

    response = client.get("/api/v1/event-data/history")

    assert response.status_code == 200
    assert response.json()[0]["match_id"] == "3869685"


def test_event_data_history_entry_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.load_processed_history_entry",
        lambda provider, match_id: {
            "provider": provider,
            "match_id": match_id,
            "competition_name": "FIFA World Cup",
            "season_name": "2022",
            "match_label": "Argentina vs Francia — 2022-12-18",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
            "canonical_events": [],
            "metrics": {
                "total_events": 100,
                "total_passes": 50,
                "total_shots": 8,
                "progressive_actions": 12,
                "final_third_actions": 20,
                "recoveries": 7,
                "total_under_pressure": 10,
                "total_xg": 1.6,
            },
            "insights": ["Insight history"],
            "raw_events_count": 120,
            "used_fallback_events": False,
            "events_status_message": "",
        },
    )

    response = client.get("/api/v1/event-data/history/StatsBomb%20Open%20Data/3869685")

    assert response.status_code == 200
    assert response.json()["match_id"] == "3869685"


def test_event_data_coach_status_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.get_ai_coach_status",
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

    response = client.get("/api/v1/event-data/coach/status")

    assert response.status_code == 200
    assert response.json()["configured"] is True


def test_event_data_coach_diagnosis_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.event_data.generate_ai_coach_diagnosis",
        lambda payload: {
            "ok": True,
            "diagnosis": "Diagnóstico táctico de prueba.",
            "error": "",
            "suggested_questions": ["¿Dónde generó más peligro?"],
        },
    )

    response = client.post(
        "/api/v1/event-data/coach/diagnosis",
        json={
            "provider": "StatsBomb Open Data",
            "match_id": 3869685,
            "team": "Argentina",
            "player": "Todos",
            "competition_name": "FIFA World Cup",
            "season_name": "2022",
            "match_label": "Argentina vs Francia — 2022-12-18",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["diagnosis"] == "Diagnóstico táctico de prueba."


def test_event_data_coach_question_returns_400_when_match_id_is_missing():
    response = client.post(
        "/api/v1/event-data/coach/question",
        json={
            "provider": "StatsBomb Open Data",
            "match_id": "",
            "question": "¿Cómo estuvo el equipo?",
        },
    )

    assert response.status_code == 400
