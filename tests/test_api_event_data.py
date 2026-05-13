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
            "match_id": "3869685",
            "competition_name": "FIFA World Cup",
            "match_label": "Argentina vs Francia — 2022-12-18",
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
            "match_id": "555",
            "competition_name": "Liga Profesional",
            "match_label": "River Plate vs Boca Juniors — 2024-05-12",
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
            "match_id": match_id,
            "competition_name": "FIFA World Cup",
            "match_label": "Argentina vs Francia — 2022-12-18",
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
