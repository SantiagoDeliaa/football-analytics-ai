import sys
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services import open_event_data_ingestion as ingestion


def test_get_available_competitions_returns_real_data_when_loader_succeeds(monkeypatch):
    payload = [
        {
            "competition_id": 11,
            "season_id": 1,
            "competition_name": "La Liga",
            "season_name": "2015/2016",
            "country_name": "Spain",
        }
    ]
    monkeypatch.setattr(ingestion, "_load_remote_json", lambda url: payload)

    competitions = ingestion.get_available_competitions()
    status = ingestion.get_ingestion_status()

    assert competitions[0]["display_name"] == "La Liga - 2015/2016"
    assert status["competitions"]["source"] == "real"


def test_get_available_competitions_falls_back_to_mock_on_network_error(monkeypatch):
    def _raise(_url):
        raise URLError("sin red")

    monkeypatch.setattr(ingestion, "_load_remote_json", _raise)

    competitions = ingestion.get_available_competitions()
    status = ingestion.get_ingestion_status()

    assert competitions
    assert status["competitions"]["source"] == "mock"
    assert "No se pudo cargar competitions.json" in status["competitions"]["message"]


def test_get_available_matches_uses_fallback_when_response_is_invalid(monkeypatch):
    monkeypatch.setattr(ingestion, "_load_remote_json", lambda url: {"unexpected": True})

    matches = ingestion.get_available_matches(43, 106)
    status = ingestion.get_ingestion_status()

    assert matches
    assert matches[0]["home_team"] == "Argentina"
    assert status["matches"]["source"] == "mock"


def test_get_match_events_returns_real_payload_and_updates_status(monkeypatch):
    payload = [{"id": "event-1"}, {"id": "event-2"}]
    monkeypatch.setattr(ingestion, "_load_remote_json", lambda url: payload)

    events = ingestion.get_match_events(999)
    status = ingestion.get_ingestion_status()

    assert events == payload
    assert status["events"]["source"] == "real"


def test_get_match_events_falls_back_without_internet(monkeypatch):
    def _raise(_url):
        raise URLError("offline")

    monkeypatch.setattr(ingestion, "_load_remote_json", _raise)

    events = ingestion.get_match_events(3869685)
    status = ingestion.get_ingestion_status()

    assert events
    assert status["events"]["source"] == "mock"
