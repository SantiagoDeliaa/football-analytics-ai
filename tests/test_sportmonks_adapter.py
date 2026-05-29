import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.providers import sportmonks_adapter as adapter


def _fixture_mock():
    return {
        "id": 999,
        "starting_at": "2026-05-28 20:00:00",
        "participants": [
            {"id": 1, "name": "River Plate", "meta": {"location": "home"}},
            {"id": 2, "name": "Boca Juniors", "meta": {"location": "away"}},
        ],
        "scores": [
            {"participant_id": 1, "score": 2, "description": "current"},
            {"participant_id": 2, "score": 1, "description": "current"},
        ],
        "events": [
            {
                "id": 5001,
                "fixture_id": 999,
                "type": "goal",
                "minute": 18,
                "participant": {"id": 1, "name": "River Plate"},
                "player": {"id": 10, "name": "Colidio"},
            }
        ],
        "statistics": [
            {
                "fixture_id": 999,
                "team_id": 1,
                "team_name": "River Plate",
                "stats": [
                    {"type": "expected-goals", "value": 1.8},
                    {"type": "expected-goals-on-target", "value": 1.4},
                    {"type": "expected-points", "value": 2.1},
                ],
            }
        ],
    }


def test_get_available_sportmonks_competitions_ok(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "client_get_sportmonks_leagues",
        lambda params=None: {
            "ok": True,
            "data": [{"id": 55, "name": "Liga Profesional", "country_name": "Argentina"}],
            "meta": {"page": 1},
        },
    )

    response = adapter.get_available_sportmonks_competitions()

    assert response["ok"] is True
    assert len(response["data"]) == 1
    assert response["data"][0].name == "Liga Profesional"
    assert response["source"] == "sportmonks"


def test_get_available_sportmonks_competitions_client_error(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "client_get_sportmonks_leagues",
        lambda params=None: {"ok": False, "error": "fallo controlado", "meta": {}},
    )

    response = adapter.get_available_sportmonks_competitions()

    assert response["ok"] is False
    assert response["error"] == "fallo controlado"


def test_get_sportmonks_match_context(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "client_get_sportmonks_fixture_full_context",
        lambda fixture_id: {"ok": True, "data": _fixture_mock(), "meta": {"fixture_id": fixture_id}},
    )

    response = adapter.get_sportmonks_match_context("999")

    assert response["ok"] is True
    assert response["data"]["match"] is not None
    assert response["data"]["timeline_events"]
    assert response["data"]["expected_metrics"]
    assert response["data"]["availability"] is not None


def test_get_sportmonks_data_availability_for_fixture(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "client_get_sportmonks_fixture_full_context",
        lambda fixture_id: {"ok": True, "data": _fixture_mock(), "meta": {"fixture_id": fixture_id}},
    )

    response = adapter.get_sportmonks_data_availability_for_fixture("999")

    assert response["ok"] is True
    assert response["data"].has_coordinates is False
