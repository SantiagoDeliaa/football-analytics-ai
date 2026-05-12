import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services import api_football_ingestion as ingestion


class _FakeResponse:
    def __init__(self, payload: dict, headers: dict[str, str] | None = None):
        self._payload = payload
        self.headers = headers or {}

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_api_football_request_returns_empty_when_key_is_missing(monkeypatch):
    monkeypatch.setattr(ingestion, "_ENV_LOADED", False)
    monkeypatch.setattr(ingestion, "PROJECT_ROOT", Path("Z:/no-env-for-tests"))
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    payload = ingestion.api_football_request("/countries")
    status = ingestion.get_api_football_status()

    assert payload == []
    assert status["status"] == "missing_key"
    assert "API_FOOTBALL_KEY" in status["message"]


def test_api_football_request_returns_response_payload_and_status(monkeypatch):
    monkeypatch.setattr(ingestion, "_ENV_LOADED", True)
    monkeypatch.setenv("API_FOOTBALL_KEY", "demo-key")
    monkeypatch.setattr(
        ingestion,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            {"errors": [], "response": [{"name": "Argentina"}]},
            headers={"x-ratelimit-requests-remaining": "97"},
        ),
    )

    payload = ingestion.api_football_request("/countries")
    status = ingestion.get_api_football_status()

    assert payload == [{"name": "Argentina"}]
    assert status["status"] == "real"
    assert status["requests_remaining"] == "97"


def test_api_football_request_handles_api_errors(monkeypatch):
    monkeypatch.setattr(ingestion, "_ENV_LOADED", True)
    monkeypatch.setenv("API_FOOTBALL_KEY", "demo-key")
    monkeypatch.setattr(
        ingestion,
        "urlopen",
        lambda request, timeout: _FakeResponse({"errors": {"token": "invalid"}, "response": []}),
    )

    payload = ingestion.api_football_request("/countries")
    status = ingestion.get_api_football_status()

    assert payload == []
    assert status["status"] == "error"
    assert status["errors"]


def test_api_football_user_message_translates_plan_errors():
    message = ingestion.get_api_football_user_message(
        {
            "message": "API-Football devolvió errores en la respuesta.",
            "errors": ["plan: Free plans do not have access to this season, try from 2022 to 2024."],
        }
    )

    assert "Tu plan actual no tiene acceso a la temporada seleccionada" in message
