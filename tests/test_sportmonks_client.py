import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.providers import sportmonks_client as client
from src.services.providers.sportmonks_client import (
    build_sportmonks_url,
    extract_sportmonks_payload,
    is_sportmonks_configured,
    sportmonks_get,
)


def test_is_sportmonks_configured_false(monkeypatch):
    monkeypatch.setattr(client, "_ENV_LOADED", True)
    monkeypatch.setattr(client, "PROJECT_ROOT", Path("Z:/no-env-for-tests"))
    monkeypatch.delenv("SPORTMONKS_API_KEY", raising=False)

    assert is_sportmonks_configured() is False


def test_is_sportmonks_configured_true(monkeypatch):
    monkeypatch.setenv("SPORTMONKS_API_KEY", "demo-key")

    assert is_sportmonks_configured() is True


def test_build_sportmonks_url_adds_endpoint_and_params(monkeypatch):
    monkeypatch.setenv("SPORTMONKS_API_KEY", "demo-key")

    url = build_sportmonks_url(
        "leagues",
        params={"include": "country", "per_page": 25, "page": 2},
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.path.endswith("/leagues")
    assert query["include"] == ["country"]
    assert query["per_page"] == ["25"]
    assert query["page"] == ["2"]
    assert query["api_token"] == ["demo-key"]


def test_build_sportmonks_url_does_not_mutate_params(monkeypatch):
    monkeypatch.setenv("SPORTMONKS_API_KEY", "demo-key")
    params = {"include": "participants", "page": 1}
    snapshot = dict(params)

    build_sportmonks_url("/fixtures/123", params=params)

    assert params == snapshot


def test_sportmonks_get_without_key_returns_controlled_error(monkeypatch):
    monkeypatch.setattr(client, "_ENV_LOADED", True)
    monkeypatch.setattr(client, "PROJECT_ROOT", Path("Z:/no-env-for-tests"))
    monkeypatch.delenv("SPORTMONKS_API_KEY", raising=False)

    response = sportmonks_get("leagues")

    assert response["ok"] is False
    assert response["data"] is None
    assert "SPORTMONKS_API_KEY" in response["error"]
    assert response["status_code"] is None


def test_extract_sportmonks_payload_with_data():
    payload = {
        "data": [{"id": 1, "name": "Liga Profesional"}],
        "pagination": {"current_page": 1, "total": 1},
        "rate_limit": {"remaining": 99},
        "subscription": {"plan": "demo"},
    }

    extracted = extract_sportmonks_payload(payload)

    assert extracted["data"] == [{"id": 1, "name": "Liga Profesional"}]
    assert extracted["pagination"] == {"current_page": 1, "total": 1}
    assert extracted["rate_limit"] == {"remaining": 99}
    assert extracted["meta"] == {"subscription": {"plan": "demo"}}


def test_extract_sportmonks_payload_without_optional_fields():
    extracted = extract_sportmonks_payload({"data": []})

    assert extracted["data"] == []
    assert extracted["pagination"] == {}
    assert extracted["rate_limit"] == {}
    assert extracted["meta"] == {}
