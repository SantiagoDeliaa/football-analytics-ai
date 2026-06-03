from __future__ import annotations

from pathlib import Path

from src.services.storage.settings import PersistenceSettings
from src.services.storage.storage_service import LocalStorageService
from src.services.storage.storage_service import build_match_object_key
from src.services.storage.storage_service import get_storage_service


def _build_local_settings(tmp_path: Path) -> PersistenceSettings:
    return PersistenceSettings(
        persistence_backend="local",
        storage_backend="local",
        database_url="",
        sqlite_db_path=tmp_path / "tip.sqlite",
        local_storage_root=tmp_path / "storage",
        r2_account_id="",
        r2_access_key_id="",
        r2_secret_access_key="",
        r2_bucket_name="",
        r2_endpoint_url="",
        r2_public_base_url="",
    )


def test_local_storage_service_saves_and_loads_json(tmp_path):
    service = LocalStorageService(root_path=tmp_path / "storage")
    object_key = service.build_object_key(
        "organizations",
        "local_demo",
        "matches",
        "3895302",
        "canonical",
        "v1",
        "events.json",
    )

    metadata = service.save_json({"ok": True, "events": [1, 2]}, object_key)

    assert metadata["storage_provider"] == "local"
    assert metadata["bucket"] == "local"
    assert metadata["object_key"] == object_key
    assert metadata["mime_type"] == "application/json"
    assert metadata["size_bytes"] > 0
    assert service.exists(object_key) is True
    assert service.load_json(object_key) == {"ok": True, "events": [1, 2]}


def test_local_storage_service_saves_file_and_exposes_file_uri(tmp_path):
    service = LocalStorageService(root_path=tmp_path / "storage")
    source_file = tmp_path / "report.txt"
    source_file.write_text("reporte tactico", encoding="utf-8")
    object_key = service.build_object_key(
        "organizations",
        "local_demo",
        "matches",
        "match-1",
        "reports",
        "v1",
        "report.txt",
    )

    metadata = service.save_file(str(source_file), object_key)

    assert metadata["file_name"] == "report.txt"
    assert metadata["mime_type"] == "text/plain"
    assert metadata["uri"].startswith("file:///")
    saved_path = Path(service.get_uri(object_key).replace("file:///", "")).exists()
    assert saved_path is True
    service.delete(object_key)
    assert service.exists(object_key) is False


def test_build_match_object_key_generates_multitenant_path():
    object_key = build_match_object_key(
        organization_id="Local Demo",
        match_id="River vs Boca 2026/05/17",
        category="metrics",
        subcategory="tactical metrics",
        version="event_metrics_v1",
        file_name="summary.json",
        provider="StatsBomb Open Data",
    )

    assert (
        object_key
        == "organizations/local_demo/matches/river_vs_boca_2026_05_17/metrics/statsbomb_open_data/tactical_metrics/event_metrics_v1/summary.json"
    )


def test_get_storage_service_returns_local_service(tmp_path):
    settings = _build_local_settings(tmp_path)

    service = get_storage_service(settings)

    assert isinstance(service, LocalStorageService)
