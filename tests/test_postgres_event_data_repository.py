from __future__ import annotations

from pathlib import Path

from src.services.storage.postgres_event_data_repository import PostgresEventDataRepository
from src.services.storage.settings import PersistenceSettings


class DummyConnection:
    def __init__(self, rows: list[dict] | None = None, row: dict | None = None) -> None:
        self.rows = rows or []
        self.row = row
        self.executed: list[tuple[str, tuple | list | None]] = []
        self.committed = False

    def cursor(self):
        return DummyCursor(self)

    def close(self) -> None:
        return None

    def commit(self) -> None:
        self.committed = True


class DummyCursor:
    def __init__(self, connection: DummyConnection) -> None:
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query: str, params=None) -> None:
        self.connection.executed.append((query, params))

    def fetchall(self):
        return self.connection.rows

    def fetchone(self):
        return self.connection.row


class FakeStorageService:
    def __init__(self) -> None:
        self.saved: dict[str, object] = {}
        self.deleted: list[str] = []

    def save_json(self, payload, object_key: str) -> dict[str, object]:
        self.saved[object_key] = payload
        return {
            "storage_provider": "local",
            "bucket": "local",
            "object_key": object_key,
            "file_name": Path(object_key).name,
            "mime_type": "application/json",
            "size_bytes": len(str(payload)),
            "checksum": f"sha-{len(self.saved)}",
            "uri": f"file:///{object_key}",
        }

    def load_json(self, object_key: str):
        return self.saved[object_key]

    def save_file(self, source_path: str, object_key: str) -> dict[str, object]:
        raise NotImplementedError()

    def exists(self, object_key: str) -> bool:
        return object_key in self.saved

    def get_uri(self, object_key: str) -> str:
        return f"file:///{object_key}"

    def build_object_key(self, *parts: str) -> str:
        return "/".join(parts)

    def delete(self, object_key: str) -> None:
        self.deleted.append(object_key)
        self.saved.pop(object_key, None)


def _build_settings(tmp_path: Path) -> PersistenceSettings:
    return PersistenceSettings(
        persistence_backend="postgres",
        storage_backend="local",
        database_url="postgresql://example",
        sqlite_db_path=tmp_path / "tip.sqlite",
        local_storage_root=tmp_path / "storage",
        r2_account_id="",
        r2_access_key_id="",
        r2_secret_access_key="",
        r2_bucket_name="",
        r2_endpoint_url="",
        r2_public_base_url="",
    )


def test_postgres_repository_save_processed_match_persists_three_payloads(monkeypatch, tmp_path):
    storage = FakeStorageService()
    connection = DummyConnection()
    repository = PostgresEventDataRepository(
        settings=_build_settings(tmp_path),
        storage_service=storage,
        connection_factory=lambda: connection,
    )

    monkeypatch.setattr(
        repository,
        "_ensure_default_organization",
        lambda conn: {"id": "org-1", "slug": "local_demo"},
    )
    monkeypatch.setattr(
        repository,
        "_upsert_match",
        lambda conn, canonical_match_id, match_metadata: {"id": "match-1", "canonical_match_id": canonical_match_id},
    )
    monkeypatch.setattr(repository, "_upsert_organization_match", lambda conn, organization_id, match_id: None)
    monkeypatch.setattr(
        repository,
        "_upsert_provider_link",
        lambda conn, match_id, provider, provider_match_id, match_metadata: {
            "id": "provider-link-1",
            "match_id": match_id,
        },
    )
    versions = iter([1, 1, 1])
    monkeypatch.setattr(
        repository,
        "_get_next_asset_version",
        lambda conn, match_id, asset_type: next(versions),
    )
    monkeypatch.setattr(
        repository,
        "_upsert_storage_object",
        lambda conn, metadata, asset_version: {"id": f"storage-{asset_version}", "object_key": metadata["object_key"]},
    )
    monkeypatch.setattr(repository, "_replace_current_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(repository, "_replace_current_match_asset", lambda *args, **kwargs: None)
    monkeypatch.setattr(repository, "_replace_metric_set", lambda *args, **kwargs: None)

    result = repository.save_processed_match(
        provider="StatsBomb Open Data",
        match_id="3895302",
        match_metadata={
            "competition_name": "World Cup",
            "season_name": "2022",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
        },
        raw_events=[{"id": 1}],
        canonical_events=[{"event_id": "1"}],
        metrics={"total_events": 1},
    )

    assert connection.committed is True
    assert len(storage.saved) == 3
    assert result["provider"] == "StatsBomb Open Data"
    assert result["match_id"] == "3895302"
    assert result["raw_path"].endswith("/raw/statsbomb_open_data/v1/events.json")
    assert result["canonical_path"].endswith("/canonical/canonical_event_model_v1/v1/events.json")
    assert result["metrics_path"].endswith("/metrics/event_data_metrics_v1/v1/summary.json")


def test_postgres_repository_get_processed_matches_returns_rows(tmp_path):
    connection = DummyConnection(
        rows=[
            {
                "provider": "StatsBomb Open Data",
                "match_id": "3895302",
                "competition_name": "World Cup",
                "season_name": "2022",
                "home_team": "Argentina",
                "away_team": "Francia",
                "match_date": "2022-12-18",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:05+00:00",
            }
        ]
    )
    repository = PostgresEventDataRepository(
        settings=_build_settings(tmp_path),
        storage_service=FakeStorageService(),
        connection_factory=lambda: connection,
    )

    rows = repository.get_processed_matches(limit=10)

    assert rows[0]["match_id"] == "3895302"
    assert "FROM match_provider_links" in connection.executed[0][0]


def test_postgres_repository_load_processed_match_payloads_uses_storage(tmp_path, monkeypatch):
    storage = FakeStorageService()
    storage.saved["organizations/local_demo/matches/3895302/raw/statsbomb_open_data/v1/events.json"] = [{"id": 1}]
    storage.saved["organizations/local_demo/matches/3895302/canonical/canonical_event_model_v1/v1/events.json"] = [
        {"event_id": "1"}
    ]
    storage.saved["organizations/local_demo/matches/3895302/metrics/event_data_metrics_v1/v1/summary.json"] = {
        "total_events": 1
    }
    repository = PostgresEventDataRepository(
        settings=_build_settings(tmp_path),
        storage_service=storage,
        connection_factory=lambda: DummyConnection(),
    )

    monkeypatch.setattr(
        repository,
        "get_processed_match",
        lambda provider, match_id: {
            "internal_match_id": "match-1",
            "provider_link_id": "provider-link-1",
            "provider": provider,
            "match_id": match_id,
            "competition_name": "World Cup",
            "season_name": "2022",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:05+00:00",
        },
    )
    monkeypatch.setattr(
        repository,
        "_get_current_dataset_storage",
        lambda conn, match_id, dataset_type, provider_link_id: {
            "object_key": (
                "organizations/local_demo/matches/3895302/raw/statsbomb_open_data/v1/events.json"
                if dataset_type == "raw_provider_data"
                else "organizations/local_demo/matches/3895302/canonical/canonical_event_model_v1/v1/events.json"
            )
        },
    )
    monkeypatch.setattr(
        repository,
        "_get_current_metric_storage",
        lambda conn, match_id: {
            "object_key": "organizations/local_demo/matches/3895302/metrics/event_data_metrics_v1/v1/summary.json"
        },
    )
    monkeypatch.setattr(repository, "_get_metric_summary", lambda conn, match_id: {"total_events": 1})

    payloads = repository.load_processed_match_payloads(
        provider="StatsBomb Open Data",
        match_id="3895302",
    )

    assert payloads is not None
    assert payloads["raw_events"] == [{"id": 1}]
    assert payloads["canonical_events"] == [{"event_id": "1"}]
    assert payloads["metrics"] == {"total_events": 1}


def test_postgres_repository_delete_processed_match_deletes_objects_and_match(tmp_path, monkeypatch):
    storage = FakeStorageService()
    connection = DummyConnection()
    repository = PostgresEventDataRepository(
        settings=_build_settings(tmp_path),
        storage_service=storage,
        connection_factory=lambda: connection,
    )

    monkeypatch.setattr(
        repository,
        "get_processed_match",
        lambda provider, match_id: {
            "internal_match_id": "match-1",
            "provider": provider,
            "match_id": match_id,
        },
    )
    monkeypatch.setattr(
        repository,
        "_get_storage_rows_for_match",
        lambda conn, match_id: [
            {"id": "storage-1", "object_key": "org/match/raw.json"},
            {"id": "storage-2", "object_key": "org/match/canonical.json"},
        ],
    )

    result = repository.delete_processed_match("StatsBomb Open Data", "3895302")

    assert result["ok"] is True
    assert storage.deleted == ["org/match/raw.json", "org/match/canonical.json"]
    assert connection.committed is True
    assert any("DELETE FROM matches" in query for query, _ in connection.executed)
