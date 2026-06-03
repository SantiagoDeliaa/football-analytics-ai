from __future__ import annotations

from io import BytesIO
from pathlib import Path

from botocore.exceptions import ClientError

from src.services.storage.settings import PersistenceSettings
from src.services.storage.storage_service import R2StorageService
from src.services.storage.storage_service import get_storage_service


class FakeR2Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, object]] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        body = self.objects[(Bucket, Key)]["Body"]
        return {"Body": BytesIO(body)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {
                    "Error": {
                        "Code": "404",
                        "Message": "Not found",
                    }
                },
                "HeadObject",
            )
        return {}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.objects.pop((Bucket, Key), None)


def _build_r2_settings(tmp_path: Path) -> PersistenceSettings:
    return PersistenceSettings(
        persistence_backend="postgres",
        storage_backend="r2",
        database_url="postgresql://example",
        sqlite_db_path=tmp_path / "tip.sqlite",
        local_storage_root=tmp_path / "storage",
        r2_account_id="acc",
        r2_access_key_id="key",
        r2_secret_access_key="secret",
        r2_bucket_name="tip-bucket",
        r2_endpoint_url="https://example.r2.cloudflarestorage.com",
        r2_public_base_url="https://cdn.tip.local",
    )


def test_r2_storage_service_saves_and_loads_json():
    client = FakeR2Client()
    service = R2StorageService(
        account_id="acc",
        access_key_id="key",
        secret_access_key="secret",
        bucket_name="tip-bucket",
        endpoint_url="https://example.r2.cloudflarestorage.com",
        public_base_url="https://cdn.tip.local",
        client=client,
    )
    object_key = service.build_object_key(
        "organizations",
        "org-1",
        "matches",
        "match-1",
        "canonical",
        "v1",
        "events.json",
    )

    metadata = service.save_json({"events": [1, 2, 3]}, object_key)

    assert metadata["storage_provider"] == "r2"
    assert metadata["bucket"] == "tip-bucket"
    assert metadata["uri"] == f"https://cdn.tip.local/{object_key}"
    assert service.exists(object_key) is True
    assert service.load_json(object_key) == {"events": [1, 2, 3]}


def test_r2_storage_service_saves_files_and_reports_missing_objects(tmp_path):
    client = FakeR2Client()
    service = R2StorageService(
        account_id="acc",
        access_key_id="key",
        secret_access_key="secret",
        bucket_name="tip-bucket",
        endpoint_url="https://example.r2.cloudflarestorage.com",
        client=client,
    )
    source_file = tmp_path / "annotated.mp4"
    source_file.write_bytes(b"video-bytes")
    object_key = service.build_object_key(
        "organizations",
        "org-1",
        "matches",
        "match-1",
        "videos",
        "annotated_v1.mp4",
    )

    metadata = service.save_file(str(source_file), object_key)

    assert metadata["file_name"] == "annotated.mp4"
    assert metadata["mime_type"] == "video/mp4"
    assert service.exists(object_key) is True
    service.delete(object_key)
    assert service.exists(object_key) is False
    assert service.exists("organizations/org-1/matches/missing/file.json") is False


def test_get_storage_service_returns_r2_service(tmp_path):
    settings = _build_r2_settings(tmp_path)

    service = get_storage_service(settings)

    assert isinstance(service, R2StorageService)
    assert service.bucket_name == "tip-bucket"
