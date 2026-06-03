from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.services.storage.settings import PersistenceSettings
from src.services.storage.settings import load_persistence_settings


def _sanitize_object_key_part(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._=-]+", "_", str(value).strip().lower())
    return normalized.strip("._-") or "unknown"


def _guess_mime_type(path: Path, *, default: str = "application/octet-stream") -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or default


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def build_match_object_key(
    *,
    organization_id: str,
    match_id: str,
    category: str,
    file_name: str,
    provider: str | None = None,
    version: str | None = None,
    subcategory: str | None = None,
) -> str:
    parts = [
        "organizations",
        organization_id,
        "matches",
        match_id,
        category,
    ]
    if provider:
        parts.append(provider)
    if subcategory:
        parts.append(subcategory)
    if version:
        parts.append(version)
    parts.append(file_name)
    return "/".join(_sanitize_object_key_part(part) for part in parts)


class LocalStorageService:
    def __init__(self, root_path: Path, *, bucket_name: str = "local") -> None:
        self.root_path = root_path
        self.bucket_name = bucket_name
        self.root_path.mkdir(parents=True, exist_ok=True)

    def build_object_key(self, *parts: str) -> str:
        sanitized = [_sanitize_object_key_part(part) for part in parts if str(part).strip()]
        return "/".join(sanitized)

    def _resolve_path(self, object_key: str) -> Path:
        normalized = str(object_key).replace("\\", "/").strip("/")
        destination = (self.root_path / normalized).resolve()
        try:
            destination.relative_to(self.root_path.resolve())
        except ValueError as exc:
            raise ValueError(f"Object key fuera del root permitido: {object_key}") from exc
        return destination

    def save_json(self, payload: Any, object_key: str) -> dict[str, Any]:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        destination = self._resolve_path(object_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(serialized)
        return self._build_metadata(
            object_key=object_key,
            payload_size=len(serialized),
            checksum=_sha256_bytes(serialized),
            file_name=destination.name,
            mime_type="application/json",
        )

    def load_json(self, object_key: str) -> Any:
        source = self._resolve_path(object_key)
        return json.loads(source.read_text(encoding="utf-8"))

    def save_file(self, source_path: str, object_key: str) -> dict[str, Any]:
        source = Path(source_path)
        payload = source.read_bytes()
        destination = self._resolve_path(object_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return self._build_metadata(
            object_key=object_key,
            payload_size=len(payload),
            checksum=_sha256_bytes(payload),
            file_name=destination.name,
            mime_type=_guess_mime_type(destination),
        )

    def exists(self, object_key: str) -> bool:
        return self._resolve_path(object_key).exists()

    def get_uri(self, object_key: str) -> str:
        return self._resolve_path(object_key).as_uri()

    def delete(self, object_key: str) -> None:
        path = self._resolve_path(object_key)
        if path.exists():
            path.unlink()

    def _build_metadata(
        self,
        *,
        object_key: str,
        payload_size: int,
        checksum: str,
        file_name: str,
        mime_type: str,
    ) -> dict[str, Any]:
        return {
            "storage_provider": "local",
            "bucket": self.bucket_name,
            "object_key": object_key,
            "file_name": file_name,
            "mime_type": mime_type,
            "size_bytes": payload_size,
            "checksum": checksum,
            "uri": self.get_uri(object_key),
        }


class R2StorageService:
    def __init__(
        self,
        *,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        endpoint_url: str,
        public_base_url: str = "",
        client: Any | None = None,
    ) -> None:
        self.account_id = account_id
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url.rstrip("/")
        self.public_base_url = public_base_url.rstrip("/")
        self.client = client or boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
        )

    def build_object_key(self, *parts: str) -> str:
        sanitized = [_sanitize_object_key_part(part) for part in parts if str(part).strip()]
        return "/".join(sanitized)

    def save_json(self, payload: Any, object_key: str) -> dict[str, Any]:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=serialized,
            ContentType="application/json",
        )
        return self._build_metadata(
            object_key=object_key,
            payload_size=len(serialized),
            checksum=_sha256_bytes(serialized),
            file_name=Path(object_key).name,
            mime_type="application/json",
        )

    def load_json(self, object_key: str) -> Any:
        response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
        raw_payload = response["Body"].read()
        return json.loads(raw_payload.decode("utf-8"))

    def save_file(self, source_path: str, object_key: str) -> dict[str, Any]:
        source = Path(source_path)
        payload = source.read_bytes()
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=payload,
            ContentType=_guess_mime_type(source),
        )
        return self._build_metadata(
            object_key=object_key,
            payload_size=len(payload),
            checksum=_sha256_bytes(payload),
            file_name=source.name,
            mime_type=_guess_mime_type(source),
        )

    def exists(self, object_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def get_uri(self, object_key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{object_key}"
        return f"r2://{self.bucket_name}/{object_key}"

    def delete(self, object_key: str) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=object_key)

    def _build_metadata(
        self,
        *,
        object_key: str,
        payload_size: int,
        checksum: str,
        file_name: str,
        mime_type: str,
    ) -> dict[str, Any]:
        return {
            "storage_provider": "r2",
            "bucket": self.bucket_name,
            "object_key": object_key,
            "file_name": file_name,
            "mime_type": mime_type,
            "size_bytes": payload_size,
            "checksum": checksum,
            "uri": self.get_uri(object_key),
        }


def get_storage_service(
    settings: PersistenceSettings | None = None,
) -> LocalStorageService | R2StorageService:
    resolved_settings = settings or load_persistence_settings()
    if resolved_settings.storage_backend == "r2":
        resolved_settings.require_r2_configuration()
        return R2StorageService(
            account_id=resolved_settings.r2_account_id,
            access_key_id=resolved_settings.r2_access_key_id,
            secret_access_key=resolved_settings.r2_secret_access_key,
            bucket_name=resolved_settings.r2_bucket_name,
            endpoint_url=resolved_settings.r2_endpoint_url,
            public_base_url=resolved_settings.r2_public_base_url,
        )

    return LocalStorageService(root_path=resolved_settings.local_storage_root)
