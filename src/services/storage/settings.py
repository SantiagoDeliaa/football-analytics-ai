from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PersistenceBackend = Literal["local", "postgres"]
StorageBackend = Literal["local", "r2"]
_ENV_LOADED = False


def _load_local_env_file(force_reload: bool = False) -> None:
    global _ENV_LOADED
    if _ENV_LOADED and not force_reload:
        return

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            pass

    _ENV_LOADED = True


def _normalize_backend(value: str | None, *, allowed: tuple[str, ...], default: str) -> str:
    normalized = str(value or default).strip().lower()
    if normalized not in allowed:
        return default
    return normalized


def _resolve_project_path(raw_path: str | None, *, default_relative_path: str) -> Path:
    candidate = str(raw_path or default_relative_path).strip()
    path = Path(candidate)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _build_database_url_from_parts() -> str:
    host = str(os.getenv("POSTGRES_HOST", "") or "").strip()
    database = str(os.getenv("POSTGRES_DATABASE", "") or "").strip()
    user = str(os.getenv("POSTGRES_USER", "") or "").strip()
    password = str(os.getenv("POSTGRES_PASSWORD", "") or "").strip()
    port = str(os.getenv("POSTGRES_PORT", "") or "").strip()
    sslmode = str(os.getenv("POSTGRES_SSLMODE", "") or "").strip()
    channel_binding = str(os.getenv("POSTGRES_CHANNEL_BINDING", "") or "").strip()

    if not host or not database or not user:
        return ""

    credentials = quote_plus(user)
    if password:
        credentials = f"{credentials}:{quote_plus(password)}"

    host_part = host
    if port:
        host_part = f"{host_part}:{port}"

    query_parts: list[str] = []
    if sslmode:
        query_parts.append(f"sslmode={quote_plus(sslmode)}")
    if channel_binding:
        query_parts.append(f"channel_binding={quote_plus(channel_binding)}")
    query_suffix = f"?{'&'.join(query_parts)}" if query_parts else ""
    return f"postgresql://{credentials}@{host_part}/{quote_plus(database)}{query_suffix}"


@dataclass(frozen=True)
class PersistenceSettings:
    persistence_backend: PersistenceBackend
    storage_backend: StorageBackend
    database_url: str
    sqlite_db_path: Path
    local_storage_root: Path
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    r2_endpoint_url: str
    r2_public_base_url: str

    @property
    def is_postgres_enabled(self) -> bool:
        return self.persistence_backend == "postgres"

    @property
    def is_r2_enabled(self) -> bool:
        return self.storage_backend == "r2"

    def require_database_url(self) -> str:
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL es obligatorio cuando PERSISTENCE_BACKEND=postgres."
            )
        return self.database_url

    def require_r2_configuration(self) -> None:
        required_values = {
            "R2_ACCOUNT_ID": self.r2_account_id,
            "R2_ACCESS_KEY_ID": self.r2_access_key_id,
            "R2_SECRET_ACCESS_KEY": self.r2_secret_access_key,
            "R2_BUCKET_NAME": self.r2_bucket_name,
            "R2_ENDPOINT_URL": self.r2_endpoint_url,
        }
        missing = [key for key, value in required_values.items() if not value]
        if missing:
            raise RuntimeError(
                "Faltan variables obligatorias para STORAGE_BACKEND=r2: "
                + ", ".join(sorted(missing))
            )


def load_persistence_settings() -> PersistenceSettings:
    _load_local_env_file()
    persistence_backend = _normalize_backend(
        os.getenv("PERSISTENCE_BACKEND"),
        allowed=("local", "postgres"),
        default="local",
    )
    storage_backend = _normalize_backend(
        os.getenv("STORAGE_BACKEND"),
        allowed=("local", "r2"),
        default="local",
    )
    database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
    if not database_url and persistence_backend == "postgres":
        database_url = _build_database_url_from_parts()

    return PersistenceSettings(
        persistence_backend=persistence_backend,
        storage_backend=storage_backend,
        database_url=database_url,
        sqlite_db_path=_resolve_project_path(
            os.getenv("SQLITE_DB_PATH"),
            default_relative_path="data/tip_event_data.sqlite",
        ),
        local_storage_root=_resolve_project_path(
            os.getenv("LOCAL_STORAGE_ROOT"),
            default_relative_path="data/storage",
        ),
        r2_account_id=str(os.getenv("R2_ACCOUNT_ID", "") or "").strip(),
        r2_access_key_id=str(os.getenv("R2_ACCESS_KEY_ID", "") or "").strip(),
        r2_secret_access_key=str(os.getenv("R2_SECRET_ACCESS_KEY", "") or "").strip(),
        r2_bucket_name=str(os.getenv("R2_BUCKET_NAME", "") or "").strip(),
        r2_endpoint_url=str(os.getenv("R2_ENDPOINT_URL", "") or "").strip(),
        r2_public_base_url=str(os.getenv("R2_PUBLIC_BASE_URL", "") or "").strip(),
    )
