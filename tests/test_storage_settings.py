from __future__ import annotations

from pathlib import Path

from src.services.storage.settings import PROJECT_ROOT
from src.services.storage.settings import load_persistence_settings


def test_load_persistence_settings_uses_local_defaults(monkeypatch):
    for key in (
        "PERSISTENCE_BACKEND",
        "STORAGE_BACKEND",
        "DATABASE_URL",
        "SQLITE_DB_PATH",
        "LOCAL_STORAGE_ROOT",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "R2_ENDPOINT_URL",
        "R2_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = load_persistence_settings()

    assert settings.persistence_backend == "local"
    assert settings.storage_backend == "local"
    assert settings.database_url == ""
    assert settings.sqlite_db_path == PROJECT_ROOT / "data" / "tip_event_data.sqlite"
    assert settings.local_storage_root == PROJECT_ROOT / "data" / "storage"


def test_load_persistence_settings_resolves_relative_paths(monkeypatch):
    monkeypatch.setenv("PERSISTENCE_BACKEND", "postgres")
    monkeypatch.setenv("STORAGE_BACKEND", "r2")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("SQLITE_DB_PATH", "var/sqlite/dev.sqlite")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", "var/storage")
    monkeypatch.setenv("R2_ACCOUNT_ID", "acc")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://cdn.example.com")

    settings = load_persistence_settings()

    assert settings.persistence_backend == "postgres"
    assert settings.storage_backend == "r2"
    assert settings.require_database_url() == "postgresql://example"
    assert settings.sqlite_db_path == PROJECT_ROOT / Path("var/sqlite/dev.sqlite")
    assert settings.local_storage_root == PROJECT_ROOT / Path("var/storage")
    settings.require_r2_configuration()


def test_load_persistence_settings_builds_database_url_from_split_parts(monkeypatch):
    monkeypatch.setenv("PERSISTENCE_BACKEND", "postgres")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "ep-example.neon.tech")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DATABASE", "neondb")
    monkeypatch.setenv("POSTGRES_USER", "neondb_owner")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret_pass")
    monkeypatch.setenv("POSTGRES_SSLMODE", "require")
    monkeypatch.setenv("POSTGRES_CHANNEL_BINDING", "require")

    settings = load_persistence_settings()

    assert (
        settings.require_database_url()
        == "postgresql://neondb_owner:secret_pass@ep-example.neon.tech:5432/neondb?sslmode=require&channel_binding=require"
    )


def test_require_r2_configuration_raises_for_missing_values(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "r2")
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("R2_BUCKET_NAME", raising=False)
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)

    settings = load_persistence_settings()

    try:
        settings.require_r2_configuration()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Se esperaba RuntimeError por configuracion incompleta de R2.")

    assert "R2_ACCESS_KEY_ID" in message
    assert "R2_BUCKET_NAME" in message
