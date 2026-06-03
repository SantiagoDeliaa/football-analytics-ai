from __future__ import annotations

import re
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Callable

import psycopg
from psycopg.rows import dict_row

from src.services.storage.settings import PersistenceSettings
from src.services.storage.settings import load_persistence_settings
from src.services.storage.storage_service import build_match_object_key
from src.services.storage.storage_service import get_storage_service


DEFAULT_ORGANIZATION_SLUG = "local_demo"
DEFAULT_ORGANIZATION_NAME = "Local Demo"
DEFAULT_ORGANIZATION_TYPE = "demo"
RAW_DATASET_TYPE = "raw_provider_data"
CANONICAL_DATASET_TYPE = "canonical_events"
RAW_ASSET_TYPE = "raw_provider_events"
CANONICAL_ASSET_TYPE = "canonical_events"
METRICS_ASSET_TYPE = "metrics_summary"
CANONICAL_SCHEMA_VERSION = "canonical_event_model_v1"
METRIC_VERSION = "event_data_metrics_v1"
ADAPTER_VERSION = "v1"


def _provider_slug(provider: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", provider.strip().lower())
    return normalized.strip("_") or "unknown_provider"


def _match_slug(match_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(match_id))
    return normalized.strip("_") or "unknown_match"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PostgresEventDataRepository:
    def __init__(
        self,
        *,
        settings: PersistenceSettings | None = None,
        storage_service: Any | None = None,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings or load_persistence_settings()
        self.storage_service = storage_service or get_storage_service(self.settings)
        self.connection_factory = connection_factory or self._default_connection_factory

    def _default_connection_factory(self) -> Any:
        return psycopg.connect(self.settings.require_database_url(), row_factory=dict_row)

    def initialize(self) -> None:
        with closing(self.connection_factory()) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            self._ensure_default_organization(connection)
            connection.commit()

    def save_processed_match(
        self,
        *,
        provider: str,
        match_id: str,
        match_metadata: dict[str, Any],
        raw_events: Any,
        canonical_events: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        with closing(self.connection_factory()) as connection:
            organization = self._ensure_default_organization(connection)
            organization_slug = str(organization["slug"])
            canonical_match_id = self._build_canonical_match_id(provider=provider, match_id=match_id)
            match_row = self._upsert_match(
                connection,
                canonical_match_id=canonical_match_id,
                match_metadata=match_metadata,
            )
            self._upsert_organization_match(
                connection,
                organization_id=str(organization["id"]),
                match_id=str(match_row["id"]),
            )
            provider_link = self._upsert_provider_link(
                connection,
                match_id=str(match_row["id"]),
                provider=provider,
                provider_match_id=match_id,
                match_metadata=match_metadata,
            )

            raw_version = self._get_next_asset_version(
                connection,
                match_id=str(match_row["id"]),
                asset_type=RAW_ASSET_TYPE,
            )
            raw_key = build_match_object_key(
                organization_id=organization_slug,
                match_id=_match_slug(match_id),
                category="raw",
                provider=_provider_slug(provider),
                version=f"v{raw_version}",
                file_name="events.json",
            )
            raw_storage = self.storage_service.save_json(raw_events, raw_key)
            raw_storage_row = self._upsert_storage_object(
                connection,
                metadata=raw_storage,
                asset_version=raw_version,
            )
            self._replace_current_dataset(
                connection,
                match_id=str(match_row["id"]),
                provider_link_id=str(provider_link["id"]),
                dataset_type=RAW_DATASET_TYPE,
                schema_version=ADAPTER_VERSION,
                adapter_version=ADAPTER_VERSION,
                storage_object_id=str(raw_storage_row["id"]),
            )
            self._replace_current_match_asset(
                connection,
                match_id=str(match_row["id"]),
                asset_type=RAW_ASSET_TYPE,
                storage_object_id=str(raw_storage_row["id"]),
            )

            canonical_version = self._get_next_asset_version(
                connection,
                match_id=str(match_row["id"]),
                asset_type=CANONICAL_ASSET_TYPE,
            )
            canonical_key = build_match_object_key(
                organization_id=organization_slug,
                match_id=_match_slug(match_id),
                category="canonical",
                subcategory=CANONICAL_SCHEMA_VERSION,
                version=f"v{canonical_version}",
                file_name="events.json",
            )
            canonical_storage = self.storage_service.save_json(canonical_events, canonical_key)
            canonical_storage_row = self._upsert_storage_object(
                connection,
                metadata=canonical_storage,
                asset_version=canonical_version,
            )
            self._replace_current_dataset(
                connection,
                match_id=str(match_row["id"]),
                provider_link_id=str(provider_link["id"]),
                dataset_type=CANONICAL_DATASET_TYPE,
                schema_version=CANONICAL_SCHEMA_VERSION,
                adapter_version=ADAPTER_VERSION,
                storage_object_id=str(canonical_storage_row["id"]),
            )
            self._replace_current_match_asset(
                connection,
                match_id=str(match_row["id"]),
                asset_type=CANONICAL_ASSET_TYPE,
                storage_object_id=str(canonical_storage_row["id"]),
            )

            metrics_version = self._get_next_asset_version(
                connection,
                match_id=str(match_row["id"]),
                asset_type=METRICS_ASSET_TYPE,
            )
            metrics_key = build_match_object_key(
                organization_id=organization_slug,
                match_id=_match_slug(match_id),
                category="metrics",
                subcategory=METRIC_VERSION,
                version=f"v{metrics_version}",
                file_name="summary.json",
            )
            metrics_storage = self.storage_service.save_json(metrics, metrics_key)
            metrics_storage_row = self._upsert_storage_object(
                connection,
                metadata=metrics_storage,
                asset_version=metrics_version,
            )
            self._replace_metric_set(
                connection,
                match_id=str(match_row["id"]),
                metric_family="event_data_metrics",
                metric_version=METRIC_VERSION,
                summary_json=metrics,
                storage_object_id=str(metrics_storage_row["id"]),
            )
            self._replace_current_match_asset(
                connection,
                match_id=str(match_row["id"]),
                asset_type=METRICS_ASSET_TYPE,
                storage_object_id=str(metrics_storage_row["id"]),
            )
            connection.commit()

        return {
            "provider": provider,
            "match_id": str(match_id),
            "raw_path": raw_key,
            "canonical_path": canonical_key,
            "metrics_path": metrics_key,
        }

    def get_processed_matches(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self.connection_factory()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        mpl.provider,
                        mpl.provider_match_id AS match_id,
                        COALESCE(mpl.provider_payload_summary ->> 'competition_name', '') AS competition_name,
                        COALESCE(mpl.provider_payload_summary ->> 'season_name', '') AS season_name,
                        COALESCE(mpl.provider_payload_summary ->> 'home_team', '') AS home_team,
                        COALESCE(mpl.provider_payload_summary ->> 'away_team', '') AS away_team,
                        COALESCE(mpl.provider_payload_summary ->> 'match_date', '') AS match_date,
                        TO_CHAR(mpl.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SSOF') AS created_at,
                        TO_CHAR(mpl.updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SSOF') AS updated_at
                    FROM match_provider_links AS mpl
                    JOIN organization_matches AS om ON om.match_id = mpl.match_id
                    JOIN organizations AS org ON org.id = om.organization_id
                    WHERE org.slug = %s
                    ORDER BY mpl.updated_at DESC
                    LIMIT %s
                    """,
                    (DEFAULT_ORGANIZATION_SLUG, int(limit)),
                )
                rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_processed_match(self, provider: str, match_id: str) -> dict[str, Any] | None:
        with closing(self.connection_factory()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        m.id AS internal_match_id,
                        mpl.id AS provider_link_id,
                        mpl.provider,
                        mpl.provider_match_id AS match_id,
                        COALESCE(mpl.provider_payload_summary ->> 'competition_name', '') AS competition_name,
                        COALESCE(mpl.provider_payload_summary ->> 'season_name', '') AS season_name,
                        COALESCE(mpl.provider_payload_summary ->> 'home_team', '') AS home_team,
                        COALESCE(mpl.provider_payload_summary ->> 'away_team', '') AS away_team,
                        COALESCE(mpl.provider_payload_summary ->> 'match_date', '') AS match_date,
                        TO_CHAR(mpl.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SSOF') AS created_at,
                        TO_CHAR(mpl.updated_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SSOF') AS updated_at
                    FROM match_provider_links AS mpl
                    JOIN matches AS m ON m.id = mpl.match_id
                    JOIN organization_matches AS om ON om.match_id = m.id
                    JOIN organizations AS org ON org.id = om.organization_id
                    WHERE org.slug = %s AND mpl.provider = %s AND mpl.provider_match_id = %s
                    LIMIT 1
                    """,
                    (DEFAULT_ORGANIZATION_SLUG, provider, str(match_id)),
                )
                row = cursor.fetchone()
        return dict(row) if row else None

    def load_processed_match_payloads(self, *, provider: str, match_id: str) -> dict[str, Any] | None:
        metadata = self.get_processed_match(provider, match_id)
        if not metadata:
            return None

        internal_match_id = str(metadata["internal_match_id"])
        provider_link_id = str(metadata["provider_link_id"])

        with closing(self.connection_factory()) as connection:
            raw_dataset = self._get_current_dataset_storage(
                connection,
                match_id=internal_match_id,
                dataset_type=RAW_DATASET_TYPE,
                provider_link_id=provider_link_id,
            )
            canonical_dataset = self._get_current_dataset_storage(
                connection,
                match_id=internal_match_id,
                dataset_type=CANONICAL_DATASET_TYPE,
                provider_link_id=provider_link_id,
            )
            metrics_storage = self._get_current_metric_storage(connection, match_id=internal_match_id)
            metrics_summary = self._get_metric_summary(connection, match_id=internal_match_id)

        if not raw_dataset or not canonical_dataset:
            return None
        if not self.storage_service.exists(str(raw_dataset["object_key"])):
            return None
        if not self.storage_service.exists(str(canonical_dataset["object_key"])):
            return None

        raw_events = self.storage_service.load_json(str(raw_dataset["object_key"]))
        canonical_events = self.storage_service.load_json(str(canonical_dataset["object_key"]))

        metrics: dict[str, Any]
        if metrics_storage and self.storage_service.exists(str(metrics_storage["object_key"])):
            metrics = self.storage_service.load_json(str(metrics_storage["object_key"]))
        else:
            metrics = metrics_summary or {}

        normalized_metadata = {
            "provider": metadata["provider"],
            "match_id": metadata["match_id"],
            "competition_name": metadata["competition_name"],
            "season_name": metadata["season_name"],
            "home_team": metadata["home_team"],
            "away_team": metadata["away_team"],
            "match_date": metadata["match_date"],
            "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"],
        }
        return {
            "metadata": normalized_metadata,
            "raw_events": raw_events,
            "canonical_events": canonical_events,
            "metrics": metrics,
        }

    def has_processed_match(self, provider: str, match_id: str) -> bool:
        return self.get_processed_match(provider, match_id) is not None

    def delete_processed_match(self, provider: str, match_id: str) -> dict[str, Any]:
        metadata = self.get_processed_match(provider, match_id)
        if not metadata:
            return {
                "ok": False,
                "message": f"No se encontró el partido procesado {provider}:{match_id}.",
            }

        internal_match_id = str(metadata["internal_match_id"])
        warnings: list[str] = []
        with closing(self.connection_factory()) as connection:
            storage_rows = self._get_storage_rows_for_match(connection, match_id=internal_match_id)
            for row in storage_rows:
                object_key = str(row.get("object_key") or "")
                if not object_key:
                    continue
                try:
                    self.storage_service.delete(object_key)
                except Exception as exc:
                    warnings.append(f"{object_key}: {exc}")

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM matches
                    WHERE id = %s
                    """,
                    (internal_match_id,),
                )
                if not warnings and storage_rows:
                    cursor.execute(
                        """
                        DELETE FROM storage_objects
                        WHERE id = ANY(%s)
                        """,
                        ([str(row["id"]) for row in storage_rows],),
                    )
            connection.commit()

        base_message = (
            f"Se eliminó el partido procesado {provider}:{match_id} del historial remoto persistido."
        )
        if warnings:
            return {
                "ok": True,
                "message": f"{base_message} Algunos objetos no pudieron borrarse: {'; '.join(warnings)}",
            }
        return {"ok": True, "message": base_message}

    def _build_canonical_match_id(self, *, provider: str, match_id: str) -> str:
        return f"{_provider_slug(provider)}:{str(match_id).strip()}"

    def _ensure_default_organization(self, connection: Any) -> dict[str, Any]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO organizations (slug, name, type)
                VALUES (%s, %s, %s)
                ON CONFLICT (slug) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    updated_at = NOW()
                RETURNING id, slug, name
                """,
                (DEFAULT_ORGANIZATION_SLUG, DEFAULT_ORGANIZATION_NAME, DEFAULT_ORGANIZATION_TYPE),
            )
            row = cursor.fetchone()
        return dict(row)

    def _upsert_match(
        self,
        connection: Any,
        *,
        canonical_match_id: str,
        match_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        status = "completed"
        match_date = str(match_metadata.get("match_date") or "").strip() or None
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO matches (canonical_match_id, match_date, status)
                VALUES (%s, %s, %s)
                ON CONFLICT (canonical_match_id) DO UPDATE SET
                    match_date = EXCLUDED.match_date,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING id, canonical_match_id
                """,
                (canonical_match_id, match_date, status),
            )
            row = cursor.fetchone()
        return dict(row)

    def _upsert_organization_match(
        self,
        connection: Any,
        *,
        organization_id: str,
        match_id: str,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO organization_matches (organization_id, match_id, visibility, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (organization_id, match_id) DO UPDATE SET
                    visibility = EXCLUDED.visibility,
                    source = EXCLUDED.source,
                    updated_at = NOW()
                """,
                (organization_id, match_id, "owned", DEFAULT_ORGANIZATION_SLUG),
            )

    def _upsert_provider_link(
        self,
        connection: Any,
        *,
        match_id: str,
        provider: str,
        provider_match_id: str,
        match_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload_summary = {
            "competition_name": str(match_metadata.get("competition_name") or ""),
            "season_name": str(match_metadata.get("season_name") or ""),
            "home_team": str(match_metadata.get("home_team") or ""),
            "away_team": str(match_metadata.get("away_team") or ""),
            "match_date": str(match_metadata.get("match_date") or ""),
        }
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO match_provider_links (
                    match_id,
                    provider,
                    provider_match_id,
                    adapter_version,
                    provider_payload_summary
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (provider, provider_match_id) DO UPDATE SET
                    match_id = EXCLUDED.match_id,
                    adapter_version = EXCLUDED.adapter_version,
                    provider_payload_summary = EXCLUDED.provider_payload_summary,
                    updated_at = NOW()
                RETURNING id, match_id, provider, provider_match_id
                """,
                (match_id, provider, str(provider_match_id), ADAPTER_VERSION, psycopg.types.json.Jsonb(payload_summary)),
            )
            row = cursor.fetchone()
        return dict(row)

    def _get_next_asset_version(
        self,
        connection: Any,
        *,
        match_id: str,
        asset_type: str,
    ) -> int:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COALESCE(MAX(so.asset_version), 0) AS max_version
                FROM match_assets AS ma
                LEFT JOIN storage_objects AS so ON so.id = ma.storage_object_id
                WHERE ma.match_id = %s AND ma.asset_type = %s
                """,
                (match_id, asset_type),
            )
            row = cursor.fetchone()
        current = int((row or {}).get("max_version") or 0)
        return current + 1

    def _upsert_storage_object(
        self,
        connection: Any,
        *,
        metadata: dict[str, Any],
        asset_version: int,
    ) -> dict[str, Any]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO storage_objects (
                    storage_provider,
                    bucket,
                    object_key,
                    file_name,
                    mime_type,
                    size_bytes,
                    checksum,
                    asset_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (storage_provider, bucket, object_key, asset_version) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    mime_type = EXCLUDED.mime_type,
                    size_bytes = EXCLUDED.size_bytes,
                    checksum = EXCLUDED.checksum,
                    updated_at = NOW()
                RETURNING id, object_key, asset_version
                """,
                (
                    metadata["storage_provider"],
                    metadata["bucket"],
                    metadata["object_key"],
                    metadata["file_name"],
                    metadata["mime_type"],
                    metadata["size_bytes"],
                    metadata["checksum"],
                    asset_version,
                ),
            )
            row = cursor.fetchone()
        return dict(row)

    def _replace_current_dataset(
        self,
        connection: Any,
        *,
        match_id: str,
        provider_link_id: str,
        dataset_type: str,
        schema_version: str,
        adapter_version: str,
        storage_object_id: str,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE event_datasets
                SET is_current = FALSE, updated_at = NOW()
                WHERE match_id = %s AND dataset_type = %s AND is_current = TRUE
                """,
                (match_id, dataset_type),
            )
            cursor.execute(
                """
                INSERT INTO event_datasets (
                    match_id,
                    provider_link_id,
                    dataset_type,
                    schema_version,
                    adapter_version,
                    storage_object_id,
                    is_current
                )
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (
                    match_id,
                    provider_link_id,
                    dataset_type,
                    schema_version,
                    adapter_version
                ) DO UPDATE SET
                    storage_object_id = EXCLUDED.storage_object_id,
                    is_current = TRUE,
                    updated_at = NOW()
                """,
                (
                    match_id,
                    provider_link_id,
                    dataset_type,
                    schema_version,
                    adapter_version,
                    storage_object_id,
                ),
            )

    def _replace_current_match_asset(
        self,
        connection: Any,
        *,
        match_id: str,
        asset_type: str,
        storage_object_id: str,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE match_assets
                SET is_current = FALSE, updated_at = NOW()
                WHERE match_id = %s AND asset_type = %s AND is_current = TRUE
                """,
                (match_id, asset_type),
            )
            cursor.execute(
                """
                INSERT INTO match_assets (
                    match_id,
                    asset_type,
                    storage_object_id,
                    is_current
                )
                VALUES (%s, %s, %s, TRUE)
                """,
                (match_id, asset_type, storage_object_id),
            )

    def _replace_metric_set(
        self,
        connection: Any,
        *,
        match_id: str,
        metric_family: str,
        metric_version: str,
        summary_json: dict[str, Any],
        storage_object_id: str,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM metric_sets
                WHERE match_id = %s
                  AND metric_family = %s
                  AND metric_version = %s
                  AND scope_type = 'match'
                  AND scope_ref IS NULL
                """,
                (match_id, metric_family, metric_version),
            )
            cursor.execute(
                """
                INSERT INTO metric_sets (
                    match_id,
                    metric_family,
                    metric_version,
                    scope_type,
                    scope_ref,
                    summary_json,
                    storage_object_id
                )
                VALUES (%s, %s, %s, 'match', NULL, %s::jsonb, %s)
                """,
                (
                    match_id,
                    metric_family,
                    metric_version,
                    psycopg.types.json.Jsonb(summary_json),
                    storage_object_id,
                ),
            )

    def _get_current_dataset_storage(
        self,
        connection: Any,
        *,
        match_id: str,
        dataset_type: str,
        provider_link_id: str,
    ) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT so.id, so.object_key
                FROM event_datasets AS ed
                JOIN storage_objects AS so ON so.id = ed.storage_object_id
                WHERE ed.match_id = %s
                  AND ed.provider_link_id = %s
                  AND ed.dataset_type = %s
                  AND ed.is_current = TRUE
                LIMIT 1
                """,
                (match_id, provider_link_id, dataset_type),
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def _get_current_metric_storage(self, connection: Any, *, match_id: str) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT so.id, so.object_key
                FROM metric_sets AS ms
                JOIN storage_objects AS so ON so.id = ms.storage_object_id
                WHERE ms.match_id = %s
                  AND ms.metric_family = 'event_data_metrics'
                  AND ms.metric_version = %s
                  AND ms.scope_type = 'match'
                  AND ms.scope_ref IS NULL
                LIMIT 1
                """,
                (match_id, METRIC_VERSION),
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def _get_metric_summary(self, connection: Any, *, match_id: str) -> dict[str, Any] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT summary_json
                FROM metric_sets
                WHERE match_id = %s
                  AND metric_family = 'event_data_metrics'
                  AND metric_version = %s
                  AND scope_type = 'match'
                  AND scope_ref IS NULL
                LIMIT 1
                """,
                (match_id, METRIC_VERSION),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return dict(row).get("summary_json")

    def _get_storage_rows_for_match(self, connection: Any, *, match_id: str) -> list[dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT so.id, so.object_key
                FROM storage_objects AS so
                WHERE so.id IN (
                    SELECT storage_object_id FROM event_datasets WHERE match_id = %s
                    UNION
                    SELECT storage_object_id FROM metric_sets WHERE match_id = %s
                    UNION
                    SELECT storage_object_id FROM match_assets WHERE match_id = %s
                )
                """,
                (match_id, match_id, match_id),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]
