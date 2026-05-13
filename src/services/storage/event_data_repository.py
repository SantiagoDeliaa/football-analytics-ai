from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.services.storage.database import PROJECT_ROOT
from src.services.storage.database import get_db_connection
from src.services.storage.database import initialize_event_data_db


def _provider_slug(provider: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", provider.strip().lower())
    return normalized.strip("_") or "unknown_provider"


def _match_slug(match_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(match_id))
    return normalized.strip("_") or "unknown_match"


def _build_storage_paths(provider: str, match_id: str) -> tuple[Path, Path, Path]:
    provider_slug = _provider_slug(provider)
    match_slug = _match_slug(match_id)
    base_dir = PROJECT_ROOT / "data" / "event_data"
    raw_dir = base_dir / "raw" / provider_slug
    canonical_dir = base_dir / "canonical" / provider_slug
    metrics_dir = base_dir / "metrics" / provider_slug
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return (
        raw_dir / f"{match_slug}.json",
        canonical_dir / f"{match_slug}.json",
        metrics_dir / f"{match_slug}.json",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_processed_match(
    provider: str,
    match_id: str,
    match_metadata: dict[str, Any],
    raw_events: list[dict[str, Any]],
    canonical_events: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    initialize_event_data_db()
    raw_path, canonical_path, metrics_path = _build_storage_paths(provider, match_id)
    _write_json(raw_path, raw_events)
    _write_json(canonical_path, canonical_events)
    _write_json(metrics_path, metrics)

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO processed_matches (
                provider,
                match_id,
                competition_name,
                season_name,
                home_team,
                away_team,
                match_date,
                raw_path,
                canonical_path,
                metrics_path,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, match_id) DO UPDATE SET
                competition_name=excluded.competition_name,
                season_name=excluded.season_name,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                match_date=excluded.match_date,
                raw_path=excluded.raw_path,
                canonical_path=excluded.canonical_path,
                metrics_path=excluded.metrics_path,
                updated_at=excluded.updated_at
            """,
            (
                provider,
                str(match_id),
                str(match_metadata.get("competition_name", "")),
                str(match_metadata.get("season_name", "")),
                str(match_metadata.get("home_team", "")),
                str(match_metadata.get("away_team", "")),
                str(match_metadata.get("match_date", "")),
                str(raw_path),
                str(canonical_path),
                str(metrics_path),
                now,
                now,
            ),
        )
        connection.commit()

    return {
        "provider": provider,
        "match_id": str(match_id),
        "raw_path": str(raw_path),
        "canonical_path": str(canonical_path),
        "metrics_path": str(metrics_path),
    }


def get_processed_matches(limit: int = 20) -> list[dict[str, Any]]:
    initialize_event_data_db()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                provider,
                match_id,
                competition_name,
                season_name,
                home_team,
                away_team,
                match_date,
                created_at,
                updated_at
            FROM processed_matches
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_processed_match(provider: str, match_id: str) -> dict[str, Any] | None:
    initialize_event_data_db()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT *
            FROM processed_matches
            WHERE provider = ? AND match_id = ?
            LIMIT 1
            """,
            (provider, str(match_id)),
        )
        row = cursor.fetchone()
    return dict(row) if row else None


def load_processed_match_payloads(provider: str, match_id: str) -> dict[str, Any] | None:
    match_row = get_processed_match(provider, match_id)
    if not match_row:
        return None

    raw_path = Path(str(match_row.get("raw_path", "")))
    canonical_path = Path(str(match_row.get("canonical_path", "")))
    metrics_path = Path(str(match_row.get("metrics_path", "")))
    if not raw_path.exists() or not canonical_path.exists() or not metrics_path.exists():
        return None

    return {
        "metadata": match_row,
        "raw_events": _read_json(raw_path),
        "canonical_events": _read_json(canonical_path),
        "metrics": _read_json(metrics_path),
    }


def has_processed_match(provider: str, match_id: str) -> bool:
    return get_processed_match(provider, match_id) is not None
