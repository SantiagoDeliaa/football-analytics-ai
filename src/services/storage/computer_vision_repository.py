from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.services.storage.database import PROJECT_ROOT
from src.services.storage.database import get_db_connection
from src.services.storage.database import initialize_event_data_db


RESULTS_DIR = PROJECT_ROOT / "data" / "computer_vision" / "results"


def build_source_fingerprint(
    *,
    source_mode: str,
    source_path: str | Path,
    source_label: str,
    reference_path: str | Path | None = None,
) -> str:
    resolved_source_path = Path(source_path)
    if source_mode == "upload":
        payload = {
            "source_mode": source_mode,
            "source_label": str(source_label),
            "size": resolved_source_path.stat().st_size if resolved_source_path.exists() else 0,
            "content_hash": _hash_file(resolved_source_path) if resolved_source_path.exists() else "",
        }
        return _hash_payload(payload)

    reference = Path(reference_path) if reference_path is not None else resolved_source_path
    reference = reference.resolve()
    stats = reference.stat() if reference.exists() else None
    payload = {
        "source_mode": source_mode,
        "source_label": str(source_label),
        "normalized_path": str(reference).lower(),
        "size": stats.st_size if stats else 0,
        "mtime_ns": getattr(stats, "st_mtime_ns", 0) if stats else 0,
    }
    return _hash_payload(payload)


def build_config_hash(config_payload: dict[str, Any]) -> str:
    return _hash_payload(config_payload)


def build_processing_id(source_fingerprint: str, config_hash: str) -> str:
    return f"cv-{hashlib.sha256(f'{source_fingerprint}:{config_hash}'.encode('utf-8')).hexdigest()[:24]}"


def save_processed_video(
    processing_id: str,
    job_id: str | None,
    source_mode: str,
    source_label: str,
    video_name: str,
    source_fingerprint: str,
    config_hash: str,
    status: str,
    result_payload: dict[str, Any],
    result_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    initialize_event_data_db()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sidecar_path = RESULTS_DIR / f"{processing_id}.json"
    existing_row = get_processed_video(processing_id)
    warnings: list[str] = []

    try:
        _write_json(sidecar_path, result_payload)
    except Exception as exc:
        return {
            "ok": False,
            "processing_id": processing_id,
            "message": f"No se pudo guardar el resultado persistido: {exc}",
        }

    now = datetime.now(timezone.utc).isoformat()
    created_at = str(existing_row.get("created_at") or now) if existing_row else now

    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO processed_videos (
                    processing_id,
                    job_id,
                    source_mode,
                    source_label,
                    video_name,
                    source_fingerprint,
                    config_hash,
                    status,
                    result_path,
                    stats_path,
                    video_path,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_fingerprint, config_hash) DO UPDATE SET
                    processing_id=excluded.processing_id,
                    job_id=excluded.job_id,
                    source_mode=excluded.source_mode,
                    source_label=excluded.source_label,
                    video_name=excluded.video_name,
                    status=excluded.status,
                    result_path=excluded.result_path,
                    stats_path=excluded.stats_path,
                    video_path=excluded.video_path,
                    updated_at=excluded.updated_at
                """,
                (
                    processing_id,
                    job_id,
                    source_mode,
                    source_label,
                    video_name,
                    source_fingerprint,
                    config_hash,
                    status,
                    str(sidecar_path),
                    str(stats_path) if stats_path else "",
                    str(video_path) if video_path else "",
                    created_at,
                    now,
                ),
            )
            connection.commit()
    except Exception as exc:
        return {
            "ok": False,
            "processing_id": processing_id,
            "message": f"No se pudo registrar el procesamiento en SQLite: {exc}",
        }

    if existing_row:
        for field_name in ("result_path", "stats_path", "video_path"):
            previous_path = str(existing_row.get(field_name) or "").strip()
            current_path = {
                "result_path": str(sidecar_path),
                "stats_path": str(stats_path) if stats_path else "",
                "video_path": str(video_path) if video_path else "",
            }[field_name]
            if previous_path and previous_path != current_path:
                try:
                    path_obj = Path(previous_path)
                    if path_obj.exists():
                        path_obj.unlink()
                except OSError as exc:
                    warnings.append(f"{field_name}: {exc}")

    message = f"Procesamiento {processing_id} guardado en historial local."
    if warnings:
        message = f"{message} Algunos artefactos previos no pudieron limpiarse: {'; '.join(warnings)}"

    return {
        "ok": True,
        "processing_id": processing_id,
        "result_path": str(sidecar_path),
        "stats_path": str(stats_path) if stats_path else "",
        "video_path": str(video_path) if video_path else "",
        "message": message,
    }


def get_processed_videos(limit: int = 20) -> list[dict[str, Any]]:
    initialize_event_data_db()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                processing_id,
                job_id,
                source_mode,
                source_label,
                video_name,
                source_fingerprint,
                config_hash,
                status,
                result_path,
                stats_path,
                video_path,
                created_at,
                updated_at
            FROM processed_videos
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_processed_video(processing_id: str) -> dict[str, Any] | None:
    initialize_event_data_db()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT *
            FROM processed_videos
            WHERE processing_id = ?
            LIMIT 1
            """,
            (str(processing_id),),
        )
        row = cursor.fetchone()
    return dict(row) if row else None


def has_processed_video(source_fingerprint: str, config_hash: str) -> bool:
    return get_processed_video_by_signature(source_fingerprint, config_hash) is not None


def get_processed_video_by_signature(source_fingerprint: str, config_hash: str) -> dict[str, Any] | None:
    initialize_event_data_db()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT *
            FROM processed_videos
            WHERE source_fingerprint = ? AND config_hash = ?
            LIMIT 1
            """,
            (source_fingerprint, config_hash),
        )
        row = cursor.fetchone()
    return dict(row) if row else None


def load_processed_video_payloads(processing_id: str) -> dict[str, Any] | None:
    video_row = get_processed_video(processing_id)
    if not video_row:
        return None

    result_path = Path(str(video_row.get("result_path") or ""))
    if not result_path.exists():
        return None

    return {
        "metadata": video_row,
        "result": _read_json(result_path),
    }


def delete_processed_video(processing_id: str) -> dict[str, Any]:
    initialize_event_data_db()
    video_row = get_processed_video(processing_id)
    if not video_row:
        return {
            "ok": False,
            "message": f"No se encontró el procesamiento guardado {processing_id}.",
        }

    warnings: list[str] = []
    for field_name in ("result_path", "stats_path", "video_path"):
        raw_path = str(video_row.get(field_name) or "").strip()
        if not raw_path:
            continue

        file_path = Path(raw_path)
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as exc:
            warnings.append(f"{field_name}: {exc}")

    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            DELETE FROM processed_videos
            WHERE processing_id = ?
            """,
            (str(processing_id),),
        )
        connection.commit()

    base_message = f"Se eliminó el procesamiento guardado {processing_id} del historial local."
    if warnings:
        return {
            "ok": True,
            "message": f"{base_message} Algunos archivos no pudieron borrarse: {'; '.join(warnings)}",
        }

    return {"ok": True, "message": base_message}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
