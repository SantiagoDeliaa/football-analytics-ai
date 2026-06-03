from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.services.storage.postgres_event_data_repository import PostgresEventDataRepository
from src.services.storage.settings import PersistenceSettings
from src.services.storage.settings import load_persistence_settings


@dataclass(frozen=True)
class SourceProcessedMatch:
    provider: str
    match_id: str
    competition_name: str
    season_name: str
    home_team: str
    away_team: str
    match_date: str
    raw_path: Path
    canonical_path: Path
    metrics_path: Path

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "competition_name": self.competition_name,
            "season_name": self.season_name,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
        }


@dataclass
class MigrationSummary:
    scanned: int = 0
    planned: int = 0
    migrated: int = 0
    skipped_existing: int = 0
    skipped_missing_files: int = 0
    validation_failed: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "planned": self.planned,
            "migrated": self.migrated,
            "skipped_existing": self.skipped_existing,
            "skipped_missing_files": self.skipped_missing_files,
            "validation_failed": self.validation_failed,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class MigrationLogEntry:
    level: str
    provider: str
    match_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "provider": self.provider,
            "match_id": self.match_id,
            "message": self.message,
        }


class TargetRepository(Protocol):
    def initialize(self) -> None: ...

    def has_processed_match(self, provider: str, match_id: str) -> bool: ...

    def save_processed_match(
        self,
        *,
        provider: str,
        match_id: str,
        match_metadata: dict[str, Any],
        raw_events: Any,
        canonical_events: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]: ...

    def load_processed_match_payloads(self, *, provider: str, match_id: str) -> dict[str, Any] | None: ...


def iter_local_processed_matches(
    *,
    sqlite_db_path: Path,
    provider: str | None = None,
    match_id: str | None = None,
    limit: int | None = None,
) -> list[SourceProcessedMatch]:
    if not sqlite_db_path.exists():
        return []

    query = """
        SELECT
            provider,
            match_id,
            competition_name,
            season_name,
            home_team,
            away_team,
            match_date,
            raw_path,
            canonical_path,
            metrics_path
        FROM processed_matches
        WHERE 1=1
    """
    params: list[Any] = []
    if provider:
        query += " AND provider = ?"
        params.append(provider)
    if match_id:
        query += " AND match_id = ?"
        params.append(str(match_id))
    query += " ORDER BY updated_at DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(sqlite_db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    results: list[SourceProcessedMatch] = []
    for row in rows:
        results.append(
            SourceProcessedMatch(
                provider=str(row["provider"] or ""),
                match_id=str(row["match_id"] or ""),
                competition_name=str(row["competition_name"] or ""),
                season_name=str(row["season_name"] or ""),
                home_team=str(row["home_team"] or ""),
                away_team=str(row["away_team"] or ""),
                match_date=str(row["match_date"] or ""),
                raw_path=Path(str(row["raw_path"] or "")),
                canonical_path=Path(str(row["canonical_path"] or "")),
                metrics_path=Path(str(row["metrics_path"] or "")),
            )
        )
    return results


def validate_source_match_paths(source_match: SourceProcessedMatch) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for path in (source_match.raw_path, source_match.canonical_path, source_match.metrics_path):
        if not path.exists():
            missing.append(str(path))
    return (len(missing) == 0, missing)


def load_source_payloads(source_match: SourceProcessedMatch) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    raw_events = json.loads(source_match.raw_path.read_text(encoding="utf-8"))
    canonical_events = json.loads(source_match.canonical_path.read_text(encoding="utf-8"))
    metrics = json.loads(source_match.metrics_path.read_text(encoding="utf-8"))
    return raw_events, canonical_events, metrics


def validate_remote_payloads(
    repository: TargetRepository,
    *,
    provider: str,
    match_id: str,
) -> bool:
    payloads = repository.load_processed_match_payloads(provider=provider, match_id=match_id)
    if not payloads:
        return False
    return all(key in payloads for key in ("metadata", "raw_events", "canonical_events", "metrics"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_migration_report(
    *,
    mode: str,
    sqlite_db_path: Path,
    source_matches: list[SourceProcessedMatch],
    summary: MigrationSummary,
    logs: list[MigrationLogEntry],
) -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "mode": mode,
        "sqlite_source": str(sqlite_db_path),
        "source_match_count": len(source_matches),
        "summary": summary.to_dict(),
        "entries": [entry.to_dict() for entry in logs],
    }


def write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def migrate_processed_matches(
    *,
    source_matches: list[SourceProcessedMatch],
    target_repository: TargetRepository,
    execute: bool,
    skip_existing: bool = True,
) -> tuple[MigrationSummary, list[MigrationLogEntry]]:
    summary = MigrationSummary()
    logs: list[MigrationLogEntry] = []
    target_repository.initialize()

    for source_match in source_matches:
        summary.scanned += 1
        exists = target_repository.has_processed_match(source_match.provider, source_match.match_id)
        if exists and skip_existing:
            summary.skipped_existing += 1
            logs.append(
                MigrationLogEntry(
                    level="skip_existing",
                    provider=source_match.provider,
                    match_id=source_match.match_id,
                    message=f"SKIP existing {source_match.provider}:{source_match.match_id}.",
                )
            )
            continue

        valid_paths, missing_paths = validate_source_match_paths(source_match)
        if not valid_paths:
            summary.skipped_missing_files += 1
            logs.append(
                MigrationLogEntry(
                    level="skip_missing_files",
                    provider=source_match.provider,
                    match_id=source_match.match_id,
                    message=(
                        f"SKIP missing_files {source_match.provider}:{source_match.match_id} -> "
                        f"{', '.join(missing_paths)}"
                    ),
                )
            )
            continue

        summary.planned += 1
        if not execute:
            logs.append(
                MigrationLogEntry(
                    level="dry_run",
                    provider=source_match.provider,
                    match_id=source_match.match_id,
                    message=f"DRY-RUN would migrate {source_match.provider}:{source_match.match_id}.",
                )
            )
            continue

        try:
            raw_events, canonical_events, metrics = load_source_payloads(source_match)
            target_repository.save_processed_match(
                provider=source_match.provider,
                match_id=source_match.match_id,
                match_metadata=source_match.metadata,
                raw_events=raw_events,
                canonical_events=canonical_events,
                metrics=metrics,
            )
            if validate_remote_payloads(
                target_repository,
                provider=source_match.provider,
                match_id=source_match.match_id,
            ):
                summary.migrated += 1
                logs.append(
                    MigrationLogEntry(
                        level="migrated",
                        provider=source_match.provider,
                        match_id=source_match.match_id,
                        message=f"MIGRATED {source_match.provider}:{source_match.match_id}.",
                    )
                )
            else:
                summary.validation_failed += 1
                logs.append(
                    MigrationLogEntry(
                        level="validation_failed",
                        provider=source_match.provider,
                        match_id=source_match.match_id,
                        message=f"FAILED validation {source_match.provider}:{source_match.match_id}.",
                    )
                )
        except Exception as exc:
            summary.errors += 1
            logs.append(
                MigrationLogEntry(
                    level="error",
                    provider=source_match.provider,
                    match_id=source_match.match_id,
                    message=f"ERROR {source_match.provider}:{source_match.match_id} -> {exc}",
                )
            )

    return summary, logs


def build_target_repository(settings: PersistenceSettings | None = None) -> PostgresEventDataRepository:
    resolved_settings = settings or load_persistence_settings()
    if resolved_settings.persistence_backend != "postgres":
        raise RuntimeError(
            "La migración remota requiere PERSISTENCE_BACKEND=postgres en el entorno de destino."
        )
    return PostgresEventDataRepository(settings=resolved_settings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migra processed_matches desde SQLite/JSON local hacia Neon + storage remoto."
    )
    parser.add_argument(
        "--sqlite-db",
        help="Ruta al SQLite fuente. Si se omite, usa SQLITE_DB_PATH del entorno.",
    )
    parser.add_argument("--provider", help="Filtra por provider.")
    parser.add_argument("--match-id", help="Filtra por match_id.")
    parser.add_argument("--limit", type=int, help="Limita la cantidad de partidos fuente.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta la migración real. Si se omite, corre en dry-run.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="No omite partidos ya existentes en remoto. Usar con cuidado.",
    )
    parser.add_argument(
        "--report-file",
        help="Ruta a un archivo JSON para guardar el resultado estructurado de la ejecución.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_persistence_settings()
    sqlite_db_path = Path(args.sqlite_db).resolve() if args.sqlite_db else settings.sqlite_db_path

    source_matches = iter_local_processed_matches(
        sqlite_db_path=sqlite_db_path,
        provider=args.provider,
        match_id=args.match_id,
        limit=args.limit,
    )

    repository = build_target_repository(settings)
    summary, logs = migrate_processed_matches(
        source_matches=source_matches,
        target_repository=repository,
        execute=bool(args.execute),
        skip_existing=not bool(args.include_existing),
    )

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] SQLite source: {sqlite_db_path}")
    print(f"[{mode}] Matches encontrados: {len(source_matches)}")
    for entry in logs:
        print(entry.message)
    report_payload = build_migration_report(
        mode=mode,
        sqlite_db_path=sqlite_db_path,
        source_matches=source_matches,
        summary=summary,
        logs=logs,
    )
    print(f"SUMMARY {summary.to_dict()}")
    if args.report_file:
        report_path = Path(args.report_file).resolve()
        write_report(report_path, report_payload)
        print(f"REPORT {report_path}")


if __name__ == "__main__":
    main()
