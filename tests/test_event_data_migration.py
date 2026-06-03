from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.migrate_event_data_to_remote import build_migration_report
from scripts.migrate_event_data_to_remote import iter_local_processed_matches
from scripts.migrate_event_data_to_remote import migrate_processed_matches
from scripts.migrate_event_data_to_remote import write_report


def _create_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    metrics_path = tmp_path / "metrics.json"
    raw_path.write_text('[{"id": "raw-1"}]', encoding="utf-8")
    canonical_path.write_text('[{"event_id": "1"}]', encoding="utf-8")
    metrics_path.write_text('{"total_events": 1}', encoding="utf-8")

    sqlite_path = tmp_path / "tip_event_data.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            CREATE TABLE processed_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                match_id TEXT NOT NULL,
                competition_name TEXT,
                season_name TEXT,
                home_team TEXT,
                away_team TEXT,
                match_date TEXT,
                raw_path TEXT,
                canonical_path TEXT,
                metrics_path TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        connection.execute(
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
            """,
            (
                "StatsBomb Open Data",
                "3895302",
                "World Cup",
                "2022",
                "Argentina",
                "France",
                "2022-12-18",
                str(raw_path),
                str(canonical_path),
                str(metrics_path),
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        connection.commit()
    return sqlite_path, raw_path, canonical_path, metrics_path


class FakeTargetRepository:
    def __init__(self, *, existing: bool = False, validate_after_save: bool = True) -> None:
        self.existing = existing
        self.validate_after_save = validate_after_save
        self.initialized = False
        self.saved_payloads: dict[tuple[str, str], dict[str, Any]] = {}

    def initialize(self) -> None:
        self.initialized = True

    def has_processed_match(self, provider: str, match_id: str) -> bool:
        return self.existing or (provider, match_id) in self.saved_payloads

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
        self.saved_payloads[(provider, match_id)] = {
            "metadata": match_metadata,
            "raw_events": raw_events,
            "canonical_events": canonical_events,
            "metrics": metrics,
        }
        return {"provider": provider, "match_id": match_id}

    def load_processed_match_payloads(self, *, provider: str, match_id: str) -> dict[str, Any] | None:
        if not self.validate_after_save:
            return None
        return self.saved_payloads.get((provider, match_id))


def test_iter_local_processed_matches_reads_sqlite_source(tmp_path: Path):
    sqlite_path, *_ = _create_source_fixture(tmp_path)

    matches = iter_local_processed_matches(sqlite_db_path=sqlite_path)

    assert len(matches) == 1
    assert matches[0].provider == "StatsBomb Open Data"
    assert matches[0].match_id == "3895302"
    assert matches[0].metadata["home_team"] == "Argentina"


def test_migrate_processed_matches_dry_run_does_not_write_target(tmp_path: Path):
    sqlite_path, *_ = _create_source_fixture(tmp_path)
    source_matches = iter_local_processed_matches(sqlite_db_path=sqlite_path)
    repository = FakeTargetRepository()

    summary, logs = migrate_processed_matches(
        source_matches=source_matches,
        target_repository=repository,
        execute=False,
    )

    assert repository.initialized is True
    assert repository.saved_payloads == {}
    assert summary.scanned == 1
    assert summary.planned == 1
    assert summary.migrated == 0
    assert any("DRY-RUN would migrate" in entry.message for entry in logs)


def test_migrate_processed_matches_skips_existing_by_default(tmp_path: Path):
    sqlite_path, *_ = _create_source_fixture(tmp_path)
    source_matches = iter_local_processed_matches(sqlite_db_path=sqlite_path)
    repository = FakeTargetRepository(existing=True)

    summary, logs = migrate_processed_matches(
        source_matches=source_matches,
        target_repository=repository,
        execute=True,
    )

    assert repository.saved_payloads == {}
    assert summary.scanned == 1
    assert summary.skipped_existing == 1
    assert summary.migrated == 0
    assert any("SKIP existing" in entry.message for entry in logs)


def test_migrate_processed_matches_executes_and_validates(tmp_path: Path):
    sqlite_path, *_ = _create_source_fixture(tmp_path)
    source_matches = iter_local_processed_matches(sqlite_db_path=sqlite_path)
    repository = FakeTargetRepository()

    summary, logs = migrate_processed_matches(
        source_matches=source_matches,
        target_repository=repository,
        execute=True,
    )

    assert ("StatsBomb Open Data", "3895302") in repository.saved_payloads
    assert summary.scanned == 1
    assert summary.planned == 1
    assert summary.migrated == 1
    assert summary.validation_failed == 0
    assert summary.errors == 0
    assert any("MIGRATED StatsBomb Open Data:3895302." == entry.message for entry in logs)


def test_build_migration_report_and_write_report(tmp_path: Path):
    sqlite_path, *_ = _create_source_fixture(tmp_path)
    source_matches = iter_local_processed_matches(sqlite_db_path=sqlite_path)
    repository = FakeTargetRepository()
    summary, logs = migrate_processed_matches(
        source_matches=source_matches,
        target_repository=repository,
        execute=False,
    )

    report = build_migration_report(
        mode="DRY-RUN",
        sqlite_db_path=sqlite_path,
        source_matches=source_matches,
        summary=summary,
        logs=logs,
    )
    report_path = tmp_path / "migration-report.json"
    write_report(report_path, report)

    assert report["mode"] == "DRY-RUN"
    assert report["summary"]["planned"] == 1
    assert report["entries"][0]["level"] == "dry_run"
    assert report_path.exists() is True
