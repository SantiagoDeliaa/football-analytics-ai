from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "db" / "migrations" / "001_tip_postgres_foundation.sql"


def test_postgres_schema_file_exists():
    assert SCHEMA_PATH.exists()


def test_postgres_schema_contains_core_tables():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    expected_snippets = (
        "CREATE TABLE IF NOT EXISTS organizations",
        "CREATE TABLE IF NOT EXISTS organization_matches",
        "CREATE TABLE IF NOT EXISTS match_participants",
        "CREATE TABLE IF NOT EXISTS match_provider_links",
        "CREATE TABLE IF NOT EXISTS storage_objects",
        "CREATE TABLE IF NOT EXISTS match_assets",
        "CREATE TABLE IF NOT EXISTS processing_jobs",
        "CREATE TABLE IF NOT EXISTS event_datasets",
        "CREATE TABLE IF NOT EXISTS metric_sets",
        "CREATE TABLE IF NOT EXISTS quality_summaries",
        "CREATE TABLE IF NOT EXISTS ai_coach_sessions",
        "CREATE TABLE IF NOT EXISTS organization_settings",
    )

    for snippet in expected_snippets:
        assert snippet in sql


def test_postgres_schema_tracks_versioning_and_jsonb_fields():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    assert "schema_version TEXT NOT NULL" in sql
    assert "pipeline_version TEXT NOT NULL DEFAULT 'v1'" in sql
    assert "summary_json JSONB NOT NULL DEFAULT '{}'::jsonb" in sql
    assert "storage_object_id UUID REFERENCES storage_objects(id)" in sql
    assert "produced_by_job_id UUID REFERENCES processing_jobs(id)" in sql
