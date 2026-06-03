from __future__ import annotations

import sqlite3

from src.services.storage.settings import PROJECT_ROOT
from src.services.storage.settings import load_persistence_settings


DB_PATH = load_persistence_settings().sqlite_db_path


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_event_data_db() -> None:
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_matches (
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
                updated_at TEXT,
                UNIQUE(provider, match_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                processing_id TEXT NOT NULL UNIQUE,
                job_id TEXT,
                source_mode TEXT,
                source_label TEXT,
                video_name TEXT,
                source_fingerprint TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                status TEXT,
                result_path TEXT,
                stats_path TEXT,
                video_path TEXT,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(source_fingerprint, config_hash)
            )
            """
        )
        connection.commit()
