from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "tip_event_data.sqlite"


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
        connection.commit()
