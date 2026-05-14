import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.storage import database
from src.services.storage import event_data_repository as repository


def _configure_temp_storage(monkeypatch, tmp_path):
    db_path = tmp_path / "data" / "tip_event_data.sqlite"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(repository, "PROJECT_ROOT", tmp_path)
    return db_path


def test_initialize_event_data_db_creates_sqlite_file(monkeypatch, tmp_path):
    db_path = _configure_temp_storage(monkeypatch, tmp_path)

    database.initialize_event_data_db()

    assert db_path.exists()


def test_repository_save_and_load_processed_match(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    raw_events = [{"id": "raw-1"}]
    canonical_events = [{"event_id": "1", "event_type": "Pass"}]
    metrics = {"total_events": 1, "field_tilt_index": 50.0}

    saved = repository.save_processed_match(
        provider="statsbomb",
        match_id="12345",
        match_metadata={
            "competition_name": "FIFA World Cup",
            "season_name": "2022",
            "home_team": "Argentina",
            "away_team": "Francia",
            "match_date": "2022-12-18",
        },
        raw_events=raw_events,
        canonical_events=canonical_events,
        metrics=metrics,
    )

    assert Path(saved["raw_path"]).exists()
    assert Path(saved["canonical_path"]).exists()
    assert Path(saved["metrics_path"]).exists()
    assert repository.has_processed_match("statsbomb", "12345") is True

    row = repository.get_processed_match("statsbomb", "12345")
    assert row is not None
    assert row["competition_name"] == "FIFA World Cup"
    assert row["home_team"] == "Argentina"

    payloads = repository.load_processed_match_payloads("statsbomb", "12345")
    assert payloads is not None
    assert payloads["raw_events"] == raw_events
    assert payloads["canonical_events"] == canonical_events
    assert payloads["metrics"] == metrics


def test_get_processed_matches_returns_latest_rows(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)

    repository.save_processed_match(
        provider="statsbomb",
        match_id="1",
        match_metadata={},
        raw_events=[],
        canonical_events=[],
        metrics={"total_events": 0},
    )
    repository.save_processed_match(
        provider="statsbomb",
        match_id="2",
        match_metadata={"competition_name": "UEFA Euro"},
        raw_events=[],
        canonical_events=[],
        metrics={"total_events": 0},
    )

    rows = repository.get_processed_matches(limit=5)

    assert len(rows) == 2
    assert {row["match_id"] for row in rows} == {"1", "2"}


def test_repository_can_store_combined_raw_payload_for_api_football(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    raw_payload = {
        "events": [{"type": "Goal"}],
        "lineups": [{"team": {"name": "River Plate"}}],
        "statistics": [],
        "players": [],
    }

    repository.save_processed_match(
        provider="api_football",
        match_id="fixture-1",
        match_metadata={"competition_name": "Liga Profesional"},
        raw_events=raw_payload,
        canonical_events=[{"event_id": "1"}],
        metrics={"total_events": 1},
    )

    payloads = repository.load_processed_match_payloads("api_football", "fixture-1")
    assert payloads is not None
    assert payloads["raw_events"] == raw_payload


def test_load_processed_match_payloads_returns_none_if_files_are_missing(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    repository.save_processed_match(
        provider="statsbomb",
        match_id="abc",
        match_metadata={},
        raw_events=[],
        canonical_events=[],
        metrics={"total_events": 0},
    )

    row = repository.get_processed_match("statsbomb", "abc")
    assert row is not None
    Path(row["metrics_path"]).unlink()

    assert repository.load_processed_match_payloads("statsbomb", "abc") is None


def test_delete_processed_match_removes_row_and_files(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    saved = repository.save_processed_match(
        provider="statsbomb",
        match_id="to-delete",
        match_metadata={},
        raw_events=[{"id": "raw-1"}],
        canonical_events=[{"event_id": "1"}],
        metrics={"total_events": 1},
    )

    result = repository.delete_processed_match("statsbomb", "to-delete")

    assert result["ok"] is True
    assert repository.get_processed_match("statsbomb", "to-delete") is None
    assert not Path(saved["raw_path"]).exists()
    assert not Path(saved["canonical_path"]).exists()
    assert not Path(saved["metrics_path"]).exists()


def test_delete_processed_match_succeeds_when_some_json_is_missing(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    saved = repository.save_processed_match(
        provider="statsbomb",
        match_id="missing-json",
        match_metadata={},
        raw_events=[{"id": "raw-1"}],
        canonical_events=[{"event_id": "1"}],
        metrics={"total_events": 1},
    )
    Path(saved["canonical_path"]).unlink()

    result = repository.delete_processed_match("statsbomb", "missing-json")

    assert result["ok"] is True
    assert repository.get_processed_match("statsbomb", "missing-json") is None
    assert not Path(saved["raw_path"]).exists()
    assert not Path(saved["metrics_path"]).exists()


def test_delete_processed_match_returns_clear_status_when_match_is_missing(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)

    result = repository.delete_processed_match("statsbomb", "unknown")

    assert result == {
        "ok": False,
        "message": "No se encontró el partido procesado statsbomb:unknown.",
    }


def test_delete_processed_match_does_not_remove_other_matches(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    repository.save_processed_match(
        provider="statsbomb",
        match_id="keep-me",
        match_metadata={"competition_name": "A"},
        raw_events=[{"id": "raw-keep"}],
        canonical_events=[{"event_id": "keep"}],
        metrics={"total_events": 1},
    )
    repository.save_processed_match(
        provider="statsbomb",
        match_id="delete-me",
        match_metadata={"competition_name": "B"},
        raw_events=[{"id": "raw-delete"}],
        canonical_events=[{"event_id": "delete"}],
        metrics={"total_events": 1},
    )

    result = repository.delete_processed_match("statsbomb", "delete-me")

    assert result["ok"] is True
    assert repository.get_processed_match("statsbomb", "delete-me") is None
    assert repository.get_processed_match("statsbomb", "keep-me") is not None
