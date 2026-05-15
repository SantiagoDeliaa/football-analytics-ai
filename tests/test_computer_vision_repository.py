from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.storage import computer_vision_repository as repository
from src.services.storage import database


def _configure_temp_storage(monkeypatch, tmp_path):
    db_path = tmp_path / "data" / "tip_event_data.sqlite"
    results_dir = tmp_path / "data" / "computer_vision" / "results"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(repository, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(repository, "RESULTS_DIR", results_dir)
    return db_path


def _build_result_payload(video_name: str = "clip.mp4"):
    return {
        "source": "api",
        "status_message": None,
        "video_name": video_name,
        "duration_seconds": 10.0,
        "total_frames": 250,
        "fps": 25.0,
        "health_summary": {"demo_mode": "stable"},
        "formations": {"team1": {"most_common": "4-3-3"}, "team2": {"most_common": "4-4-2"}},
        "metrics": {"team1": {}, "team2": {}},
        "timeline": {
            "pressure_height": {"frames": [], "team1": [], "team2": []},
            "compactness": {"frames": [], "team1": [], "team2": []},
            "offensive_width": {"frames": [], "team1": [], "team2": []},
        },
        "possession": None,
        "scouting": {
            "confidence": {
                "team1": {"label": "Alta", "score": 80},
                "team2": {"label": "Media", "score": 60},
            },
            "bullets": {"team1": ["A"], "team2": ["B"]},
        },
        "exports": {"json": True, "csv": True, "pdf": False},
        "quality_control": {},
        "speed_distance": None,
        "scouting_heatmaps": None,
        "homography_telemetry": None,
        "artifacts": {
            "video_url": "/api/static/computer-vision/clip_job-1_processed.mp4",
            "stats_json_url": "/api/static/computer-vision/clip_job-1_processed_stats.json",
            "pdf_url": None,
        },
        "warnings": [],
        "interpretation": ["ok"],
    }


def test_save_and_load_processed_video(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    video_path = tmp_path / "outputs" / "api" / "clip_job-1_processed.mp4"
    stats_path = tmp_path / "outputs" / "api" / "clip_job-1_processed_stats.json"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_text("video", encoding="utf-8")
    stats_path.write_text("{}", encoding="utf-8")

    saved = repository.save_processed_video(
        processing_id="cv-123",
        job_id="job-1",
        source_mode="upload",
        source_label="clip.mp4",
        video_name="clip.mp4",
        source_fingerprint="fingerprint-1",
        config_hash="config-1",
        status="completed",
        result_payload=_build_result_payload(),
        stats_path=stats_path,
        video_path=video_path,
    )

    assert saved["ok"] is True
    assert Path(saved["result_path"]).exists()
    assert repository.has_processed_video("fingerprint-1", "config-1") is True

    row = repository.get_processed_video("cv-123")
    assert row is not None
    assert row["video_name"] == "clip.mp4"
    assert row["video_path"] == str(video_path)

    payloads = repository.load_processed_video_payloads("cv-123")
    assert payloads is not None
    assert payloads["result"]["video_name"] == "clip.mp4"
    assert payloads["metadata"]["job_id"] == "job-1"


def test_get_processed_videos_returns_latest_rows(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)

    repository.save_processed_video(
        processing_id="cv-1",
        job_id="job-1",
        source_mode="upload",
        source_label="uno.mp4",
        video_name="uno.mp4",
        source_fingerprint="fingerprint-1",
        config_hash="config-1",
        status="completed",
        result_payload=_build_result_payload("uno.mp4"),
    )
    repository.save_processed_video(
        processing_id="cv-2",
        job_id="job-2",
        source_mode="soccernet",
        source_label="videos/dos.mp4",
        video_name="dos.mp4",
        source_fingerprint="fingerprint-2",
        config_hash="config-2",
        status="completed",
        result_payload=_build_result_payload("dos.mp4"),
    )

    rows = repository.get_processed_videos(limit=5)

    assert len(rows) == 2
    assert {row["processing_id"] for row in rows} == {"cv-1", "cv-2"}


def test_load_processed_video_payloads_returns_none_if_sidecar_is_missing(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    repository.save_processed_video(
        processing_id="cv-missing",
        job_id="job-1",
        source_mode="upload",
        source_label="clip.mp4",
        video_name="clip.mp4",
        source_fingerprint="fingerprint-1",
        config_hash="config-1",
        status="completed",
        result_payload=_build_result_payload(),
    )

    row = repository.get_processed_video("cv-missing")
    assert row is not None
    Path(row["result_path"]).unlink()

    assert repository.load_processed_video_payloads("cv-missing") is None


def test_delete_processed_video_removes_row_and_files(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    video_path = tmp_path / "outputs" / "api" / "clip_job-1_processed.mp4"
    stats_path = tmp_path / "outputs" / "api" / "clip_job-1_processed_stats.json"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_text("video", encoding="utf-8")
    stats_path.write_text("{}", encoding="utf-8")

    saved = repository.save_processed_video(
        processing_id="cv-delete",
        job_id="job-1",
        source_mode="upload",
        source_label="clip.mp4",
        video_name="clip.mp4",
        source_fingerprint="fingerprint-delete",
        config_hash="config-delete",
        status="completed",
        result_payload=_build_result_payload(),
        stats_path=stats_path,
        video_path=video_path,
    )

    result = repository.delete_processed_video("cv-delete")

    assert result["ok"] is True
    assert repository.get_processed_video("cv-delete") is None
    assert not Path(saved["result_path"]).exists()
    assert not video_path.exists()
    assert not stats_path.exists()


def test_delete_processed_video_succeeds_when_some_files_are_missing(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    video_path = tmp_path / "outputs" / "api" / "clip_job-1_processed.mp4"
    stats_path = tmp_path / "outputs" / "api" / "clip_job-1_processed_stats.json"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_text("video", encoding="utf-8")
    stats_path.write_text("{}", encoding="utf-8")

    saved = repository.save_processed_video(
        processing_id="cv-missing-file",
        job_id="job-1",
        source_mode="upload",
        source_label="clip.mp4",
        video_name="clip.mp4",
        source_fingerprint="fingerprint-missing",
        config_hash="config-missing",
        status="completed",
        result_payload=_build_result_payload(),
        stats_path=stats_path,
        video_path=video_path,
    )
    Path(saved["result_path"]).unlink()

    result = repository.delete_processed_video("cv-missing-file")

    assert result["ok"] is True
    assert repository.get_processed_video("cv-missing-file") is None
    assert not video_path.exists()
    assert not stats_path.exists()


def test_delete_processed_video_returns_clear_status_when_record_is_missing(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)

    result = repository.delete_processed_video("unknown")

    assert result == {
        "ok": False,
        "message": "No se encontró el procesamiento guardado unknown.",
    }


def test_delete_processed_video_does_not_remove_other_records(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    repository.save_processed_video(
        processing_id="cv-keep",
        job_id="job-1",
        source_mode="upload",
        source_label="keep.mp4",
        video_name="keep.mp4",
        source_fingerprint="fingerprint-keep",
        config_hash="config-keep",
        status="completed",
        result_payload=_build_result_payload("keep.mp4"),
    )
    repository.save_processed_video(
        processing_id="cv-delete",
        job_id="job-2",
        source_mode="upload",
        source_label="delete.mp4",
        video_name="delete.mp4",
        source_fingerprint="fingerprint-delete",
        config_hash="config-delete",
        status="completed",
        result_payload=_build_result_payload("delete.mp4"),
    )

    result = repository.delete_processed_video("cv-delete")

    assert result["ok"] is True
    assert repository.get_processed_video("cv-delete") is None
    assert repository.get_processed_video("cv-keep") is not None


def test_has_processed_video_uses_source_fingerprint_and_config_hash(monkeypatch, tmp_path):
    _configure_temp_storage(monkeypatch, tmp_path)
    repository.save_processed_video(
        processing_id="cv-hash",
        job_id="job-1",
        source_mode="upload",
        source_label="clip.mp4",
        video_name="clip.mp4",
        source_fingerprint="fingerprint-hash",
        config_hash="config-hash",
        status="completed",
        result_payload=_build_result_payload(),
    )

    assert repository.has_processed_video("fingerprint-hash", "config-hash") is True
    assert repository.has_processed_video("fingerprint-hash", "other-config") is False
